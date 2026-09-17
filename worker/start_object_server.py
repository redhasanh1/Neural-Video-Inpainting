#!/usr/bin/env python3
"""
SAM2 Selection Worker
Real-time mask preview using Redis pub/sub and TensorRT SAM2

Features:
- Instant mask preview via Redis pub/sub messaging
- TensorRT SAM2 model kept in memory for fast inference
- Web dashboard at http://localhost:5555
- ~50-150ms total latency per click
"""

import os

# --- network hardening -------------------------------------------------------
# Community GPU nodes advertise IPv6 via DNS but often have no route for it, so
# the first connect dies instantly with "[Errno 101] Network is unreachable".
if os.getenv('FORCE_IPV4', '1') == '1':
    try:
        import socket as _socket
        import urllib3.util.connection as _u3conn
        _u3conn.allowed_gai_family = lambda: _socket.AF_INET
        print("[NET] outbound HTTP pinned to IPv4")
    except Exception as _exc:
        print(f"[NET] could not pin outbound HTTP to IPv4: {_exc}")

# Redis here crosses Railway's public TCP proxy, which reaps idle sockets while
# the GPU renders. Without socket_timeout a BRPOP on a half-open connection
# blocks FOREVER - the process stays alive but deaf, and the tmux watchdog only
# restarts things that actually exit. health_check_interval re-pings an idle
# connection and reconnects transparently.
MRAI_REDIS_KWARGS = {
    'socket_keepalive': True,
    'socket_connect_timeout': 10,
    'socket_timeout': 60,
    'health_check_interval': 30,
    'retry_on_timeout': True,
}


def redis_from_url(url, **kwargs):
    """redis.from_url that cannot hang forever on a reaped proxy connection."""
    import redis as _redis
    opts = dict(MRAI_REDIS_KWARGS)
    opts.update(kwargs)
    try:
        from redis.retry import Retry
        from redis.backoff import ExponentialBackoff
        from redis.exceptions import ConnectionError as _CE, TimeoutError as _TE
        opts.setdefault('retry', Retry(ExponentialBackoff(cap=3, base=0.1), 3))
        opts.setdefault('retry_on_error', [_CE, _TE, ConnectionResetError])
    except Exception:
        pass
    return _redis.from_url(url, **opts)
# --- end network hardening ---------------------------------------------------


import sys
import json
import time
import base64
import threading
import argparse
from io import BytesIO
from datetime import datetime
from pathlib import Path

import redis
import numpy as np
from PIL import Image as PILImage
from flask import Flask, render_template_string, jsonify
from dotenv import load_dotenv

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Import TensorRT predictor
from sam2_trt_predictor import SAM2TensorRTPredictor

# Load environment
load_dotenv()

# Configuration
# Try environment variable, then redis_url.txt file, then localhost
REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL:
    redis_url_file = Path(__file__).parent / 'redis_url.txt'
    if redis_url_file.exists():
        REDIS_URL = redis_url_file.read_text().strip()
        print(f"[CONFIG] Loaded Redis URL from {redis_url_file}")
    else:
        REDIS_URL = 'redis://localhost:6379/0'

# TensorRT Engine Paths (use env vars for Docker, fallback to Windows paths)
ENCODER_ENGINE = os.getenv('SAM2_ENCODER_ENGINE', r"D:\watermarkz\sam2_trt_inference\engines\sam2_encoder_fp16.engine")
DECODER_ENGINE = os.getenv('SAM2_DECODER_ENGINE', r"D:\watermarkz\sam2_trt_inference\engines\sam2_decoder_fp16_dynamic.engine")

# Redis channels/lists
LIST_REQUEST = 'sam2:selection:request'  # Changed to list for proper load distribution
CHANNEL_RESPONSE_PREFIX = 'sam2:selection:response:'

# Worker configuration (set by command line args)
WORKER_ID = 0
DASHBOARD_PORT = 5555

# Global state
worker_stats = {
    'status': 'initializing',
    'worker_id': 0,
    'requests_processed': 0,
    'requests_failed': 0,
    'current_request': None,
    'last_request': None,
    'model_loaded': False,
    'uptime_start': datetime.now(),
    'avg_processing_time_ms': 0,
    'redis_connected': False
}

# Global SAM2 predictor (cached)
sam2_predictor = None
redis_client = None
redis_pubsub = None


def load_sam2_model():
    """Load SAM2 TensorRT model (once, cached globally)"""
    global sam2_predictor, worker_stats

    if sam2_predictor is not None:
        print("[MODEL] SAM2 already loaded!")
        return sam2_predictor

    print("[MODEL] Loading SAM2 TensorRT FP16 predictor...")

    if not os.path.exists(ENCODER_ENGINE):
        print(f"[ERROR] TensorRT encoder not found: {ENCODER_ENGINE}")
        return None

    if not os.path.exists(DECODER_ENGINE):
        print(f"[ERROR] TensorRT decoder not found: {DECODER_ENGINE}")
        return None

    try:
        sam2_predictor = SAM2TensorRTPredictor(ENCODER_ENGINE, DECODER_ENGINE)
        worker_stats['model_loaded'] = True
        print(f"[OK] SAM2 TensorRT loaded successfully!")
        return sam2_predictor
    except Exception as e:
        print(f"[ERROR] Failed to load SAM2: {e}")
        worker_stats['model_loaded'] = False
        return None


def connect_redis():
    """Connect to Redis"""
    global redis_client, redis_pubsub, worker_stats

    try:
        redis_client = redis_from_url(REDIS_URL, decode_responses=False)
        redis_client.ping()

        worker_stats['redis_connected'] = True
        print(f"[REDIS] Connected to {REDIS_URL[:40]}...")
        print(f"[REDIS] Listening on list: {LIST_REQUEST} (BRPOP)")
        return True
    except Exception as e:
        print(f"[ERROR] Redis connection failed: {e}")
        worker_stats['redis_connected'] = False
        return False


def process_selection_request(request_data):
    """Process SAM2 selection request and return mask"""
    global sam2_predictor, worker_stats

    try:
        start_time = time.time()

        request_id = request_data.get('request_id')
        frame_base64 = request_data.get('frame_data')
        points = request_data.get('points', [])

        print(f"[WORKER {WORKER_ID}] Processing {request_id} with {len(points)} points")
        worker_stats['current_request'] = request_id

        # Decode frame from base64
        frame_bytes = base64.b64decode(frame_base64)
        frame_img = PILImage.open(BytesIO(frame_bytes))
        frame_array = np.array(frame_img.convert('RGB'))

        # Scale points if frame dimensions differ from sent video dimensions
        frame_height, frame_width = frame_array.shape[:2]
        video_width = request_data.get('video_width')
        video_height = request_data.get('video_height')

        if video_width and video_height:
            if frame_width != video_width or frame_height != video_height:
                scale_x = frame_width / video_width
                scale_y = frame_height / video_height
                points = [{'x': p['x'] * scale_x, 'y': p['y'] * scale_y, 'label': p['label']} for p in points]
                print(f"[SAM2] Scaled points by {scale_x:.4f}x{scale_y:.4f} (video {video_width}x{video_height} -> frame {frame_width}x{frame_height})")

        # Ensure model is loaded
        predictor = load_sam2_model()
        if not predictor:
            raise Exception("SAM2 model failed to load")

        # Set image
        predictor.set_image(frame_array)

        # Prepare points and labels
        points_array = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        labels_array = np.array([p['label'] for p in points], dtype=np.int32)

        print(f"[TRT] Points: {points_array.tolist()}")
        print(f"[TRT] Labels: {labels_array.tolist()} (0=exclude, 1=include)")

        # Get mask
        mask, score = predictor.predict(points_array, labels_array)

        # Encode mask as PNG base64
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_img = PILImage.fromarray(mask_uint8, mode='L')

        buffer = BytesIO()
        mask_img.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Update stats
        worker_stats['requests_processed'] += 1
        worker_stats['last_request'] = datetime.now()

        # Update average processing time
        total = worker_stats['requests_processed']
        prev_avg = worker_stats['avg_processing_time_ms']
        worker_stats['avg_processing_time_ms'] = int((prev_avg * (total - 1) + processing_time_ms) / total)

        response = {
            'request_id': request_id,
            'status': 'success',
            'mask': mask_base64,
            'score': float(score) if score is not None else None,
            'processing_time_ms': processing_time_ms
        }

        print(f"[WORKER {WORKER_ID}] Processed {request_id} in {processing_time_ms}ms (IoU: {score:.3f})")
        worker_stats['current_request'] = None

        return response

    except Exception as e:
        print(f"[ERROR] Request processing failed: {e}")
        import traceback
        traceback.print_exc()

        worker_stats['requests_failed'] += 1
        worker_stats['current_request'] = None

        return {
            'request_id': request_data.get('request_id', 'unknown'),
            'status': 'error',
            'error': str(e)
        }


def redis_listener():
    """Listen for Redis messages and process them using BRPOP for load distribution"""
    global redis_client, worker_stats

    print(f"[WORKER {WORKER_ID}] Starting Redis listener (BRPOP)...")
    worker_stats['status'] = 'idle'

    # Pre-load SAM2 model
    load_sam2_model()

    while True:
        try:
            # Block waiting for work from the list (proper load distribution)
            # BRPOP returns (key, value) tuple or None on timeout
            result = redis_client.brpop(LIST_REQUEST, timeout=1)  # 1 second timeout

            if result:
                worker_stats['status'] = 'processing'

                # Parse request (result is tuple: (list_name, data))
                _, message_data = result
                request_data = json.loads(message_data)
                request_id = request_data.get('request_id')

                # Process request
                response = process_selection_request(request_data)

                # Publish response to specific channel (still use pub/sub for responses)
                response_channel = CHANNEL_RESPONSE_PREFIX + request_id
                redis_client.publish(response_channel, json.dumps(response))

                worker_stats['status'] = 'idle'

        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to decode message: {e}")
        except Exception as e:
            print(f"[ERROR] Worker error: {e}")
            worker_stats['status'] = 'error'
            time.sleep(0.1)  # Back off on error


# Flask app for web dashboard
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SAM2 Selection Worker</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #fff;
            padding: 40px;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #00ff00;
            font-size: 36px;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            font-size: 14px;
            margin-bottom: 40px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
        }
        .stat-label {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #00ff00;
        }
        .status-idle { color: #00ff00; }
        .status-processing { color: #ffaa00; }
        .status-error { color: #ff0000; }
        .info-section {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #222;
        }
        .info-row:last-child {
            border-bottom: none;
        }
        .info-label {
            color: #888;
        }
        .info-value {
            color: #fff;
            font-family: monospace;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge-success { background: #00ff00; color: #000; }
        .badge-error { background: #ff0000; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ SAM2 Selection Worker</h1>
        <div class="subtitle">Real-time TensorRT mask preview via Redis pub/sub</div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Status</div>
                <div class="stat-value status-{{ stats.status }}">{{ stats.status|upper }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Requests Processed</div>
                <div class="stat-value">{{ stats.requests_processed }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Requests Failed</div>
                <div class="stat-value" style="color: #ff0000;">{{ stats.requests_failed }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Processing Time</div>
                <div class="stat-value">{{ stats.avg_processing_time_ms }}ms</div>
            </div>
        </div>

        <div class="info-section">
            <h3 style="margin-top: 0; color: #00ff00;">System Info</h3>
            <div class="info-row">
                <span class="info-label">SAM2 Model</span>
                <span class="info-value">
                    {% if stats.model_loaded %}
                    <span class="badge badge-success">LOADED</span>
                    {% else %}
                    <span class="badge badge-error">NOT LOADED</span>
                    {% endif %}
                </span>
            </div>
            <div class="info-row">
                <span class="info-label">Redis Connection</span>
                <span class="info-value">
                    {% if stats.redis_connected %}
                    <span class="badge badge-success">CONNECTED</span>
                    {% else %}
                    <span class="badge badge-error">DISCONNECTED</span>
                    {% endif %}
                </span>
            </div>
            <div class="info-row">
                <span class="info-label">Current Request</span>
                <span class="info-value">{{ stats.current_request or 'None' }}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Last Request</span>
                <span class="info-value">{{ last_request }}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Uptime</span>
                <span class="info-value">{{ uptime }}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Channel</span>
                <span class="info-value">{{ channel }}</span>
            </div>
        </div>

        <div style="color: #555; font-size: 12px; text-align: center;">
            Auto-refreshing every 2 seconds | Polling Redis every 10ms
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """Web dashboard"""
    stats = worker_stats.copy()

    uptime = datetime.now() - stats['uptime_start']
    uptime_str = str(uptime).split('.')[0]  # Remove microseconds

    last_request = stats['last_request'].strftime('%H:%M:%S') if stats['last_request'] else 'Never'

    return render_template_string(
        DASHBOARD_HTML,
        stats=stats,
        uptime=uptime_str,
        last_request=last_request,
        channel=CHANNEL_REQUEST
    )


@app.route('/api/stats')
def api_stats():
    """JSON stats endpoint"""
    return jsonify(worker_stats)


def main():
    """Start worker and dashboard"""
    global WORKER_ID, DASHBOARD_PORT, worker_stats

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SAM2 Selection Worker")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker ID (0-4) for multi-worker setup")
    parser.add_argument("--port", type=int, default=None,
                        help="Dashboard port (default: 5555 + worker_id)")
    args = parser.parse_args()

    # Set worker configuration
    WORKER_ID = args.worker_id
    DASHBOARD_PORT = args.port if args.port else 5555 + WORKER_ID
    worker_stats['worker_id'] = WORKER_ID

    print("="*70)
    print(f"SAM2 SELECTION WORKER #{WORKER_ID}")
    print("="*70)
    print(f"[CONFIG] Worker ID: {WORKER_ID}")
    print(f"[CONFIG] Dashboard Port: {DASHBOARD_PORT}")
    print(f"[CONFIG] Redis: {REDIS_URL[:40]}...")
    print(f"[CONFIG] Request List: {LIST_REQUEST} (BRPOP)")
    print(f"[CONFIG] Response Channel: {CHANNEL_RESPONSE_PREFIX}<request_id>")
    print(f"[CONFIG] Encoder: {ENCODER_ENGINE}")
    print(f"[CONFIG] Decoder: {DECODER_ENGINE}")
    print(f"[CONFIG] Mode: Multi-worker (proper load distribution)")
    print("="*70)

    # Connect to Redis
    if not connect_redis():
        print("[ERROR] Failed to connect to Redis!")
        return

    # Start Redis listener thread
    listener_thread = threading.Thread(target=redis_listener, daemon=True)
    listener_thread.start()

    # Start Flask dashboard
    print(f"\n[DASHBOARD] Starting web dashboard on http://localhost:{DASHBOARD_PORT}")
    print(f"[DASHBOARD] Visit http://localhost:{DASHBOARD_PORT} to monitor worker\n")

    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    main()
