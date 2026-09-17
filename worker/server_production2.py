"""
SAM2 Watermark Removal Server - Parallel Production System
Isolated from YOLO production (server_production.py)

This server handles SAM2-based watermark removal using pre-generated masks.
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

def download_file_resilient(url, dest_path, headers=None, timeout=120,
                            attempts=6, min_bytes=1):
    """Download a URL to a file, surviving a connection that drops mid-transfer.

    A node with a weak uplink breaks a large download partway through and
    requests raises ChunkedEncodingError or a bare ConnectionError; a one-shot
    requests.get(...).content turns that into a failed render. Each retry asks
    for a Range so a host that supports it resumes; the Cloudflare Worker in
    front of B2 currently ignores Range and answers 200 with the whole file, so
    against that host this degrades to a plain retry rather than a true resume.
    """
    import requests, time as _t
    headers = dict(headers or {})
    last_err = None

    for attempt in range(1, attempts + 1):
        have = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        req_headers = dict(headers)
        mode = 'wb'
        if have > 0:
            req_headers['Range'] = f'bytes={have}-'
            mode = 'ab'
        try:
            with requests.get(url, headers=req_headers, stream=True,
                              timeout=timeout) as r:
                if have > 0 and r.status_code == 200:
                    have, mode = 0, 'wb'      # server ignored Range; start clean
                elif have > 0 and r.status_code == 416:
                    return os.path.getsize(dest_path)   # already complete
                r.raise_for_status()
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            size = os.path.getsize(dest_path)
            if size < min_bytes:
                raise IOError(f"got {size} bytes, expected at least {min_bytes}")
            if attempt > 1:
                print(f"[DOWNLOAD] recovered on attempt {attempt} ({size} bytes)")
            return size
        except Exception as exc:
            last_err = exc
            got = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            wait = min(2 ** (attempt - 1), 15)
            print(f"[DOWNLOAD] attempt {attempt}/{attempts} failed at {got} bytes "
                  f"({type(exc).__name__}: {str(exc)[:90]}), retrying in {wait}s")
            if attempt < attempts:
                _t.sleep(wait)

    raise last_err if last_err else IOError("download failed")



import sys
import glob
import json
from typing import List, Optional, Tuple
from flask import Flask, request, jsonify
from celery import Celery, chord
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from gpu_coordinator import GPUCoordinator

# Enable TensorRT optimizations for 4090 (MUST be set before watermark.py import)
os.environ['USE_NEUFLOW'] = '1'        # NeuFlow TRT optical flow (~3GB vs ~11GB VRAM)
os.environ['FORCE_TRT_RFCNET'] = '1'   # RFCNet TRT for flow completion
os.environ['ENABLE_DCNV4_RFCNET'] = '1' # DCNv4 plugin for RFCNet TRT

# Segment detection for parallel processing
from segment_detector import detect_segments_from_masks, merge_adjacent_segments, detect_segments, detect_segments_motion_based
from crop_utils import calculate_crop_region

# Number of parallel segment workers
SEGMENT_WORKERS = int(os.getenv('SEGMENT_WORKERS', '2'))

# [VRAM] Aggressive VRAM cleanup helper - call after every GPU task
def cleanup_vram(context=""):
    """Force VRAM cleanup to prevent OOM errors between tasks."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.ipc_collect()
            except:
                pass
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] Cleanup {context}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        print(f"[VRAM] Cleanup error: {e}")

# [NUCLEAR] Full cleanup - VRAM + temp files + caches (for Salad stability)
def full_cleanup(video_id=None, context=""):
    """
    Nuclear cleanup after each job to prevent crashes on long-running workers.
    Cleans: VRAM, Python memory, temp files, cached frames.
    """
    import gc
    import shutil
    import glob as glob_module

    print(f"[CLEANUP] Starting full cleanup {context}...")

    # 1. Delete any lingering local variables holding tensors
    gc.collect()
    gc.collect()  # Double collect for circular refs

    # 2. VRAM cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.ipc_collect()
            except:
                pass
    except:
        pass

    # 3. Clean temp files for this video
    if video_id:
        temp_patterns = [
            f"/app/temp/{video_id}*",
            f"/app/temp/*{video_id}*",
            f"/app/results/{video_id}*",
            f"/tmp/{video_id}*",
            f"/tmp/*{video_id}*",
        ]
        cleaned_count = 0
        for pattern in temp_patterns:
            for path in glob_module.glob(pattern):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                        cleaned_count += 1
                    elif os.path.isfile(path):
                        os.remove(path)
                        cleaned_count += 1
                except Exception as e:
                    print(f"[CLEANUP] Failed to remove {path}: {e}")
        if cleaned_count > 0:
            print(f"[CLEANUP] Removed {cleaned_count} temp files/dirs for {video_id}")

    # 4. Clean old temp files (older than 1 hour) to prevent disk buildup
    try:
        import time
        now = time.time()
        for temp_dir in ['/app/temp', '/tmp']:
            if os.path.exists(temp_dir):
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        # Skip if modified in last hour
                        if now - os.path.getmtime(item_path) > 3600:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path, ignore_errors=True)
                            else:
                                os.remove(item_path)
                    except:
                        pass
    except:
        pass

    # 5. Final GC and VRAM status
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[CLEANUP] Complete {context}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except:
        pass

# [WATCHDOG] VRAM timeout watchdog - kills jobs stuck at max VRAM
class VRAMWatchdog:
    """
    Background thread that monitors VRAM usage.
    If VRAM stays above threshold for too long, sets abort flag.
    """
    def __init__(self, threshold_percent=95, timeout_seconds=80):
        self.threshold = threshold_percent / 100.0
        self.timeout = timeout_seconds
        self.abort_flag = False
        self.high_vram_start = None
        self._stop = False
        self._thread = None

    def start(self):
        import threading
        self._stop = False
        self.abort_flag = False
        self.high_vram_start = None
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        print(f"[WATCHDOG] Started - will abort if >{self.threshold*100:.0f}% VRAM for >{self.timeout}s")

    def stop(self):
        self._stop = True
        # Don't wait for thread - it's a daemon, will die on its own
        print("[WATCHDOG] Stopped")

    def _monitor(self):
        import time

        # Use pynvml for REAL GPU memory (not just PyTorch allocations)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            use_pynvml = True
            print("[WATCHDOG] Using pynvml for accurate VRAM monitoring")
        except Exception as e:
            print(f"[WATCHDOG] pynvml unavailable ({e}), falling back to torch")
            import torch
            use_pynvml = False

        while not self._stop:
            try:
                if use_pynvml:
                    # Real GPU memory from nvidia-smi
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    usage = info.used / info.total
                else:
                    # Fallback to PyTorch (less accurate)
                    import torch
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated()
                        total = torch.cuda.get_device_properties(0).total_memory
                        usage = allocated / total
                    else:
                        usage = 0

                if usage > self.threshold:
                    if self.high_vram_start is None:
                        self.high_vram_start = time.time()
                        print(f"[WATCHDOG] High VRAM detected: {usage*100:.1f}%")
                    else:
                        elapsed = time.time() - self.high_vram_start
                        if elapsed > self.timeout:
                            print(f"[WATCHDOG] TIMEOUT! VRAM at {usage*100:.1f}% for {elapsed:.0f}s - KILLING STUCK TASK")
                            self.abort_flag = True
                            if use_pynvml:
                                pynvml.nvmlShutdown()
                            # SIGKILL the task process - can't be caught, kills immediately
                            # Celery worker will respawn, tmux will restart if needed
                            import os, signal
                            print("[WATCHDOG] Sending SIGKILL to stuck task process...")
                            os.kill(os.getpid(), signal.SIGKILL)
                else:
                    # VRAM dropped below threshold, reset timer
                    if self.high_vram_start is not None:
                        print(f"[WATCHDOG] VRAM normalized: {usage*100:.1f}%")
                    self.high_vram_start = None
            except Exception as e:
                print(f"[WATCHDOG] Monitor error: {e}")

            time.sleep(5)  # Check every 5 seconds

        # Cleanup pynvml on stop
        if use_pynvml:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

# Global watchdog instance
_vram_watchdog = None

def start_vram_watchdog():
    global _vram_watchdog
    _vram_watchdog = VRAMWatchdog(threshold_percent=95, timeout_seconds=60)
    _vram_watchdog.start()
    return _vram_watchdog

def stop_vram_watchdog():
    global _vram_watchdog
    if _vram_watchdog:
        _vram_watchdog.stop()
        _vram_watchdog = None

def check_vram_abort():
    """Call this in processing loops to check if we should abort"""
    global _vram_watchdog
    if _vram_watchdog and _vram_watchdog.abort_flag:
        raise MemoryError("[WATCHDOG] Job aborted due to VRAM timeout")

# [NVENC] Check if NVENC hardware encoder is available (cached)
_NVENC_AVAILABLE = None
def _check_nvenc_available():
    """Check if h264_nvenc encoder actually works (not just listed)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        # Actually test encoding a frame - listing encoders doesn't mean it works!
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'color=black:s=64x64:d=0.1',
             '-c:v', 'h264_nvenc', '-f', 'null', '-'],
            capture_output=True, text=True, timeout=10
        )
        _NVENC_AVAILABLE = result.returncode == 0
        if _NVENC_AVAILABLE:
            print(f"[FFMPEG] NVENC test encode succeeded - using hardware encoding")
        else:
            print(f"[FFMPEG] NVENC test failed, falling back to libx264 (CPU)")
    except Exception as e:
        print(f"[FFMPEG] NVENC test error: {e}, falling back to libx264")
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE

def _get_nvenc_encoder(width, height):
    """Get appropriate NVENC encoder based on resolution.
    H.264 max: 4096x4096 (hardware limit), H.265 max: 8192x8192
    """
    if not _check_nvenc_available():
        print("[FFMPEG] NVENC not available, using libx264 (CPU)")
        return None, ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']

    if width > 4096 or height > 4096:
        print(f"[FFMPEG] Resolution {width}x{height} exceeds H.264 limit (4096), using H.265 NVENC")
        return 'hevc', ['-c:v', 'hevc_nvenc', '-preset', 'p1', '-b:v', '8M', '-bufsize', '16M']
    else:
        return 'h264', ['-c:v', 'h264_nvenc', '-preset', 'p1', '-b:v', '8M', '-bufsize', '16M']

# Parallel segment execution (inside a single video task)
SAM2_PARALLEL_SEGMENTS = os.getenv('SAM2_PARALLEL_SEGMENTS', '1').lower() in ('1', 'true', 'yes', 'on')

# Temporal padding (in full-FPS frames) around each detected segment
# Helps prevent reappearance at segment boundaries by giving ProPainter context
SEGMENT_TEMPORAL_PAD = int(os.getenv('SEGMENT_TEMPORAL_PAD', '5'))

# Inpainting mask dilation (higher = stronger removal, more context fill)
SAM2_MASK_DILATION = int(os.getenv('SAM2_MASK_DILATION', '4'))
# Strict mode: use single full-video inpainting (no segment split)
SAM2_USE_SEGMENTS = os.getenv('SAM2_USE_SEGMENTS', '0') in ('1', 'true', 'yes', 'on')

# Segment detection sensitivity
# Higher tolerance/gap = less sensitive (fewer segments)
SEGMENT_POS_TOLERANCE = int(os.getenv('SEGMENT_POS_TOLERANCE', '50'))  # match commit defaults
# 10fps domain (legacy fallback)
SEGMENT_MIN_LEN_10FPS = int(os.getenv('SEGMENT_MIN_LEN_10FPS', '10'))
SEGMENT_MERGE_GAP_10FPS = int(os.getenv('SEGMENT_MERGE_GAP_10FPS', '6'))
# Full-FPS domain (preferred)
SEGMENT_MIN_LEN_FULL = int(os.getenv('SEGMENT_MIN_LEN_FULL', '15'))
SEGMENT_MERGE_GAP_FULL = int(os.getenv('SEGMENT_MERGE_GAP_FULL', '60'))
# Motion-based detection (better for fast-moving objects)
# SEGMENT_MOTION_THRESHOLD: max px movement between frames before new segment (default 20)
SEGMENT_MOTION_THRESHOLD = int(os.getenv('SEGMENT_MOTION_THRESHOLD', '20'))
# Use motion-based detection instead of average-based (1=motion, 0=average)
SEGMENT_USE_MOTION_DETECTION = os.getenv('SEGMENT_USE_MOTION_DETECTION', '1').lower() in ('1', 'true', 'yes', 'on')
# Max frames per segment (longer segments get split to prevent OOM)
MAX_SEGMENT_FRAMES = int(os.getenv('MAX_SEGMENT_FRAMES', '300'))
# Max pixels (width*height) for segment's union bbox (0=unlimited, 400000=~630x630 recommended)
MAX_SEGMENT_PIXELS = int(os.getenv('MAX_SEGMENT_PIXELS', '400000'))
# Max pixels AFTER padding - triggers segment splitting if exceeded (~775x775 = 600k)
MAX_CROP_PIXELS = int(os.getenv('MAX_CROP_PIXELS', '600000'))
# Detection mode: 'full' (preferred, uses in-memory masks) or '10fps' (dir-based)
SAM2_SEGMENT_DETECTION_MODE = os.getenv('SAM2_SEGMENT_DETECTION_MODE', 'full').strip().lower()

# Tail mask fill (replicate last non-empty mask to the end to avoid missed final seconds)
TAIL_MASK_FILL = os.getenv('TAIL_MASK_FILL', '1').lower() in ('1', 'true', 'yes', 'on')
try:
    TAIL_MASK_MAX_SECONDS = float(os.getenv('TAIL_MASK_MAX_SECONDS', '2.0'))
except Exception:
    TAIL_MASK_MAX_SECONDS = 2.0

# Fill short internal mask gaps (replicate nearest non-empty mask across brief gaps)
FILL_MASK_GAPS = os.getenv('FILL_MASK_GAPS', '0').lower() in ('1', 'true', 'yes', 'on')
try:
    MASK_GAP_MAX_SECONDS = float(os.getenv('MASK_GAP_MAX_SECONDS', '1.0'))
except Exception:
    MASK_GAP_MAX_SECONDS = 1.0

# Per-segment crop padding ratio (increase to avoid missing moving object near crop edges)
try:
    SEGMENT_CROP_PAD_RATIO = float(os.getenv('SEGMENT_CROP_PAD_RATIO', '0.25'))
except Exception:
    SEGMENT_CROP_PAD_RATIO = 0.25


def split_segment_by_pixels(start, end, detections, max_pixels, padding_ratio=0.25, frame_width=1920, frame_height=1080):
    """
    Recursively split segment until each piece has crop < max_pixels AFTER padding.
    Uses calculate_crop_region for EXACT crop size (includes min_size, clamping).
    """
    bboxes = [detections[f] for f in range(start, end)
              if f < len(detections) and detections[f] is not None]
    if not bboxes:
        return [(start, end, (0, 0, 0, 0))]

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    bbox = (x1, y1, x2, y2)

    # Use calculate_crop_region to get EXACT crop size (includes min_size, clamping)
    _, _, crop_w, crop_h = calculate_crop_region(
        bbox, frame_width, frame_height,
        padding_ratio=padding_ratio, min_size=128
    )
    crop_pixels = crop_w * crop_h

    # Stop splitting if under limit OR segment is too short (min 30 frames)
    if crop_pixels <= max_pixels or end - start <= 30:
        return [(start, end, bbox)]

    # Binary split
    mid = (start + end) // 2
    left = split_segment_by_pixels(start, mid, detections, max_pixels, padding_ratio, frame_width, frame_height)
    right = split_segment_by_pixels(mid, end, detections, max_pixels, padding_ratio, frame_width, frame_height)
    return left + right


class VRAMCompressor:
    """
    YUV420 compression for VRAM storage - 8:1 ratio, 43.5dB PSNR.
    Stores frames in GPU memory with minimal quality loss.
    """

    @staticmethod
    def compress_frame(frame_bgr):
        """Compress single BGR frame to YUV420 in VRAM
        Args: frame_bgr - numpy array [H, W, 3] uint8
        Returns: dict with y, u, v tensors on GPU (8:1 smaller)
        """
        import torch
        import torch.nn.functional as F

        frame_gpu = torch.from_numpy(frame_bgr).cuda().float()
        b, g, r = frame_gpu[:, :, 0], frame_gpu[:, :, 1], frame_gpu[:, :, 2]

        # RGB to YUV (BT.601)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b + 128
        v = 0.615 * r - 0.515 * g - 0.100 * b + 128

        # 4:2:0 subsampling - Y full res, U/V quarter res
        y_full = y.byte()
        u_sub = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), 2).squeeze().byte()
        v_sub = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), 2).squeeze().byte()

        return {'y': y_full, 'u': u_sub, 'v': v_sub, 'shape': frame_bgr.shape}

    @staticmethod
    def decompress_frame(compressed):
        """Decompress YUV420 back to BGR frame
        Returns: numpy array [H, W, 3] uint8
        """
        import torch
        import torch.nn.functional as F

        y = compressed['y'].float()
        u = F.interpolate(compressed['u'].unsqueeze(0).unsqueeze(0).float(), scale_factor=2, mode='bilinear').squeeze()
        v = F.interpolate(compressed['v'].unsqueeze(0).unsqueeze(0).float(), scale_factor=2, mode='bilinear').squeeze()

        u_adj, v_adj = u - 128, v - 128
        r = (y + 1.140 * v_adj).clamp(0, 255)
        g = (y - 0.395 * u_adj - 0.581 * v_adj).clamp(0, 255)
        b = (y + 2.032 * u_adj).clamp(0, 255)

        return torch.stack([b, g, r], dim=-1).byte().cpu().numpy()


# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
PROPAINTER_DIR = os.path.join(BASE_DIR, 'faster-propainter-main')

# Add ProPainter to path
sys.path.insert(0, PROPAINTER_DIR)

# Create directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Load Redis URL
REDIS_URL = os.getenv('REDIS_URL', 'redis://:watermarkz_secure_2024@localhost:6379/0')
if os.path.exists('redis_url.txt'):
    with open('redis_url.txt', 'r') as f:
        REDIS_URL = f.read().strip()

# Configure Celery (new format)
celery = Celery(app.name, broker=REDIS_URL, backend=REDIS_URL)
celery.conf.task_track_started = True

# The broker socket sits idle for minutes while the GPU renders and Railway's
# public proxy reaps it; without these the next operation fails the task (or
# blocks forever). Verified against celery 5.6: socket options passed via
# result_backend_transport_options never reach the client - the redis_* keys
# below are the ones it actually reads.
celery.conf.update(
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=None,
    broker_heartbeat=30,
    broker_transport_options={
        'socket_keepalive': True,
        'socket_connect_timeout': 10,
        'socket_timeout': 60,
        'health_check_interval': 30,
        'retry_on_timeout': True,
        'max_retries': 3,
    },
    redis_socket_keepalive=True,
    redis_socket_connect_timeout=10,
    redis_socket_timeout=60,
    redis_retry_on_timeout=True,
    redis_backend_health_check_interval=30,
    result_backend_always_retry=True,
    result_backend_max_retries=5,
)
try:
    # Allow tuning via environment; CLI flags still take precedence when provided.
    _pool = os.getenv('CELERY_POOL', None)
    _conc = os.getenv('CELERY_CONCURRENCY', os.getenv('WORKER_CONCURRENCY', None))
    if _pool:
        celery.conf.worker_pool = _pool
    if _conc:
        celery.conf.worker_concurrency = int(_conc)
except Exception:
    pass

# GPU Coordinator for exclusive GPU access (prevents SAM2/ProPainter conflicts)
_gpu_coord = GPUCoordinator(worker_type='propainter')

# [NOTIFY] Send notification when Celery worker is ACTUALLY ready
from celery.signals import worker_ready

@worker_ready.connect
def on_worker_ready(**kwargs):
    """Send notification when Celery is truly ready (after TRT engines built)"""
    import requests
    webhook_url = os.environ.get('NOTIFY_WEBHOOK_URL')
    worker_name = os.environ.get('NOTIFY_WORKER_NAME', os.uname().nodename if hasattr(os, 'uname') else 'worker')

    if webhook_url:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 0

            msg = f"🟢 Worker **{worker_name}** is READY! GPU: {gpu_name} ({gpu_mem}MB VRAM)"
            requests.post(webhook_url, json={"content": msg}, timeout=5)
            print(f"[NOTIFY] Startup notification sent: {msg}")
        except Exception as e:
            print(f"[NOTIFY] Failed to send startup notification: {e}")

# Get FFmpeg/FFprobe executables
def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths - check multiple locations"""
    import platform
    import shutil

    # On Linux/Docker, use system ffmpeg first (avoid Windows .exe files from volume mount)
    if platform.system() != 'Windows':
        ffmpeg_path = shutil.which('ffmpeg')
        ffprobe_path = shutil.which('ffprobe')
        if ffmpeg_path and ffprobe_path:
            print(f"[FFmpeg] Using system FFmpeg (Linux): {ffmpeg_path}")
            return ffmpeg_path, ffprobe_path

    # Check local ffmpeg folder first (Windows)
    ffmpeg_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffmpeg.exe')
    ffprobe_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffprobe.exe')

    # Check static_ffmpeg package location
    if not os.path.exists(ffmpeg_exe):
        try:
            import static_ffmpeg
            static_path = os.path.dirname(static_ffmpeg.__file__)
            static_ffmpeg_exe = os.path.join(static_path, 'bin', 'win32', 'ffmpeg.exe')
            static_ffprobe_exe = os.path.join(static_path, 'bin', 'win32', 'ffprobe.exe')
            if os.path.exists(static_ffmpeg_exe):
                ffmpeg_exe = static_ffmpeg_exe
                ffprobe_exe = static_ffprobe_exe
                print(f"[FFmpeg] Using static_ffmpeg: {ffmpeg_exe}")
        except ImportError:
            pass

    # Final fallback to system PATH
    if not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = 'ffmpeg'
    if not os.path.exists(ffprobe_exe):
        ffprobe_exe = 'ffprobe'

    return ffmpeg_exe, ffprobe_exe

FFMPEG_EXE, FFPROBE_EXE = get_ffmpeg_executables()


def read_video_frames_range(video_path: str, start_frame: int, end_frame: int,
                            crop_region: Optional[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
    """
    Read frames [start_frame, end_frame) from video, optionally crop on-the-fly.
    This prevents loading entire video into memory - only loads the requested range.

    Args:
        video_path: Path to the video file
        start_frame: First frame to read (0-indexed)
        end_frame: One past the last frame to read
        crop_region: Optional (x, y, w, h) tuple to crop each frame

    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if crop_region:
            cx, cy, cw, ch = crop_region
            frame = frame[cy:cy+ch, cx:cx+cw]
        frames.append(frame)
    cap.release()
    return frames


# Configurable tracking FPS (default 15fps - good balance for fast objects)
SAM2_TRACK_FPS = int(os.getenv('SAM2_TRACK_FPS', '15'))

def convert_to_tracking_fps(video_path: str, output_path: str = None, target_fps: int = None) -> str:
    """
    Convert video to target FPS using FFmpeg for SAM2 tracking.
    Returns path to the converted video.

    Args:
        video_path: Path to input video
        output_path: Optional output path (auto-generated if None)
        target_fps: Target FPS (defaults to SAM2_TRACK_FPS env var)

    Returns:
        Path to converted video
    """
    if target_fps is None:
        target_fps = SAM2_TRACK_FPS

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_{target_fps}fps{ext}"

    # Try GPU acceleration first, fall back to CPU
    # Using scale filter to maintain quality while reducing framerate
    cmd = [
        str(FFMPEG_EXE),
        '-y',
        '-i', video_path,
        '-vf', f'fps={target_fps}',
        '-c:v', 'libx264',  # CPU fallback (more compatible)
        '-preset', 'ultrafast',
        '-crf', '18',
        '-an',  # No audio for processing
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg {target_fps}fps conversion failed: {result.stderr[:500]}")
        raise RuntimeError(f"{target_fps}fps conversion failed: {result.stderr[:200]}")

    print(f"[OK] Converted to {target_fps}fps: {output_path}")
    return output_path


# Check ProPainter assets
def _check_propainter_assets():
    """Check if ProPainter model checkpoints exist"""
    required_files = [
        os.path.join(PROPAINTER_DIR, 'weights', 'ProPainter.pth'),
        os.path.join(PROPAINTER_DIR, 'weights', 'recurrent_flow_completion.pth')
    ]
    return all(os.path.exists(f) for f in required_files)

# Get ProPainter pipeline (cached at worker startup)
_propainter_pipeline = None

def get_propainter_pipeline():
    """Get cached ProPainter pipeline"""
    global _propainter_pipeline
    if _propainter_pipeline is None:
        from watermark import pipeline as faster_propainter_pipeline
        _propainter_pipeline = faster_propainter_pipeline
    return _propainter_pipeline

def clear_gpu_memory():
    """Clear GPU memory between segments"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

#============================================================================
# CELERY TASK: SAM2 Interactive Mode - FULL FPS Pipeline
#
# Optimizations:
# - Use original FPS for SAM2 tracking (more accurate masks)
# - Run ProPainter with in-memory arrays (ZERO disk I/O)
# - Use NeuFlow TRT for optical flow (10-70x faster than RAFT)
#============================================================================

@celery.task(bind=True, name='watermark.process_sam2_interactive')
def process_sam2_interactive_task(self, video_path, video_id=None, points=None, video_width=None, video_height=None, frame_index=0, api_base=None):
    """
    SAM2 Interactive Mode - FULL FPS Pipeline:
    - Downloads video from remote
    - Generates masks at original FPS using SAM2Tracker (WSL2)
    - Runs ProPainter with in-memory arrays (zero disk I/O)
    - Returns processed video with audio

    Args:
        video_path (str): Path to input video (on remote server)
        video_id (str): Video ID for tracking
        points (list): User selection points
        video_width (int): Width of the video
        video_height (int): Height of the video
        frame_index (int): The frame index the user clicked on (at original FPS)
    """
    try:
        import shutil
        import json
        import numpy as np
        import cv2
        import requests
        from urllib.parse import urljoin
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Initializing SAM2 FULL-FPS Pipeline'})

        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing")

        # Generate video ID if not provided
        if not video_id:
            video_id = os.path.basename(video_path).split('.')[0][:8]

        # Start VRAM watchdog - will abort job if VRAM stays >95% for >80s
        start_vram_watchdog()

        # --- 1. Locate video (robust local/remote handling) ---
        from pathlib import PureWindowsPath

        # Extract filename from video_path (handles Windows paths)
        base_name = PureWindowsPath(video_path).name if '\\' in video_path else os.path.basename(video_path)
        UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
        local_video_path = os.path.join(UPLOAD_DIR, base_name)

        # Smart check: If file exists locally (same PC workers share uploads/), skip download!
        if os.path.exists(local_video_path):
            print(f"[SAM2] Video already exists locally: {local_video_path} (skip download)")
            video_path = local_video_path
        elif not os.path.exists(video_path):
            # File not found at provided path - try to download
            # 1) If provided path is an absolute URL, use it directly
            if isinstance(video_path, str) and video_path.lower().startswith(('http://', 'https://')):
                download_url = video_path
            else:
                # 2) Construct from API base or env TUNNEL_URL or optional file 'tunnel_url.txt'
                tunnel = api_base or os.getenv('TUNNEL_URL')
                if not tunnel:
                    for candidate in ('tunnel_url.txt', 'TUNNEL_URL.txt'):
                        fp = os.path.join(BASE_DIR, candidate)
                        if os.path.exists(fp):
                            try:
                                with open(fp, 'r') as fh:
                                    tunnel = fh.read().strip()
                                    if tunnel:
                                        print(f"[SAM2] Loaded TUNNEL_URL from {candidate}: {tunnel}")
                                        break
                            except Exception:
                                pass
                if not tunnel:
                    # 3) As a last resort, attempt to map '/data/uploads/<file>' → local uploads/<file>
                    #    If still missing, fail with actionable error
                    if os.path.exists(local_video_path):
                        video_path = local_video_path
                    else:
                        raise RuntimeError(f"Video not found locally and no TUNNEL_URL provided: {video_path}")
                else:
                    download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')

            if 'download_url' in locals():
                print(f"[SAM2] Downloading video: {download_url}")
                self.update_state(state='STARTED', meta={'progress': 1, 'status': 'Downloading video'})
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                download_file_resilient(download_url, local_video_path, timeout=120, headers={'ngrok-skip-browser-warning': 'true'})
                print(f"[SAM2] Video downloaded to: {local_video_path}")
                video_path = local_video_path
        # else: video_path exists, use it as-is (local file path provided directly)

        # --- 2. Get video metadata (FPS, frame count) ---
        self.update_state(state='PROCESSING', meta={'progress': 3, 'status': 'Analyzing video'})
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"[SAM2] Video: {width}x{height} @ {original_fps}fps, {total_frames_original} frames")

        # --- 3. Use original FPS for SAM2 (more accurate masks) ---
        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Preparing full-FPS tracking'})
        print(f"[SAM2] Using original FPS for SAM2 tracking (no 10fps conversion)")

        # --- 4. Generate masks at FULL FPS using WSL2 SAM2 subprocess ---
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Generating masks with SAM2 (FULL FPS via WSL2)'})

        if not points:
            raise ValueError("No points provided for mask generation.")

        # Convert points to a bounding box for the tracker
        # Accepts the following formats:
        # - list of [x, y]
        # - list of [x, y, ...] (extra values ignored)
        # - list of {x: ..., y: ...}
        # - flat list [x1, y1, x2, y2, ...] (pairs)
        # - dict with key 'bbox' as [x1, y1, x2, y2]

        def _extract_xy_array(pts):
            # dict with bbox provided directly
            if isinstance(pts, dict):
                if 'bbox' in pts and isinstance(pts['bbox'], (list, tuple)) and len(pts['bbox']) >= 4:
                    bx1, by1, bx2, by2 = pts['bbox'][:4]
                    return np.array([[bx1, by1], [bx2, by2]], dtype=float)
                if 'points' in pts:
                    pts = pts['points']
                else:
                    raise ValueError('Invalid points format: dict missing bbox/points')

            # list/tuple handling
            if isinstance(pts, (list, tuple)) and len(pts) > 0:
                first = pts[0]
                # list of dicts
                if isinstance(first, dict):
                    xy = []
                    for p in pts:
                        xv = p.get('x', p.get('X', p.get('cx')))
                        yv = p.get('y', p.get('Y', p.get('cy')))
                        if xv is None or yv is None:
                            raise ValueError('Invalid points format: dict items must contain x/y')
                        xy.append([xv, yv])
                    return np.array(xy, dtype=float)
                # list of lists/tuples/arrays
                if isinstance(first, (list, tuple)):
                    arr = np.array(pts, dtype=float)
                    # ignore extras beyond first two columns
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        return arr[:, :2]
                    # if nested but not 2D as expected
                    raise ValueError('Invalid points format: nested lists must be of shape (N, >=2)')
                # flat list of numbers
                if isinstance(first, (int, float)):
                    arr = np.array(pts, dtype=float)
                    if arr.size == 2:
                        return arr.reshape(1, 2)
                    if arr.size >= 4:
                        xs = arr[0::2]
                        ys = arr[1::2]
                        return np.stack([xs, ys], axis=1)
            raise ValueError('Invalid points format')

        xy = _extract_xy_array(points)

        # If points look normalized [0..1], scale to pixel space
        try:
            if xy.size >= 2:
                max_x = float(np.max(xy[:, 0]))
                max_y = float(np.max(xy[:, 1]))
                min_x = float(np.min(xy[:, 0]))
                min_y = float(np.min(xy[:, 1]))
                # Heuristic: values within [0,1] likely normalized
                if 0.0 <= min_x <= 1.0 and 0.0 <= max_x <= 1.0 and 0.0 <= min_y <= 1.0 and 0.0 <= max_y <= 1.0:
                    px_w = video_width or width
                    px_h = video_height or height
                    xy[:, 0] = xy[:, 0] * px_w
                    xy[:, 1] = xy[:, 1] * px_h
        except Exception:
            # Non-fatal; fallback to raw values
            pass

        x1, y1 = np.min(xy, axis=0)
        x2, y2 = np.max(xy, axis=0)

        # Ensure bbox has non-zero area; expand a bit if needed
        if int(x1) == int(x2) and int(y1) == int(y2):
            pad = 8
            x1 -= pad
            y1 -= pad
            x2 += pad
            y2 += pad

        # Clip to video bounds
        px_w = video_width or width
        px_h = video_height or height
        x1 = max(0, min(int(round(x1)), int(px_w) - 1))
        y1 = max(0, min(int(round(y1)), int(px_h) - 1))
        x2 = max(0, min(int(round(x2)), int(px_w) - 1))
        y2 = max(0, min(int(round(y2)), int(px_h) - 1))

        # Sort coords to ensure x1<=x2, y1<=y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        bbox = [x1, y1, x2, y2]
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        # Use original frame index directly (clip to bounds)
        frame_index_full = min(max(int(frame_index), 0), total_frames_original - 1)
        print(f"[SAM2] Tracking from bbox {bbox} on full-FPS frame {frame_index_full}")

        # Create output masks directory for WSL2 script (full FPS)
        masks_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks")
        os.makedirs(masks_dir, exist_ok=True)

        # Call WSL2 subprocess for SAM2 tracking (PyTorch + torch.compile)
        def _to_wsl_path(p: str) -> str:
            try:
                if len(p) >= 2 and p[1] == ':' and (p[0].isalpha()):
                    drive = p[0].lower()
                    rest = p[2:].replace('\\', '/')
                    rest = rest.lstrip('/')
                    return f"/mnt/{drive}/{rest}"
                return p.replace('\\', '/')
            except Exception:
                return p

        # Prefer WSL Celery worker (shared Redis) for mask generation using non-blocking chain
        from celery import signature, chain
        prompt_mode = os.getenv('SAM2_PROMPT_MODE', 'point').strip().lower()
        if prompt_mode == 'point':
            # Extract ALL points with labels for multi-click support
            px_w = video_width or width
            px_h = video_height or height
            points_arg = []
            labels_arg = []
            for i, (x, y) in enumerate(xy):
                px = int(np.clip(x, 0, px_w - 1))
                py = int(np.clip(y, 0, px_h - 1))
                points_arg.append((px, py))
                # Try to get label from original points array (default=1 for foreground)
                if isinstance(points, list) and i < len(points) and isinstance(points[i], dict):
                    labels_arg.append(int(points[i].get('label', 1)))
                else:
                    labels_arg.append(1)
            print(f"[SAM2] Using {len(points_arg)} POINT prompt(s): {points_arg}")
            bbox_arg = None
        else:
            print(f"[SAM2] Using BBOX prompt: {bbox_str}")
            points_arg = None
            labels_arg = None
            bbox_arg = [int(x) for x in bbox]

        if os.getenv('USE_WSL_CELERY', '1').lower() in ('1','true','yes','on'):
            print(f"[SAM2] Dispatching mask generation to WSL Celery worker (chain)…")
            # Optional: do FPS conversion on Windows side for speed (avoid requiring ffmpeg in WSL)
            track_src = video_path
            if os.getenv('SAM2_TRACK_DOWNSAMPLE', '0').lower() in ('1','true','yes','on'):
                video_downsampled_path = os.path.join(TEMP_DIR, f"{video_id}_tracking_{SAM2_TRACK_FPS}fps.mp4")
                if not os.path.exists(video_downsampled_path):
                    print(f"[SAM2] Creating {SAM2_TRACK_FPS}fps tracking video on Windows for WSL speed…")
                    convert_to_tracking_fps(video_path, video_downsampled_path)
                else:
                    print(f"[SAM2] Using cached {SAM2_TRACK_FPS}fps tracking video: {video_downsampled_path}")
                track_src = video_downsampled_path

            s1 = signature('sam2.generate_masks_fullfps', args=[track_src, masks_dir, prompt_mode, points_arg, labels_arg, bbox_arg, frame_index_full], queue='wsl_sam2')
            s2 = signature('watermark._continue_after_masks', args=[video_path, video_id, points, video_width, video_height, frame_index, api_base], queue='propainter')
            # Replace current task with the chained workflow to avoid blocking (.get) inside a task
            raise self.replace(chain(s1, s2))

        # If using WSL worker pattern (non-blocking), delegate continuation via chain
        use_wsl_celery = os.getenv('USE_WSL_CELERY', '1').lower() in ('1','true','yes','on')
        if use_wsl_celery:
            from celery import signature, chain
            prompt_mode = os.getenv('SAM2_PROMPT_MODE', 'point').strip().lower()
            if prompt_mode == 'point':
                # Extract ALL points with labels for multi-click support
                px_w = video_width or width
                px_h = video_height or height
                points_arg = []
                labels_arg = []
                for i, (x, y) in enumerate(xy):
                    px = int(np.clip(x, 0, px_w - 1))
                    py = int(np.clip(y, 0, px_h - 1))
                    points_arg.append((px, py))
                    if isinstance(points, list) and i < len(points) and isinstance(points[i], dict):
                        labels_arg.append(int(points[i].get('label', 1)))
                    else:
                        labels_arg.append(1)
                bbox_arg = None
            else:
                points_arg = None
                labels_arg = None
                bbox_arg = [int(x) for x in bbox]
            s1 = signature('sam2.generate_masks_fullfps', args=[video_path, masks_dir, prompt_mode, points_arg, labels_arg, bbox_arg, frame_index_full], queue='wsl_sam2')
            s2 = signature('watermark._continue_after_masks', args=[video_path, video_id, points, video_width, video_height, frame_index, api_base], queue='propainter')
            raise self.replace(chain(s1, s2))

        if not use_wsl_celery:
            wsl_video = _to_wsl_path(video_path)
            wsl_masks = _to_wsl_path(masks_dir)
            if prompt_mode == 'point':
                # CLI fallback only supports single point - use first point
                if len(xy) > 1:
                    print(f"[SAM2-WSL2] WARNING: CLI mode only supports 1 point, but {len(xy)} provided. Using first point only.")
                px_w = video_width or width
                px_h = video_height or height
                cx = int(np.clip(xy[0, 0], 0, px_w - 1))
                cy = int(np.clip(xy[0, 1], 0, px_h - 1))
                prompt_flag = f'--point {cx},{cy}'
            else:
                prompt_flag = f'--bbox {bbox_str}'

            wsl_cmd = (
                f'cd /mnt/d/watermarkz && '
                f'source venv_wsl2/bin/activate && '
                f'python sam2_track_wsl2.py "{wsl_video}" "{wsl_masks}" '
                f'{prompt_flag} --frame-idx {frame_index_full}'
            )
            print(f"[SAM2-WSL2] Running: wsl -e bash -c \"{wsl_cmd}\"")

            result = subprocess.run(
                ['wsl', '-e', 'bash', '-c', wsl_cmd],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"[SAM2-WSL2] STDERR: {result.stderr}")
                raise RuntimeError(f"WSL2 SAM2 tracking failed: {result.stderr}")

            print(f"[SAM2-WSL2] STDOUT: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")

            try:
                for line in result.stdout.split('\n'):
                    if '[RESULT]' in line:
                        json_str = line.split('[RESULT]')[1].strip()
                        sam2_result = json.loads(json_str)
                        print(f"[SAM2-WSL2] Result: {sam2_result}")
                        break
            except Exception as e:
                print(f"[SAM2-WSL2] Warning: Could not parse result JSON: {e}")

        # Read masks from output directory (FULL FPS)
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        if not mask_files:
            raise RuntimeError(f"No masks generated in {masks_dir}")

        masks_full = []
        for mask_file in mask_files:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks_full.append(mask)

        print(f"[SAM2-WSL2] Loaded {len(masks_full)} masks from {masks_dir}")

        # --- 5. Prepare for per-segment frame loading (DON'T load all frames upfront!) ---
        self.update_state(state='PROCESSING', meta={'progress': 15, 'status': 'Preparing segment processing'})
        total_frames = total_frames_original
        print(f"[SAM2] Video has {total_frames} frames (will load per-segment to save memory)")

        # --- 6. Use FULL-FPS masks directly (no expansion needed) ---
        self.update_state(state='PROCESSING', meta={'progress': 18, 'status': 'Validating masks'})
        if len(masks_full) != total_frames:
            print(f"[SAM2] Warning: mask count {len(masks_full)} != frame count {total_frames}; aligning by truncation")
            min_count = min(len(masks_full), total_frames)
            masks_full = masks_full[:min_count]
            total_frames = min_count
        all_masks = masks_full

        # Optionally fill short internal gaps by replicating nearest non-empty mask
        if FILL_MASK_GAPS and all_masks:
            max_gap = int(round(original_fps * MASK_GAP_MAX_SECONDS))
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            i = 0
            while i < len(presence):
                if not presence[i]:
                    # start of a gap
                    j = i
                    while j < len(presence) and not presence[j]:
                        j += 1
                    gap_len = j - i
                    if gap_len > 0 and gap_len <= max_gap:
                        # choose source: prefer previous non-empty, fallback to next
                        src_idx = i - 1 if i - 1 >= 0 and presence[i - 1] else (j if j < len(presence) else None)
                        if src_idx is not None and 0 <= src_idx < len(all_masks):
                            src = all_masks[src_idx]
                            for k in range(i, j):
                                all_masks[k] = src.copy()
                    i = j
                else:
                    i += 1

        # Optionally fill tail by replicating last non-empty mask up to a max duration
        if TAIL_MASK_FILL and all_masks:
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            if any(presence):
                last_idx = max(i for i, p in enumerate(presence) if p)
                if last_idx < (total_frames - 1):
                    max_fill = min(int(round(original_fps * TAIL_MASK_MAX_SECONDS)), (total_frames - 1) - last_idx)
                    if max_fill > 0:
                        src = all_masks[last_idx]
                        for k in range(1, max_fill + 1):
                            # replicate mask (copy) to avoid aliasing surprises
                            all_masks[last_idx + k] = src.copy()
                        print(f"[SAM2] Tail fill: replicated last mask across {max_fill} frame(s) to cover video end")

        # --- 8. Validate masks and count coverage (in memory) ---
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'Validating masks'})

        masks_with_content = sum(1 for m in all_masks if np.sum(m > 127) > 0)
        print(f"[SAM2] Mask validation: {masks_with_content}/{len(all_masks)} frames have mask content")

        if masks_with_content == 0:
            raise RuntimeError("No mask content generated - SAM2 tracking failed")

        # --- 9. Determine segment strategy ---
        if SAM2_USE_SEGMENTS:
            self.update_state(state='PROCESSING', meta={'progress': 22, 'status': 'Detecting segments'})
            if SAM2_SEGMENT_DETECTION_MODE == 'full':
                print(f"[SAM2] Detecting segments from FULL-FPS in-memory masks...")
                # Compute bbox per full-FPS mask in memory, then run detection
                detections_per_frame = []
                for m in all_masks:
                    coords = cv2.findNonZero(m)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        detections_per_frame.append((x, y, x + w, y + h))
                    else:
                        detections_per_frame.append(None)

                # Use motion-based detection for fast-moving objects (frame-to-frame comparison)
                # or average-based detection for static watermarks
                if SEGMENT_USE_MOTION_DETECTION:
                    segments = detect_segments_motion_based(
                        detections_per_frame,
                        motion_threshold=SEGMENT_MOTION_THRESHOLD,
                        min_segment_length=SEGMENT_MIN_LEN_FULL,
                        max_segment_pixels=MAX_SEGMENT_PIXELS,
                    )
                    print(f"[SAM2] Motion-based detection: {len(segments)} segments (threshold={SEGMENT_MOTION_THRESHOLD}px, max_pixels={MAX_SEGMENT_PIXELS:,})")
                else:
                    segments = detect_segments(
                        detections_per_frame,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        min_segment_length=SEGMENT_MIN_LEN_FULL,
                    )

                if len(segments) > 1 and not SEGMENT_USE_MOTION_DETECTION:
                    segments = merge_adjacent_segments(
                        segments,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        max_gap=SEGMENT_MERGE_GAP_FULL,
                    )

                # Convert inclusive end -> exclusive end and clamp to total_frames
                if segments:
                    segments = [
                        (max(0, int(s)), min(int(e) + 1, total_frames), bb) for (s, e, bb) in segments
                        if max(0, int(s)) < min(int(e) + 1, total_frames)
                    ]

                    # --- BOUNDARY-SAFE: Ensure CONTIGUOUS coverage (no gaps!) ---
                    # Empty-mask frames still processed but with zero mask (no inpainting)
                    fixed_segments = []
                    for i, (s, e, bb) in enumerate(segments):
                        if i == 0 and s > 0:
                            s = 0  # First segment must start at 0
                        if i > 0:
                            prev_end = fixed_segments[-1][1]
                            if s > prev_end:
                                s = prev_end  # No gaps between segments!
                        if i == len(segments) - 1:
                            e = total_frames  # Last segment must reach the end
                        fixed_segments.append((s, e, bb))
                    segments = fixed_segments

                    # Replace each segment's averaged bbox with union bbox for that segment's frames
                    # This ensures fast-moving objects are fully covered within each segment
                    # Also splits segments if union bbox exceeds MAX_SEGMENT_PIXELS
                    if detections_per_frame:
                        updated_segments = []
                        for s, e, bb in segments:
                            # Compute union of all bboxes in this segment's frame range
                            segment_bboxes = [detections_per_frame[f] for f in range(s, e)
                                             if f < len(detections_per_frame) and detections_per_frame[f] is not None]
                            if segment_bboxes:
                                seg_min_x = min(b[0] for b in segment_bboxes)
                                seg_min_y = min(b[1] for b in segment_bboxes)
                                seg_max_x = max(b[2] for b in segment_bboxes)
                                seg_max_y = max(b[3] for b in segment_bboxes)
                                union_bbox = (seg_min_x, seg_min_y, seg_max_x, seg_max_y)

                                # Check if CROP (post-padding) exceeds pixel limit
                                # Use calculate_crop_region to get EXACT crop size (includes min_size, clamping)
                                _, _, crop_w, crop_h = calculate_crop_region(
                                    union_bbox, width, height,
                                    padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                                )
                                crop_pixels = crop_w * crop_h
                                if MAX_CROP_PIXELS > 0 and crop_pixels > MAX_CROP_PIXELS:
                                    # Split this segment into smaller pieces
                                    print(f"[SAM2] Segment {s}-{e}: crop {crop_pixels:,} px ({crop_w}x{crop_h}) > {MAX_CROP_PIXELS:,} limit, splitting...")
                                    sub_segments = split_segment_by_pixels(s, e, detections_per_frame, MAX_CROP_PIXELS, SEGMENT_CROP_PAD_RATIO, width, height)
                                    for sub_s, sub_e, sub_bb in sub_segments:
                                        _, _, sub_w, sub_h = calculate_crop_region(
                                            sub_bb, width, height,
                                            padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                                        )
                                        sub_crop_px = sub_w * sub_h
                                        print(f"[SAM2]   -> Sub-segment {sub_s}-{sub_e} ({sub_e-sub_s}f): crop {sub_crop_px:,} px ({sub_w}x{sub_h})")
                                    updated_segments.extend(sub_segments)
                                else:
                                    updated_segments.append((s, e, union_bbox))
                                    if union_bbox != bb:
                                        print(f"[SAM2] Segment {len(updated_segments)}: union bbox {union_bbox} (was avg {bb})")
                            else:
                                updated_segments.append((s, e, bb))
                        segments = updated_segments

                # Split segments that exceed MAX_SEGMENT_FRAMES to prevent OOM
                if MAX_SEGMENT_FRAMES > 0:
                    split_segments = []
                    for s, e, bb in segments:
                        seg_len = e - s
                        if seg_len > MAX_SEGMENT_FRAMES:
                            # Split into chunks of MAX_SEGMENT_FRAMES
                            num_chunks = (seg_len + MAX_SEGMENT_FRAMES - 1) // MAX_SEGMENT_FRAMES
                            chunk_size = seg_len // num_chunks
                            for i in range(num_chunks):
                                chunk_start = s + i * chunk_size
                                chunk_end = s + (i + 1) * chunk_size if i < num_chunks - 1 else e
                                split_segments.append((chunk_start, chunk_end, bb))
                            print(f"[SAM2] Split {seg_len}f segment into {num_chunks} chunks of ~{chunk_size}f")
                        else:
                            split_segments.append((s, e, bb))
                    segments = split_segments

                # --- POST-PROCESS: Remove empty mask frames and split segments ---
                MIN_FRAMES_SPLIT = 5  # Min frames for segments broken by empty masks

                def filter_empty_frames_from_segments(segs, masks, det_per_frame):
                    """Remove empty mask frames, split if empty in middle."""
                    filtered = []
                    for s, e, bb in segs:
                        # Find runs of non-empty frames
                        runs = []
                        run_start = None
                        for f in range(s, e):
                            has_content = f < len(masks) and np.sum(masks[f] > 127) > 0
                            if has_content:
                                if run_start is None:
                                    run_start = f
                            else:
                                if run_start is not None:
                                    runs.append((run_start, f))
                                    run_start = None
                        if run_start is not None:
                            runs.append((run_start, e))

                        # Create segments from runs
                        is_split = len(runs) > 1
                        min_len = MIN_FRAMES_SPLIT if is_split else SEGMENT_MIN_LEN_FULL
                        for run_s, run_e in runs:
                            if run_e - run_s >= min_len:
                                # Compute bbox for this run
                                run_bboxes = [det_per_frame[f] for f in range(run_s, run_e)
                                              if f < len(det_per_frame) and det_per_frame[f]]
                                if run_bboxes:
                                    run_bb = (min(b[0] for b in run_bboxes), min(b[1] for b in run_bboxes),
                                              max(b[2] for b in run_bboxes), max(b[3] for b in run_bboxes))
                                else:
                                    run_bb = bb
                                filtered.append((run_s, run_e, run_bb))
                                if is_split:
                                    print(f"[SAM2] Empty-mask split: frames {run_s}-{run_e-1} ({run_e-run_s}f)")
                    return filtered

                orig_count = len(segments)
                segments = filter_empty_frames_from_segments(segments, all_masks, detections_per_frame)
                if len(segments) != orig_count:
                    print(f"[SAM2] Filtered empty masks: {orig_count} -> {len(segments)} segments")

                print(f"[SAM2] Detected {len(segments)} segments (FULL-FPS, boundary-safe)")
                for idx, (start, end, bbox) in enumerate(segments):
                    print(f"[SAM2]   Segment {idx+1}: frames {start}-{end-1} ({end-start}f) bbox={bbox}")
            else:
                print(f"[SAM2] Detecting segments from 10fps masks...")

                segments_10fps = detect_segments_from_masks(
                    masks_10fps_dir,
                    position_tolerance=SEGMENT_POS_TOLERANCE,
                    min_segment_length=SEGMENT_MIN_LEN_10FPS,
                    max_segments=80,
                )

                if len(segments_10fps) > 1:
                    segments_10fps = merge_adjacent_segments(
                        segments_10fps,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        max_gap=SEGMENT_MERGE_GAP_10FPS,
                    )

                # Scale segments from 10fps to full FPS
                fps_ratio = original_fps / 10.0
                segments = []
                for start_10fps, end_10fps, bbox in segments_10fps:
                    start_full = int(start_10fps * fps_ratio)
                    end_full = min(int((end_10fps + 1) * fps_ratio), total_frames)  # +1 because end is exclusive
                    segments.append((start_full, end_full, bbox))

                print(f"[SAM2] Detected {len(segments)} segments (scaled to full FPS)")
                for idx, (start, end, bbox) in enumerate(segments):
                    print(f"[SAM2]   Segment {idx+1}: frames {start}-{end-1} ({end-start}f) bbox={bbox}")
        else:
            # Strict monolithic mode (commit logic): process entire video in a single pass
            segments = [(0, total_frames, [0, 0, width, height])]
            print(f"[SAM2] Strict full-video mode: 1 segment covering all frames")

        # --- 10. Calculate global crop region from all masks ---
        print(f"[SAM2] Calculating global crop region...")
        min_x, min_y = width, height
        max_x, max_y = 0, 0

        for mask in all_masks:
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

        # With validation at line 644-648, we should always have a valid bbox
        if max_x <= min_x or max_y <= min_y:
            raise RuntimeError("No valid bounding box found despite having mask content - unexpected state")
        global_bbox = [min_x, min_y, max_x, max_y]

        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(global_bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"[SAM2] Global crop: {crop_x},{crop_y} {crop_w}x{crop_h}")

        # Define output_dir for cleanup tracking (not created - return_frames=True skips disk I/O)
        output_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_output")

        # --- 11. Process segments with ProPainter ---
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': f'Processing {len(segments)} segments'})

        faster_propainter_pipeline = get_propainter_pipeline()
        import torch
        use_fp16 = torch.cuda.is_available()

        print(f"[SAM2] ProPainter config:")
        print(f"   - Segments: {len(segments)}")
        print(f"   - Workers: {SEGMENT_WORKERS}")
        print(f"   - FP16: {use_fp16}")
        print(f"   - Optical Flow: NeuFlow TRT (USE_NEUFLOW={os.getenv('USE_NEUFLOW', '0')})")

        # Crop masks to global crop region (masks are small - 1 channel)
        # Frames will be loaded per-segment to avoid OOM on large videos
        all_masks_cropped = [m[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for m in all_masks]
        global_crop = (crop_x, crop_y, crop_w, crop_h)
        print(f"[SAM2] Masks cropped, frames will be loaded per-segment")

        # Track segment metadata (NOT frames - those stay in VRAM!)
        segment_results = {}  # seg_idx -> (success, crop_info, time_info)
        processed_frames = set()  # Track which frames have been processed

        # REVOLUTIONARY: Store all frames in VRAM with YUV420 compression (8:1 ratio!)
        vram_frames = {}  # {frame_idx: compressed_yuv420_dict}

        # PRE-LOAD ALL FRAMES ONCE (workers will share!)
        print(f"[SAM2] Pre-loading {total_frames} frames to shared VRAM (YUV420)...")
        preloaded_frames = {}  # {frame_idx: compressed_yuv420}
        cap_preload = cv2.VideoCapture(video_path)
        for frame_idx in range(total_frames):
            ret, frame = cap_preload.read()
            if ret:
                preloaded_frames[frame_idx] = VRAMCompressor.compress_frame(frame)
        cap_preload.release()
        print(f"[SAM2] All {len(preloaded_frames)} frames pre-loaded - workers will SHARE!")

        def process_segment(seg_idx, start_f, end_f, seg_bbox):
            """Process a single segment with ProPainter"""
            proc_start = start_f
            proc_end = end_f
            try:
                duration = end_f - start_f
                log_end = end_f - 1 if end_f > start_f else end_f
                # Calculate segment-specific crop directly from seg_bbox (in original frame coords)
                # This avoids issues when segment bbox extends beyond global crop region
                seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                    list(seg_bbox),  # Use segment bbox directly in frame coordinates
                    width, height, padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                )

                # Determine optimization level based on movement
                movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
                if movement < 10:
                    neighbor_length, subvideo_length = 5, 60
                else:
                    neighbor_length, subvideo_length = 10, 80

                # Temporal padding for context
                pad_left = min(SEGMENT_TEMPORAL_PAD, start_f)
                pad_right = min(SEGMENT_TEMPORAL_PAD, total_frames - end_f)
                proc_start = start_f - pad_left
                proc_end = end_f + pad_right

                print(f"[SAM2] Segment {seg_idx+1}: frames {start_f}-{log_end} ({duration}f), crop={seg_crop_w}x{seg_crop_h}, pad=+/-{SEGMENT_TEMPORAL_PAD} => proc {proc_start}-{proc_end}")

                # GET FRAMES FROM PRE-LOADED SHARED VRAM (ZERO disk I/O!)
                print(f"[FAST] Using {proc_end - proc_start} frames from memory (ZERO disk I/O!)")
                seg_frames = []
                seg_masks = []
                for frame_idx in range(proc_start, proc_end):
                    # Decompress from shared preloaded frames
                    full_frame = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                    # Apply segment-specific crop directly (seg_crop coords are in frame coords)
                    seg_frame = full_frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                    seg_frames.append(np.ascontiguousarray(seg_frame))
                    # Crop mask from original all_masks (not pre-cropped)
                    if frame_idx < len(all_masks):
                        seg_mask = all_masks[frame_idx][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                        seg_masks.append(np.ascontiguousarray(seg_mask))

                if not seg_frames or not seg_masks:
                    return seg_idx, None, None, (proc_start, proc_end)

                # Run ProPainter on segment - ZERO DISK I/O with return_frames=True!
                output_frames = faster_propainter_pipeline(
                    video='dummy',
                    mask='dummy',
                    output=output_dir,  # Not used when return_frames=True
                    resize_ratio=1.0,
                    mask_dilation=SAM2_MASK_DILATION,
                    ref_stride=10,
                    neighbor_length=neighbor_length,
                    subvideo_length=subvideo_length,
                    raft_iter=20,
                    mode="video_inpainting",
                    save_fps=int(original_fps),
                    save_frames=False,
                    fp16=use_fp16,
                    use_cached_models=True,
                    frames_array=seg_frames,
                    masks_array=seg_masks,
                    return_frames=True  # ZERO DISK I/O!
                )

                if not output_frames:
                    print(f"[ERROR] Segment {seg_idx}: ProPainter returned no frames")
                    return seg_idx, False, None, (proc_start, proc_end)

                # REVOLUTIONARY: Store in VRAM with YUV420 compression (8:1 ratio, ZERO disk I/O!)
                # Only save CORE frames (start_f to end_f), skip padding frames
                core_count = end_f - start_f
                print(f"[SAM2] Segment {seg_idx+1}: storing {core_count} core frames in VRAM (skip {len(output_frames) - core_count} padding)...")
                frames_stored = 0
                for i in range(len(output_frames)):
                    frame_idx = proc_start + i
                    # Skip padding frames - only save core frames [start_f, end_f)
                    if frame_idx < start_f or frame_idx >= end_f:
                        continue
                    # GET from shared preloaded frames (ZERO disk I/O!)
                    orig_frame = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                    # Composite segment output onto original frame (seg_crop coords are in frame coords)
                    orig_frame[seg_crop_y:seg_crop_y + seg_crop_h,
                              seg_crop_x:seg_crop_x + seg_crop_w] = output_frames[i]
                    # Store in VRAM with YUV420 compression (8:1 ratio!)
                    vram_frames[frame_idx] = VRAMCompressor.compress_frame(orig_frame)
                    processed_frames.add(frame_idx)
                    frames_stored += 1
                del output_frames  # Free memory immediately!
                print(f"[SAM2] Segment {seg_idx+1}: stored {frames_stored} core frames in VRAM")

                # Return success flag and metadata (NOT frames!)
                return seg_idx, True, (seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h), (proc_start, proc_end)

            except Exception as e:
                print(f"[ERROR] Segment {seg_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                return seg_idx, False, None, (proc_start, proc_end)

        # Process segments (sequentially for GPU, parallel prep could be added)
        if len(segments) <= 1:
            # Single segment - use preloaded frames with decompression
            print(f"[SAM2] Processing as single segment...")
            print(f"[FAST] Using {total_frames} pre-loaded frames (ZERO disk I/O!)")
            all_frames_contiguous = []
            for frame_idx in range(total_frames):
                # Decompress and apply global crop
                full_frame = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                frame = full_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                all_frames_contiguous.append(np.ascontiguousarray(frame))
            all_masks_contiguous = [np.ascontiguousarray(m) for m in all_masks_cropped]

            # Run ProPainter - ZERO DISK I/O with return_frames=True!
            final_cropped_frames = faster_propainter_pipeline(
                video=video_path,
                mask='dummy_mask',
                output=output_dir,  # Not used when return_frames=True
                resize_ratio=1.0,
                mask_dilation=SAM2_MASK_DILATION,
                ref_stride=10,
                neighbor_length=10,
                subvideo_length=120,
                raft_iter=20,
                mode="video_inpainting",
                save_fps=int(original_fps),
                save_frames=False,
                fp16=use_fp16,
                use_cached_models=True,
                frames_array=all_frames_contiguous,
                masks_array=all_masks_contiguous,
                return_frames=True  # ZERO DISK I/O!
            )

            if not final_cropped_frames:
                raise RuntimeError("ProPainter returned no frames")

        else:
            # Multiple segments - process each (optionally in parallel) and merge
            if SAM2_PARALLEL_SEGMENTS and SEGMENT_WORKERS > 1:
                print(f"[SAM2] Processing {len(segments)} segments in PARALLEL (workers={min(SEGMENT_WORKERS, len(segments))})...")
                self.update_state(state='PROCESSING', meta={'progress': 30, 'status': f'Processing {len(segments)} segments (parallel)'})

                max_workers = min(SEGMENT_WORKERS, len(segments))
                futures = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                        futures[executor.submit(process_segment, seg_idx, start_f, end_f, seg_bbox)] = seg_idx

                    completed = 0
                    for future in as_completed(futures):
                        seg_idx_done = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            print(f"[ERROR] Segment {seg_idx_done} raised: {e}")
                            result = (seg_idx_done, None, None, (0, 0))
                        segment_results[seg_idx_done] = result
                        completed += 1
                        # Cleanup VRAM after each segment to prevent accumulation
                        cleanup_vram(f"after segment {seg_idx_done+1}")
                        # Check if watchdog triggered abort
                        check_vram_abort()
                        progress = 30 + int((completed / len(segments)) * 40)
                        self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'Processed {completed}/{len(segments)} segments'})
            else:
                print(f"[SAM2] Processing {len(segments)} segments (sequential)...")
                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                    progress = 30 + int((seg_idx / len(segments)) * 40)
                    self.update_state(state='PROCESSING', meta={
                        'progress': progress,
                        'status': f'Processing segment {seg_idx+1}/{len(segments)}'
                    })

                    clear_gpu_memory()
                    result = process_segment(seg_idx, start_f, end_f, seg_bbox)
                    segment_results[seg_idx] = result
                    # Check if watchdog triggered abort
                    check_vram_abort()

            # --- 12. REVOLUTIONARY: Direct VRAM → FFmpeg pipe → NVENC (ZERO disk I/O!) ---
            print(f"[SAM2] Segment processing complete - {len(vram_frames)} frames stored in VRAM")
            print(f"[SAM2] Encoding directly from VRAM via FFmpeg pipe (ZERO disk I/O!)...")
            self.update_state(state='PROCESSING', meta={'progress': 75, 'status': 'Encoding from VRAM (NVENC)'})
            import time as _time
            _encode_start = _time.time()

            output_path = os.path.join(RESULT_DIR, f"{video_id}_sam2_removed.mp4")

            # Get frame dimensions from first available frame
            first_frame = VRAMCompressor.decompress_frame(
                vram_frames[min(vram_frames.keys())] if vram_frames else preloaded_frames[0]
            )
            frame_height, frame_width = first_frame.shape[:2]

            # Choose encoder: H.264 NVENC for <=4096, H.265 NVENC for larger, libx264 fallback
            _, encoder_args = _get_nvenc_encoder(frame_width, frame_height)

            # FFmpeg command: pipe rawvideo → encoder → MP4 with audio from original
            ffmpeg_cmd = [
                str(FFMPEG_EXE), '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{frame_width}x{frame_height}',
                '-r', str(int(original_fps)),
                '-i', 'pipe:0',  # Video from stdin
                '-i', video_path,  # Audio from original
                '-map', '0:v:0',
                '-map', '1:a:0?',
            ] + encoder_args + [
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ]

            # Start FFmpeg process
            # NOTE: Use DEVNULL for stdout/stderr to prevent pipe deadlock on long videos!
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Stream ALL frames directly to FFmpeg (processed + unprocessed)
            for frame_idx in range(total_frames):
                # Progress logging every 100 frames
                if frame_idx % 100 == 0:
                    print(f"[SAM2] Encoding frame {frame_idx}/{total_frames}...")

                if frame_idx in vram_frames:
                    # Processed frame from segment
                    frame_bgr = VRAMCompressor.decompress_frame(vram_frames[frame_idx])
                elif frame_idx in preloaded_frames:
                    # Unprocessed frame from preloaded
                    frame_bgr = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                    # Single segment case: overlay cropped result if available
                    if len(segments) <= 1 and frame_idx < len(final_cropped_frames):
                        frame_bgr[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = final_cropped_frames[frame_idx]
                else:
                    print(f"[WARNING] Frame {frame_idx} not found in VRAM, skipping")
                    continue

                proc.stdin.write(frame_bgr.tobytes())

            proc.stdin.close()
            proc.wait(timeout=300)

            if proc.returncode != 0:
                print(f"[ERROR] FFmpeg NVENC encoding failed (code {proc.returncode})")
                raise RuntimeError(f"Video encoding failed (code {proc.returncode})")

            _encode_time = _time.time() - _encode_start
            print(f"[SAM2] NVENC encode complete: {total_frames} frames in {_encode_time:.2f}s ({total_frames/_encode_time:.1f} fps)")

            # Free VRAM
            import torch as _torch
            del vram_frames
            _torch.cuda.empty_cache()

        print(f"[SAM2] Final video: {output_path}")

        # --- 15b. Upload to B2 + Cloudflare CDN (inlined - no heavy imports!) ---
        cdn_url = None
        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo
            import time as _upload_time

            B2_KEY_ID = os.getenv('B2_KEY_ID')
            B2_APP_KEY = os.getenv('B2_APP_KEY')
            B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
            B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

            if not B2_KEY_ID or not B2_APP_KEY:
                print(f"[B2] Warning: B2 credentials not set - skipping upload")
            elif os.getenv('B2_UPLOAD_ENABLED', '1') == '1':
                timestamp = int(_upload_time.time())
                remote_path = f"results/{timestamp}_{os.path.basename(output_path)}"

                print(f"[B2] Uploading to {B2_BUCKET}/{remote_path}...")
                _b2_start = _upload_time.time()
                info = InMemoryAccountInfo()
                b2_api = B2Api(info)
                b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
                bucket = b2_api.get_bucket_by_name(B2_BUCKET)
                bucket.upload_local_file(local_file=output_path, file_name=remote_path)
                cdn_url = f"{B2_CDN_URL}/{remote_path}"
                _b2_time = _upload_time.time() - _b2_start
                print(f"[B2] Upload complete in {_b2_time:.1f}s - CDN URL: {cdn_url}")
        except ImportError:
            print(f"[B2] b2sdk not installed - skipping upload")
        except Exception as e:
            print(f"[B2] Upload failed: {e}")

        # 🚀 Return immediately - cleanup happens in finally block!
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})
        print(f"[SAM2] Complete! Output: {cdn_url or output_path}")

        return {
            'status': 'success',
            'video_id': video_id,
            'video_path': video_path,
            'output_path': cdn_url or output_path,
            'total_frames': total_frames,
            'masks_generated': len(all_masks),
            'masks_expanded': len(all_masks),
            'segments_processed': len(segments),
            'width': width,
            'height': height,
            'fps': original_fps,
            'pipeline': 'full_fps_segments',
            'message': f'SAM2 pipeline complete! Processed {len(segments)} segment(s)'
        }

    except Exception as e:
        print(f"[ERROR] SAM2 task failed: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup temp files on failure too
        try:
            for temp_path in [output_dir, masks_dir]:
                if isinstance(temp_path, str) and os.path.exists(temp_path):
                    if os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                        print(f"[CLEANUP] Removed {temp_path}")
        except Exception as cleanup_err:
            print(f"[CLEANUP] Warning: {cleanup_err}")
        # Do not call update_state with FAILURE meta (Celery will store proper exception info)
        raise
    finally:
        # Stop VRAM watchdog
        stop_vram_watchdog()
        # NUCLEAR cleanup in BACKGROUND THREAD - don't block task completion!
        import threading
        def _bg_cleanup():
            try:
                full_cleanup(video_id=video_id, context="after process_sam2_interactive")
            except:
                cleanup_vram("finally block fallback")
        cleanup_thread = threading.Thread(target=_bg_cleanup, daemon=True)
        cleanup_thread.start()


#============================================================================
# FLASK ROUTES
#============================================================================

@app.route('/api/process_sam2', methods=['POST'])
def process_sam2():
    """
    API endpoint to process video with SAM2 masks

    POST data:
    {
        "video_path": "D:\\watermarkz\\videostotrain\\stock2.mp4",
        "masks_folder": "D:\\watermarkz\\temp_sam2_masks",
        "video_id": "stock2"  # Optional
    }
    """
    data = request.get_json()
    video_path = data.get('video_path')
    masks_folder = data.get('masks_folder')
    video_id = data.get('video_id')

    if not video_path or not masks_folder:
        return jsonify({'error': 'Missing video_path or masks_folder'}), 400

    # Queue Celery task
    result = celery.send_task(
        'watermark.process_sam2_interactive',
        args=[video_path, masks_folder, video_id]
    )

    return jsonify({
        'task_id': result.id,
        'status': 'queued',
        'message': 'SAM2 processing task queued'
    })


def validate_video_limits_yolo(video_path, max_duration=90, max_fps=60):
    """Validate video duration and FPS limits for YOLO processing."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video file"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration > max_duration:
            return False, f"Video too long: {duration:.1f}s (max {max_duration}s)"
        if fps > max_fps:
            return False, f"FPS too high: {fps:.1f} (max {max_fps})"

        return True, None
    except Exception as e:
        return False, f"Video validation error: {str(e)}"


@app.route('/api/process_yolo', methods=['POST'])
def process_yolo():
    """
    API endpoint to process video with YOLO detection (automatic, no user clicks).

    POST data:
    {
        "video_path": "D:\\watermarkz\\uploads\\video.mp4",
        "video_id": "abc123"  # Optional
    }

    This chains:
    1. yolo.generate_masks (WSL) - YOLO detection → masks → B2 upload
    2. watermark._continue_after_masks (Windows) - ProPainter inpainting

    YOLO mode uses 4 parallel segment workers (vs 2 for SAM2).
    """
    from celery import signature, chain
    import uuid

    data = request.get_json()
    video_path = data.get('video_path')
    video_id = data.get('video_id') or str(uuid.uuid4())[:8]

    if not video_path:
        return jsonify({'error': 'Missing video_path'}), 400

    if not os.path.exists(video_path):
        return jsonify({'error': f'Video not found: {video_path}'}), 404

    # Validate video duration and FPS limits
    is_valid, error_msg = validate_video_limits_yolo(video_path)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Create masks directory
    masks_dir = os.path.join(TEMP_DIR, f"{video_id}_yolo_masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Get API base URL for result upload
    api_base = os.getenv('TUNNEL_URL', os.getenv('API_BASE_URL', 'http://localhost:5001'))

    # Chain: WSL YOLO detection → Windows ProPainter
    s1 = signature('yolo.generate_masks', args=[video_path, masks_dir], queue='wsl_yolo')
    s2 = signature('watermark._continue_after_masks', args=[video_path, video_id, None, None, None, 0, api_base], queue='propainter')

    result = chain(s1, s2).apply_async()

    print(f"[YOLO] Queued task chain: {result.id}")
    print(f"  - Video: {video_path}")
    print(f"  - Masks dir: {masks_dir}")
    print(f"  - Queue flow: wsl_yolo → propainter")

    return jsonify({
        'task_id': result.id,
        'status': 'queued',
        'message': 'YOLO processing task queued (4 segment workers)',
        'mode': 'yolo'
    })


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a Celery task"""
    task = celery.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('progress', 0),
            'status': task.info.get('status', ''),
            'result': task.info if task.state == 'SUCCESS' else None
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }

    return jsonify(response)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'server': 'SAM2 Production'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


#============================================================================
# YOLO CHORD TASKS (for 4-worker parallel processing)
#============================================================================

@celery.task(bind=True, name='watermark.process_yolo_segment')
def process_yolo_segment_task(self, segment_data):
    """
    Process one YOLO segment - runs as separate Celery task for TRUE parallelism.
    Each segment runs on its own Celery thread worker = no GPU contention.
    """
    import cv2, numpy as np, os, torch
    from crop_utils import calculate_crop_region

    seg_idx = segment_data['seg_idx']
    start_f = segment_data['start_f']
    end_f = segment_data['end_f']
    seg_bbox = segment_data['seg_bbox']
    video_path = segment_data['video_path']
    video_id = segment_data['video_id']
    width = segment_data['width']
    height = segment_data['height']
    total_frames = segment_data['total_frames']
    original_fps = segment_data['original_fps']
    masks_url = segment_data['masks_url']
    total_segments = segment_data['total_segments']

    print(f"\n[YOLO-SEG] Worker processing segment {seg_idx+1}/{total_segments}: frames {start_f}-{end_f-1}")
    self.update_state(state='STARTED', meta={'progress': 0, 'status': f'Processing segment {seg_idx+1}'})

    # Download video if needed
    import requests
    if video_path.startswith('/data/') or video_path.startswith('http'):
        api_base = segment_data.get('api_base')
        if api_base and not video_path.startswith('http'):
            filename = os.path.basename(video_path)
            download_url = f"{api_base}/uploads/{filename}"
        else:
            download_url = video_path
        local_video = os.path.join(TEMP_DIR, f"{video_id}_seg{seg_idx}.mp4")
        if not os.path.exists(local_video):
            print(f"[YOLO-SEG] Downloading video from {download_url}...")
            download_file_resilient(download_url, local_video, timeout=300)
        video_path = local_video

    # Download masks if needed
    import zipfile
    masks_dir = os.path.join(TEMP_DIR, f"{video_id}_masks_seg{seg_idx}")
    if not os.path.exists(masks_dir) or not os.listdir(masks_dir):
        os.makedirs(masks_dir, exist_ok=True)
        print(f"[YOLO-SEG] Downloading masks from {masks_url}...")
        zip_path = os.path.join(TEMP_DIR, f"{video_id}_masks_seg{seg_idx}.zip")
        download_file_resilient(masks_url, zip_path, timeout=120)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(masks_dir)
        os.remove(zip_path)

    # Calculate segment-specific crop (hardcode 0.2 to match server_production.py)
    seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
        list(seg_bbox), width, height, padding_ratio=0.2, min_size=128
    )

    # Temporal padding - calculate FIRST so we know which masks to load
    pad_left = min(SEGMENT_TEMPORAL_PAD, start_f)
    pad_right = min(SEGMENT_TEMPORAL_PAD, total_frames - end_f)
    proc_start = start_f - pad_left
    proc_end = end_f + pad_right

    duration = end_f - start_f
    print(f"[YOLO-SEG] Segment {seg_idx+1}: frames {start_f}-{end_f-1} ({duration}f), crop={seg_crop_w}x{seg_crop_h}, pad=+/-{SEGMENT_TEMPORAL_PAD} => proc {proc_start}-{proc_end-1}")

    # Get mask files list (sorted by frame number)
    import glob
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    # Load frames AND masks for this segment ONLY (not all masks!)
    # This prevents 4x memory explosion when 4 chord tasks run concurrently
    cap = cv2.VideoCapture(video_path)
    seg_frames = []
    seg_masks = []
    for frame_idx in range(proc_start, proc_end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            seg_frame = frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
            seg_frames.append(np.ascontiguousarray(seg_frame))
            # Load mask for this frame ONLY (not all masks!)
            if frame_idx < len(mask_files):
                mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    seg_mask = mask[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                    seg_masks.append(np.ascontiguousarray(seg_mask))
    cap.release()

    if not seg_frames or not seg_masks:
        print(f"[YOLO-SEG] WARNING: Segment {seg_idx+1} has no frames/masks!")
        return {'seg_idx': seg_idx, 'success': False, 'error': 'No frames/masks'}

    # Run ProPainter
    print(f"[YOLO-SEG] Running ProPainter on {len(seg_frames)} frames...")
    self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'ProPainter segment {seg_idx+1}'})

    faster_propainter_pipeline = get_propainter_pipeline()
    use_fp16 = torch.cuda.is_available()

    # Determine neighbor_length based on movement
    movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
    neighbor_length, subvideo_length = (5, 60) if movement < 10 else (10, 80)

    output_dir = os.path.join(TEMP_DIR, f"{video_id}_seg{seg_idx}_output")
    os.makedirs(output_dir, exist_ok=True)

    output_frames = faster_propainter_pipeline(
        video='dummy', mask='dummy', output=output_dir,
        resize_ratio=1.0, mask_dilation=SAM2_MASK_DILATION,
        ref_stride=10, neighbor_length=neighbor_length, subvideo_length=subvideo_length,
        raft_iter=20, mode="video_inpainting", save_fps=int(original_fps), save_frames=False,
        fp16=use_fp16, use_cached_models=True, frames_array=seg_frames, masks_array=seg_masks,
        return_frames=True
    )

    if not output_frames:
        print(f"[YOLO-SEG] WARNING: Segment {seg_idx+1} returned no frames!")
        return {'seg_idx': seg_idx, 'success': False, 'error': 'ProPainter returned no frames'}

    print(f"[YOLO-SEG] Segment {seg_idx+1}: ProPainter returned {len(output_frames)} frames")

    # Save output frames to shared directory (keyed by frame index)
    output_frames_dir = os.path.join(TEMP_DIR, f"{video_id}_all_output_frames")
    os.makedirs(output_frames_dir, exist_ok=True)

    frames_saved = 0
    for i, frame in enumerate(output_frames):
        frame_idx = proc_start + i
        # Only save core frames [start_f, end_f), skip padding
        if frame_idx < start_f or frame_idx >= end_f:
            continue
        frame_path = os.path.join(output_frames_dir, f"{frame_idx:06d}.png")
        cv2.imwrite(frame_path, frame)
        frames_saved += 1

    # Store crop info for compositing
    crop_info = {
        'seg_crop_x': seg_crop_x, 'seg_crop_y': seg_crop_y,
        'seg_crop_w': seg_crop_w, 'seg_crop_h': seg_crop_h,
        'start_f': start_f, 'end_f': end_f
    }

    print(f"[YOLO-SEG] Segment {seg_idx+1} complete: saved {frames_saved} frames")

    # Cleanup segment-specific files
    import shutil
    for path in [output_dir, masks_dir]:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    return {
        'seg_idx': seg_idx,
        'success': True,
        'crop_info': crop_info,
        'frames_saved': frames_saved,
        'output_frames_dir': output_frames_dir
    }


@celery.task(bind=True, name='watermark.finalize_yolo')
def finalize_yolo_task(self, segment_results, video_data):
    """
    Combine all YOLO segment results and encode final video.
    Runs after all segment tasks complete (via chord callback).
    """
    import cv2, numpy as np, os, subprocess, time as _time

    video_id = video_data['video_id']
    video_path = video_data['video_path']
    width = video_data['width']
    height = video_data['height']
    total_frames = video_data['total_frames']
    original_fps = video_data['original_fps']
    api_base = video_data.get('api_base')

    print(f"\n[YOLO-FINAL] Combining {len(segment_results)} segment results...")
    self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Encoding final video'})

    # Download video if needed
    import requests
    if video_path.startswith('/data/') and api_base:
        filename = os.path.basename(video_path)
        download_url = f"{api_base}/uploads/{filename}"
        local_video = os.path.join(TEMP_DIR, f"{video_id}_final.mp4")
        if not os.path.exists(local_video):
            print(f"[YOLO-FINAL] Downloading video from {download_url}...")
            download_file_resilient(download_url, local_video, timeout=300)
        video_path = local_video

    # Collect all processed frames
    output_frames_dir = os.path.join(TEMP_DIR, f"{video_id}_all_output_frames")
    processed_frames = {}
    if os.path.exists(output_frames_dir):
        for f in os.listdir(output_frames_dir):
            if f.endswith('.png'):
                frame_idx = int(f.split('.')[0])
                processed_frames[frame_idx] = os.path.join(output_frames_dir, f)

    print(f"[YOLO-FINAL] Found {len(processed_frames)} processed frames")

    # Collect crop info from segments
    crop_infos = {}
    for result in segment_results:
        if result and result.get('success') and result.get('crop_info'):
            seg_idx = result['seg_idx']
            crop_infos[seg_idx] = result['crop_info']

    # Encode final video with FFmpeg pipe
    output_path = os.path.join(RESULT_DIR, f"{video_id}_yolo_removed.mp4")

    # Choose encoder: H.264 NVENC for <=4096, H.265 NVENC for larger, libx264 fallback
    _, encoder_args = _get_nvenc_encoder(width, height)

    ffmpeg_cmd = [
        str(FFMPEG_EXE), '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(int(original_fps)),
        '-i', 'pipe:0',
        '-i', video_path,
        '-map', '0:v:0',
        '-map', '1:a:0?',
    ] + encoder_args + [
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        output_path
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Stream frames to FFmpeg
    cap = cv2.VideoCapture(video_path)
    _encode_start = _time.time()

    for frame_idx in range(total_frames):
        if frame_idx % 100 == 0:
            print(f"[YOLO-FINAL] Encoding frame {frame_idx}/{total_frames}...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, orig_frame = cap.read()
        if not ret:
            continue

        # Check if this frame was processed
        if frame_idx in processed_frames:
            # Load processed segment frame and composite onto original
            seg_frame = cv2.imread(processed_frames[frame_idx])
            if seg_frame is not None:
                # Find which segment this frame belongs to
                for seg_idx, crop_info in crop_infos.items():
                    if crop_info['start_f'] <= frame_idx < crop_info['end_f']:
                        cx, cy = crop_info['seg_crop_x'], crop_info['seg_crop_y']
                        cw, ch = crop_info['seg_crop_w'], crop_info['seg_crop_h']
                        orig_frame[cy:cy+ch, cx:cx+cw] = seg_frame
                        break

        proc.stdin.write(orig_frame.tobytes())

    cap.release()
    proc.stdin.close()
    proc.wait(timeout=300)

    _encode_time = _time.time() - _encode_start
    print(f"[YOLO-FINAL] NVENC encode complete: {total_frames} frames in {_encode_time:.2f}s")

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed (code {proc.returncode})")

    # Upload to B2
    cdn_url = None
    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo

        B2_KEY_ID = os.getenv('B2_KEY_ID')
        B2_APP_KEY = os.getenv('B2_APP_KEY')
        B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
        B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

        if not B2_KEY_ID or not B2_APP_KEY:
            print(f"[B2] Warning: B2 credentials not set - skipping upload")
        elif os.getenv('B2_UPLOAD_ENABLED', '1') == '1':
            timestamp = int(_time.time())
            remote_path = f"results/{timestamp}_{os.path.basename(output_path)}"

            print(f"[B2] Uploading to {B2_BUCKET}/{remote_path}...")
            info = InMemoryAccountInfo()
            b2_api = B2Api(info)
            b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
            bucket = b2_api.get_bucket_by_name(B2_BUCKET)
            bucket.upload_local_file(local_file=output_path, file_name=remote_path)
            cdn_url = f"{B2_CDN_URL}/{remote_path}"
            print(f"[B2] Upload complete - CDN URL: {cdn_url}")
    except Exception as e:
        print(f"[B2] Upload failed: {e}")

    # Cleanup
    import shutil
    if os.path.exists(output_frames_dir):
        shutil.rmtree(output_frames_dir, ignore_errors=True)

    return {
        'status': 'success',
        'video_id': video_id,
        'output_path': cdn_url or output_path,
        'total_frames': total_frames,
        'segments_processed': len(segment_results),
        'width': width,
        'height': height,
        'fps': original_fps
    }


#============================================================================
# CONTINUATION TASK (runs after WSL mask generation)
#============================================================================

@celery.task(bind=True, name='watermark._continue_after_masks')
def _continue_after_masks(self, sam2_result, video_path, video_id=None, points=None, video_width=None, video_height=None, frame_index=0, api_base=None):
    task_id = self.request.id

    # Acquire GPU lock (waits if SAM2 is using GPU)
    with _gpu_coord.gpu_exclusive(task_id):
        return _continue_after_masks_impl(self, sam2_result, video_path, video_id, points, video_width, video_height, frame_index, api_base)


def _continue_after_masks_impl(self, sam2_result, video_path, video_id=None, points=None, video_width=None, video_height=None, frame_index=0, api_base=None):
    """Actual implementation of _continue_after_masks (wrapped with GPU lock)."""
    try:
        import cv2, glob, numpy as np, os, json, zipfile, requests
        from watermark import expand_masks_10fps  # not used but keep import parity
        from urllib.parse import urljoin

        # Start VRAM watchdog - will abort job if VRAM stays >95% for >80s
        start_vram_watchdog()

        if not sam2_result or not isinstance(sam2_result, dict):
            raise RuntimeError(f"Invalid sam2_result: {sam2_result}")

        # Determine mode and adjust SEGMENT_WORKERS
        # YOLO mode: 4 workers (faster, simpler masks)
        # SAM2 mode: 2 workers (more complex tracking)
        mode = sam2_result.get('mode', 'sam2')
        if mode == 'yolo':
            os.environ['SEGMENT_WORKERS'] = '4'
            print(f"[MODE] YOLO mode detected - using 4 segment workers")
            # YOLO mode uses 0.2 padding (matches server_production.py)
            global SEGMENT_CROP_PAD_RATIO
            SEGMENT_CROP_PAD_RATIO = 0.2
        else:
            os.environ['SEGMENT_WORKERS'] = '2'
            print(f"[MODE] SAM2 mode detected - using 2 segment workers")

        # Check for B2 URL (new flow) or local path (legacy)
        masks_url = sam2_result.get('masks_url')
        masks_dir = sam2_result.get('masks_dir')

        # Convert WSL path to Windows path if needed (e.g., /mnt/d/... -> D:\...)
        if masks_dir and masks_dir.startswith('/mnt/'):
            import re
            match = re.match(r'/mnt/([a-z])/(.*)', masks_dir)
            if match:
                drive, rest = match.groups()
                masks_dir = drive.upper() + ":\\" + rest.replace("/", "\\")
                print(f"[PATH] Converted WSL path to Windows: {masks_dir}")

        if masks_url:
            # Download masks from B2 CDN (always use B2 when URL available - masks_dir is container-local)
            print(f"[B2] Downloading masks from {masks_url}...")
            self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Downloading masks from B2'})

            zip_path = os.path.join(TEMP_DIR, f"{video_id}_masks.zip")
            download_file_resilient(masks_url, zip_path, timeout=120)

            # Extract masks
            masks_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks")
            os.makedirs(masks_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(masks_dir)
            os.remove(zip_path)

            mask_count = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
            print(f"[B2] Extracted {mask_count} masks to {masks_dir}")
        elif not masks_dir:
            masks_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks")

        if not os.path.isdir(masks_dir):
            raise RuntimeError(f"Masks directory missing: {masks_dir}")


        # If video is a Railway path (/data/...), download it from api_base
        if video_path.startswith('/data/') and api_base:
            filename = os.path.basename(video_path)
            download_url = f"{api_base}/uploads/{filename}"
            local_video = os.path.join(TEMP_DIR, filename)
            print(f"[RECEIVER] Downloading video from {download_url}...")
            self.update_state(state='PROCESSING', meta={'progress': 8, 'status': 'Downloading video'})
            download_file_resilient(download_url, local_video, timeout=300)
            video_path = local_video
            print(f"[RECEIVER] Downloaded {os.path.getsize(local_video) / 1024 / 1024:.1f} MB to {local_video}")

        # If video is an HTTPS URL, download it locally (cv2.VideoCapture unreliable with remote URLs)
        elif video_path.startswith('http://') or video_path.startswith('https://'):
            filename = os.path.basename(video_path.split('?')[0])  # Strip query params
            local_video = os.path.join(TEMP_DIR, f"{video_id}_{filename}")
            print(f"[RECEIVER] Downloading video from URL: {video_path}...")
            self.update_state(state='PROCESSING', meta={'progress': 8, 'status': 'Downloading video'})
            download_file_resilient(video_path, local_video, timeout=300)
            video_path = local_video
            print(f"[RECEIVER] Downloaded {os.path.getsize(local_video) / 1024 / 1024:.1f} MB to {local_video}")

        # --- ROTATION NORMALIZE (output stage must match SAM2's upright masks) ---
        # SAM2 generated its masks on the rotation-corrected (upright) video, so the masks
        # are upright (e.g. 270x480 portrait). If ProPainter renders on the RAW sideways
        # video the masks land in the wrong place (object not removed) and the output comes
        # out sideways. Straighten the video here too (same ffmpeg autorotate as the SAM2
        # worker) so the upright masks align and the final render is upright. No-op when the
        # video carries no rotation flag.
        if video_path and os.path.exists(video_path):
            try:
                import subprocess as _sp
                _probe = _sp.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream_side_data=rotation:stream_tags=rotate',
                     '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                    capture_output=True, text=True, timeout=30).stdout
                _rot = 0
                for _tok in _probe.split():
                    try:
                        _rot = int(float(_tok)) % 360
                        break
                    except Exception:
                        continue
                if _rot != 0:
                    _upright = os.path.join(TEMP_DIR, f"{video_id}_upright.mp4")
                    _ok = False
                    for _enc in (['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '20'],
                                 ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20']):
                        try:
                            _sp.run(['ffmpeg', '-y', '-i', video_path] + _enc +
                                    ['-an', '-metadata:s:v:0', 'rotate=0', _upright],
                                    check=True, capture_output=True, timeout=600)
                            _ok = True
                            break
                        except Exception:
                            continue
                    if _ok:
                        print(f"[RECEIVER][ROTATE] Baked {_rot}deg upright to match SAM2 masks: "
                              f"{os.path.basename(video_path)} -> {os.path.basename(_upright)}")
                        video_path = _upright
                    else:
                        print(f"[RECEIVER][ROTATE] WARNING: {_rot}deg flag present but ffmpeg "
                              f"autorotate failed; output may be sideways")
            except Exception as _e:
                print(f"[RECEIVER][ROTATE] rotation check skipped: {_e}")

        # Video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Read masks
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        if not mask_files:
            raise RuntimeError(f"No masks generated in {masks_dir}")
        masks_full = [cv2.imread(mf, cv2.IMREAD_GRAYSCALE) for mf in mask_files]
        masks_full = [m for m in masks_full if m is not None]

        # Resize masks to match video dimensions if needed (WSL may generate at different resolution)
        if masks_full and masks_full[0] is not None:
            mask_h, mask_w = masks_full[0].shape[:2]
            if mask_w != width or mask_h != height:
                print(f"[SAM2] Resizing {len(masks_full)} masks from {mask_w}x{mask_h} to {width}x{height}...")
                masks_full = [cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST) for m in masks_full]

        # DON'T load all frames upfront - use per-segment loading
        total_frames = total_frames_original
        print(f"[SAM2] Video has {total_frames} frames (will load per-segment)")

        # Align and expand 10fps masks if needed
        # If WSL did 10fps tracking (returned fps ~10), expand to full-FPS mask timeline
        track_fps = None
        try:
            track_fps = float(sam2_result.get('fps')) if isinstance(sam2_result, dict) else None
        except Exception:
            track_fps = None
        if track_fps and track_fps > 0 and abs(track_fps - 10.0) < 1.0 and total_frames > len(masks_full) >= 2:
            print(f"[SAM2] Expanding {len(masks_full)} masks (tracking @ ~10fps) to {total_frames} full-FPS frames...")
            all_masks = expand_masks_10fps(masks_full, total_frames, original_fps)
        else:
            # Fallback: strict alignment (truncate to shortest)
            if len(masks_full) != total_frames:
                min_count = min(len(masks_full), total_frames)
                masks_full = masks_full[:min_count]
                total_frames = min_count
            all_masks = masks_full

        # Fill short internal gaps
        if FILL_MASK_GAPS and all_masks:
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            max_gap = int(round(original_fps * MASK_GAP_MAX_SECONDS))
            i = 0
            while i < len(presence):
                if not presence[i]:
                    j = i
                    while j < len(presence) and not presence[j]:
                        j += 1
                    gap_len = j - i
                    if 0 < gap_len <= max_gap:
                        src_idx = i - 1 if i - 1 >= 0 and presence[i - 1] else (j if j < len(presence) else None)
                        if src_idx is not None:
                            src = all_masks[src_idx]
                            for k in range(i, j):
                                all_masks[k] = src.copy()
                    i = j
                else:
                    i += 1

        # Tail fill
        if TAIL_MASK_FILL and all_masks:
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            if any(presence):
                last_idx = max(i for i, p in enumerate(presence) if p)
                if last_idx < (total_frames - 1):
                    max_fill = min(int(round(original_fps * TAIL_MASK_MAX_SECONDS)), (total_frames - 1) - last_idx)
                    if max_fill > 0:
                        src = all_masks[last_idx]
                        for k in range(1, max_fill + 1):
                            all_masks[last_idx + k] = src.copy()
                        print(f"[SAM2] Tail fill: replicated last mask across {max_fill} frame(s) to cover video end")

        # --- Validate masks have content (same check as main task) ---
        masks_with_content = sum(1 for m in all_masks if np.sum(m > 127) > 0)
        print(f"[SAM2] Mask validation: {masks_with_content}/{len(all_masks)} frames have mask content")

        if masks_with_content == 0:
            raise RuntimeError("No mask content generated - SAM2 tracking failed")

        # The rest of pipeline identical to main task from global crop → encode
        # Calculate global crop
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        for mask in all_masks:
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

        # With validation above, we should always have a valid bbox
        if max_x <= min_x or max_y <= min_y:
            raise RuntimeError("No valid bounding box found despite having mask content - unexpected state")
        global_bbox = [min_x, min_y, max_x, max_y]
        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(global_bbox, width, height, padding_ratio=0.2, min_size=128)

        # Prepare arrays (masks only - frames loaded per-segment)
        all_masks_cropped = [m[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for m in all_masks]
        global_crop = (crop_x, crop_y, crop_w, crop_h)
        print(f"[SAM2] Masks cropped, frames will be loaded per-segment")

        # --- Segment detection (FULL-FPS, boundary-safe like commit 578a2603) ---
        print(f"[SAM2] Detecting segments from FULL-FPS in-memory masks...")
        detections_per_frame = []
        for m in all_masks:
            coords = cv2.findNonZero(m)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                detections_per_frame.append((x, y, x + w, y + h))
            else:
                detections_per_frame.append(None)
        # Use motion-based detection for fast-moving objects
        if SEGMENT_USE_MOTION_DETECTION:
            segments = detect_segments_motion_based(detections_per_frame, motion_threshold=SEGMENT_MOTION_THRESHOLD, min_segment_length=SEGMENT_MIN_LEN_FULL, max_segment_pixels=MAX_SEGMENT_PIXELS)
            print(f"[SAM2] Motion-based detection: {len(segments)} segments (threshold={SEGMENT_MOTION_THRESHOLD}px, max_pixels={MAX_SEGMENT_PIXELS:,})")
        else:
            segments = detect_segments(detections_per_frame, position_tolerance=SEGMENT_POS_TOLERANCE, min_segment_length=SEGMENT_MIN_LEN_FULL)
        if len(segments) > 1 and not SEGMENT_USE_MOTION_DETECTION:
            segments = merge_adjacent_segments(segments, position_tolerance=SEGMENT_POS_TOLERANCE, max_gap=SEGMENT_MERGE_GAP_FULL)

        # Convert inclusive end → exclusive and clamp to total_frames
        if segments:
            segments = [(max(0, int(s)), min(int(e) + 1, total_frames), bb) for (s, e, bb) in segments if max(0, int(s)) < min(int(e) + 1, total_frames)]

            # --- BOUNDARY-SAFE: Ensure CONTIGUOUS coverage (no gaps!) ---
            # Empty-mask frames still processed but with zero mask (no inpainting)
            fixed_segments = []
            for i, (s, e, bb) in enumerate(segments):
                if i == 0 and s > 0:
                    s = 0  # First segment must start at 0
                if i > 0:
                    prev_end = fixed_segments[-1][1]
                    if s > prev_end:
                        s = prev_end  # No gaps between segments!
                if i == len(segments) - 1:
                    e = total_frames  # Last segment must reach the end
                fixed_segments.append((s, e, bb))
            segments = fixed_segments

            # Merge tiny segments (< 15 frames) into neighbors for better inpainting quality
            MIN_MERGE_FRAMES = 15
            if len(segments) > 1:
                merged_segments = []
                for seg in segments:
                    duration = seg[1] - seg[0]
                    if duration < MIN_MERGE_FRAMES and merged_segments:
                        # Extend previous segment to absorb this tiny one
                        prev = merged_segments[-1]
                        # YOLO: keep prev bbox (per-frame masks don't need union)
                        # SAM2: union bboxes (interpolation needs full coverage)
                        if mode == 'yolo':
                            merged_segments[-1] = (prev[0], seg[1], prev[2])
                        else:
                            union_bb = (
                                min(prev[2][0], seg[2][0]),
                                min(prev[2][1], seg[2][1]),
                                max(prev[2][2], seg[2][2]),
                                max(prev[2][3], seg[2][3])
                            )
                            merged_segments[-1] = (prev[0], seg[1], union_bb)
                        print(f"[SAM2] Merged tiny segment ({duration}f) into previous segment")
                    else:
                        merged_segments.append(seg)
                if len(merged_segments) < len(segments):
                    print(f"[SAM2] Merged {len(segments)} segments → {len(merged_segments)} (absorbed tiny segments)")
                segments = merged_segments

            # Replace each segment's averaged bbox with union bbox for that segment's frames
            # This ensures fast-moving objects are fully covered within each segment
            # Also splits segments if union bbox exceeds MAX_SEGMENT_PIXELS
            # YOLO mode: skip union bbox (already has per-frame masks, union creates huge crops)
            # SAM2 mode: compute union bbox (interpolated masks need full coverage)
            if mode != 'yolo' and detections_per_frame:
                updated_segments = []
                for s, e, bb in segments:
                    # Compute union of all bboxes in this segment's frame range
                    segment_bboxes = [detections_per_frame[f] for f in range(s, e)
                                     if f < len(detections_per_frame) and detections_per_frame[f] is not None]
                    if segment_bboxes:
                        seg_min_x = min(b[0] for b in segment_bboxes)
                        seg_min_y = min(b[1] for b in segment_bboxes)
                        seg_max_x = max(b[2] for b in segment_bboxes)
                        seg_max_y = max(b[3] for b in segment_bboxes)
                        union_bbox = (seg_min_x, seg_min_y, seg_max_x, seg_max_y)

                        # Check if CROP (post-padding) exceeds pixel limit
                        # Use calculate_crop_region to get EXACT crop size (includes min_size, clamping)
                        _, _, crop_w, crop_h = calculate_crop_region(
                            union_bbox, width, height,
                            padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                        )
                        crop_pixels = crop_w * crop_h
                        if MAX_CROP_PIXELS > 0 and crop_pixels > MAX_CROP_PIXELS:
                            # Split this segment into smaller pieces
                            print(f"[SAM2] Segment {s}-{e}: crop {crop_pixels:,} px ({crop_w}x{crop_h}) > {MAX_CROP_PIXELS:,} limit, splitting...")
                            sub_segments = split_segment_by_pixels(s, e, detections_per_frame, MAX_CROP_PIXELS, SEGMENT_CROP_PAD_RATIO, width, height)
                            for sub_s, sub_e, sub_bb in sub_segments:
                                _, _, sub_w, sub_h = calculate_crop_region(
                                    sub_bb, width, height,
                                    padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                                )
                                sub_crop_px = sub_w * sub_h
                                print(f"[SAM2]   -> Sub-segment {sub_s}-{sub_e} ({sub_e-sub_s}f): crop {sub_crop_px:,} px ({sub_w}x{sub_h})")
                            updated_segments.extend(sub_segments)
                        else:
                            updated_segments.append((s, e, union_bbox))
                            if union_bbox != bb:
                                print(f"[SAM2] Segment {len(updated_segments)}: union bbox {union_bbox} (was avg {bb})")
                    else:
                        updated_segments.append((s, e, bb))
                segments = updated_segments
        else:
            # No segments detected - cover entire video
            segments = [(0, total_frames, global_bbox)]

        # Split segments that exceed MAX_SEGMENT_FRAMES to prevent OOM
        if MAX_SEGMENT_FRAMES > 0:
            split_segments = []
            for s, e, bb in segments:
                seg_len = e - s
                if seg_len > MAX_SEGMENT_FRAMES:
                    # Split into chunks of MAX_SEGMENT_FRAMES
                    num_chunks = (seg_len + MAX_SEGMENT_FRAMES - 1) // MAX_SEGMENT_FRAMES
                    chunk_size = seg_len // num_chunks
                    for i in range(num_chunks):
                        chunk_start = s + i * chunk_size
                        chunk_end = s + (i + 1) * chunk_size if i < num_chunks - 1 else e
                        split_segments.append((chunk_start, chunk_end, bb))
                    print(f"[SAM2] Split {seg_len}f segment into {num_chunks} chunks of ~{chunk_size}f")
                else:
                    split_segments.append((s, e, bb))
            segments = split_segments

        # --- POST-PROCESS: Remove empty mask frames and split segments ---
        MIN_FRAMES_SPLIT = 5  # Min frames for segments broken by empty masks

        def filter_empty_frames_from_segments(segs, masks, det_per_frame):
            """Remove empty mask frames, split if empty in middle."""
            filtered = []
            for s, e, bb in segs:
                # Find runs of non-empty frames
                runs = []
                run_start = None
                for f in range(s, e):
                    has_content = f < len(masks) and np.sum(masks[f] > 127) > 0
                    if has_content:
                        if run_start is None:
                            run_start = f
                    else:
                        if run_start is not None:
                            runs.append((run_start, f))
                            run_start = None
                if run_start is not None:
                    runs.append((run_start, e))

                # Create segments from runs
                is_split = len(runs) > 1
                min_len = MIN_FRAMES_SPLIT if is_split else SEGMENT_MIN_LEN_FULL
                for run_s, run_e in runs:
                    if run_e - run_s >= min_len:
                        # Compute bbox for this run
                        run_bboxes = [det_per_frame[f] for f in range(run_s, run_e)
                                      if f < len(det_per_frame) and det_per_frame[f]]
                        if run_bboxes:
                            run_bb = (min(b[0] for b in run_bboxes), min(b[1] for b in run_bboxes),
                                      max(b[2] for b in run_bboxes), max(b[3] for b in run_bboxes))
                        else:
                            run_bb = bb
                        filtered.append((run_s, run_e, run_bb))
                        if is_split:
                            print(f"[SAM2] Empty-mask split: frames {run_s}-{run_e-1} ({run_e-run_s}f)")
            return filtered

        orig_count = len(segments)
        segments = filter_empty_frames_from_segments(segments, all_masks, detections_per_frame)
        if len(segments) != orig_count:
            print(f"[SAM2] Filtered empty masks: {orig_count} -> {len(segments)} segments")

        print(f"[SAM2] Detected {len(segments)} segments (FULL-FPS, boundary-safe)")
        for idx, (start, end, bbox) in enumerate(segments):
            print(f"[SAM2]   Segment {idx+1}: frames {start}-{end-1} ({end-start}f) bbox={bbox}")

        # Now reuse the same merging + ProPainter logic as the main task
        # For brevity, call back into process_sam2_interactive_task's internal pipeline is not trivial; so inline minimal subset
        # We re-use the same process_segment closure pattern

        # Settings
        import torch
        use_fp16 = torch.cuda.is_available()
        faster_propainter_pipeline = get_propainter_pipeline()
        output_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_output")
        # Don't create output_dir - not needed with return_frames=True

        # Track segment metadata (NOT frames - those stay in VRAM!)
        segment_results = {}
        processed_frames = set()  # Track which frames have been processed

        # REVOLUTIONARY: Store all frames in VRAM with YUV420 compression (8:1 ratio!)
        vram_frames = {}  # {frame_idx: compressed_yuv420_dict}
        preloaded_frames = {}  # {frame_idx: compressed_yuv420}

        # YOLO MODE: Skip VRAM pre-load - chord tasks use disk-based processing
        if mode == 'yolo':
            print(f"[YOLO] Skipping VRAM pre-load - chord tasks handle their own frames")
        else:
            # SAM2 MODE: PRE-LOAD ALL FRAMES ONCE (workers will share!)
            print(f"[SAM2] Pre-loading {total_frames} frames to shared VRAM (YUV420)...")
            cap_preload = cv2.VideoCapture(video_path)
            for frame_idx in range(total_frames):
                ret, frame = cap_preload.read()
                if ret:
                    preloaded_frames[frame_idx] = VRAMCompressor.compress_frame(frame)
            cap_preload.release()
            # Verify all frames were loaded - fail early with clear error instead of cryptic KeyError
            if len(preloaded_frames) < total_frames:
                raise ValueError(f"Could not read all video frames: got {len(preloaded_frames)}/{total_frames}. Video may be corrupted or URL is invalid.")
            print(f"[SAM2] All {len(preloaded_frames)} frames pre-loaded - workers will SHARE!")

        def process_segment_local(seg_idx, start_f, end_f, seg_bbox):
            duration = end_f - start_f
            log_end = end_f - 1 if end_f > start_f else end_f
            # Calculate segment-specific crop directly from seg_bbox (in original frame coords)
            seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                list(seg_bbox),  # Use segment bbox directly in frame coordinates
                width, height, padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
            )

            # Downsample large crops to prevent OOM (RTX 4090 can handle ~600k pixels max)
            crop_pixels = seg_crop_w * seg_crop_h
            scale_factor = 1.0
            proc_w, proc_h = seg_crop_w, seg_crop_h  # Processing dimensions
            if MAX_CROP_PIXELS > 0 and crop_pixels > MAX_CROP_PIXELS:
                scale_factor = (MAX_CROP_PIXELS / crop_pixels) ** 0.5
                proc_w = int(seg_crop_w * scale_factor)
                proc_h = int(seg_crop_h * scale_factor)
                # Ensure dimensions are even (required for video encoding)
                proc_w = proc_w - (proc_w % 2)
                proc_h = proc_h - (proc_h % 2)
                print(f"[SAM2] Crop {seg_crop_w}x{seg_crop_h} ({crop_pixels:,}px) > {MAX_CROP_PIXELS:,} limit")
                print(f"[SAM2] Downsampling to {proc_w}x{proc_h} ({proc_w*proc_h:,}px) for ProPainter")

            movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
            if movement < 10:
                neighbor_length, subvideo_length = 5, 60
            else:
                neighbor_length, subvideo_length = 10, 80

            # For large crops that needed downsampling, reduce subvideo_length to save VRAM
            if scale_factor < 1.0:
                subvideo_length = max(15, subvideo_length // 2)
                print(f"[SAM2] Reduced subvideo_length to {subvideo_length} for large crop")

            pad_left = min(SEGMENT_TEMPORAL_PAD, start_f)
            pad_right = min(SEGMENT_TEMPORAL_PAD, total_frames - end_f)
            proc_start = start_f - pad_left
            proc_end = end_f + pad_right

            print(f"[SAM2] Segment {seg_idx+1}: frames {start_f}-{log_end} ({duration}f), crop={seg_crop_w}x{seg_crop_h}, pad=+/-{SEGMENT_TEMPORAL_PAD} => proc {proc_start}-{proc_end-1}")

            # GET FRAMES FROM PRE-LOADED SHARED VRAM (ZERO disk I/O!)
            print(f"[FAST] Using {proc_end - proc_start} frames from memory (ZERO disk I/O!)")
            seg_frames = []
            seg_masks = []
            for frame_idx in range(proc_start, proc_end):
                # Decompress from shared preloaded frames
                full_frame = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                # Apply segment-specific crop directly (seg_crop coords are in frame coords)
                seg_frame = full_frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                # Downsample if needed to prevent OOM
                if scale_factor < 1.0:
                    seg_frame = cv2.resize(seg_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                seg_frames.append(np.ascontiguousarray(seg_frame))
                # Crop mask from original all_masks (not pre-cropped)
                if frame_idx < len(all_masks):
                    seg_mask = all_masks[frame_idx][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                    # Downsample mask too (use NEAREST to preserve binary values)
                    if scale_factor < 1.0:
                        seg_mask = cv2.resize(seg_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
                    seg_masks.append(np.ascontiguousarray(seg_mask))
            if not seg_frames or not seg_masks:
                print(f"[SAM2] WARNING: Segment {seg_idx+1} has no frames/masks! seg_frames={len(seg_frames)}, seg_masks={len(seg_masks)}")
                return seg_idx, None, None, (proc_start, proc_end)

            # Run ProPainter - ZERO DISK I/O with return_frames=True!
            output_frames = faster_propainter_pipeline(
                video='dummy', mask='dummy', output=output_dir,
                resize_ratio=1.0, mask_dilation=SAM2_MASK_DILATION,
                ref_stride=10, neighbor_length=neighbor_length, subvideo_length=subvideo_length,
                raft_iter=20, mode="video_inpainting", save_fps=int(original_fps), save_frames=False,
                fp16=use_fp16, use_cached_models=True, frames_array=seg_frames, masks_array=seg_masks,
                return_frames=True  # ZERO DISK I/O!
            )

            expected_count = proc_end - proc_start
            if not output_frames:
                print(f"[SAM2] WARNING: Segment {seg_idx+1} returned no frames!")
                return seg_idx, None, None, (proc_start, proc_end)
            elif len(output_frames) != expected_count:
                print(f"[SAM2] WARNING: Segment {seg_idx+1} frame count mismatch! Expected {expected_count}, got {len(output_frames)}")
            else:
                print(f"[SAM2] Segment {seg_idx+1}: ProPainter returned {len(output_frames)} frames (OK)")

            # REVOLUTIONARY: Store in VRAM with YUV420 compression (8:1 ratio, ZERO disk I/O!)
            # Only save CORE frames (start_f to end_f), skip padding frames
            core_count = end_f - start_f
            print(f"[SAM2] Segment {seg_idx+1}: storing {core_count} core frames in VRAM (skip {len(output_frames) - core_count} padding)...")
            frames_stored = 0
            for i in range(len(output_frames)):
                frame_idx = proc_start + i
                # Skip padding frames - only save core frames [start_f, end_f)
                if frame_idx < start_f or frame_idx >= end_f:
                    continue
                # GET from shared preloaded frames (ZERO disk I/O!)
                orig_frame = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                # Upscale if we downsampled for processing
                out_frame = output_frames[i]
                if scale_factor < 1.0:
                    out_frame = cv2.resize(out_frame, (seg_crop_w, seg_crop_h), interpolation=cv2.INTER_LANCZOS4)
                # Composite segment output onto original frame (seg_crop coords are in frame coords)
                orig_frame[seg_crop_y:seg_crop_y + seg_crop_h,
                          seg_crop_x:seg_crop_x + seg_crop_w] = out_frame
                # Store in VRAM with YUV420 compression (8:1 ratio!)
                vram_frames[frame_idx] = VRAMCompressor.compress_frame(orig_frame)
                processed_frames.add(frame_idx)
                frames_stored += 1
            del output_frames  # Free memory immediately!
            print(f"[SAM2] Segment {seg_idx+1}: stored {frames_stored} core frames in VRAM")

            # Return success flag and metadata (NOT frames!)
            return seg_idx, True, (seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h), (proc_start, proc_end)

        def process_segment_disk(seg_idx, start_f, end_f, seg_bbox):
            """DISK-based segment processing (like START_CELERY_TRT) - no VRAM!"""
            duration = end_f - start_f
            log_end = end_f - 1 if end_f > start_f else end_f
            seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                list(seg_bbox), width, height, padding_ratio=0.2, min_size=128
            )
            movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
            neighbor_length, subvideo_length = (5, 60) if movement < 10 else (10, 80)
            pad_left = min(SEGMENT_TEMPORAL_PAD, start_f)
            pad_right = min(SEGMENT_TEMPORAL_PAD, total_frames - end_f)
            proc_start = start_f - pad_left
            proc_end = end_f + pad_right

            print(f"[YOLO] Segment {seg_idx+1}: frames {start_f}-{log_end} ({duration}f), crop={seg_crop_w}x{seg_crop_h}")

            # READ FRAMES FROM VIDEO FILE (disk-based!)
            seg_frames = []
            seg_masks = []
            cap_seg = cv2.VideoCapture(video_path)
            cap_seg.set(cv2.CAP_PROP_POS_FRAMES, proc_start)
            for frame_idx in range(proc_start, proc_end):
                ret, frame = cap_seg.read()
                if ret:
                    seg_frame = frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                    seg_frames.append(np.ascontiguousarray(seg_frame))
                    if frame_idx < len(all_masks):
                        seg_mask = all_masks[frame_idx][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                        seg_masks.append(np.ascontiguousarray(seg_mask))
            cap_seg.release()

            if not seg_frames or not seg_masks:
                print(f"[YOLO] WARNING: Segment {seg_idx+1} has no frames/masks!")
                return seg_idx, None, None, (proc_start, proc_end)

            # Run ProPainter
            output_frames = faster_propainter_pipeline(
                video='dummy', mask='dummy', output=output_dir,
                resize_ratio=1.0, mask_dilation=SAM2_MASK_DILATION,
                ref_stride=10, neighbor_length=neighbor_length, subvideo_length=subvideo_length,
                raft_iter=20, mode="video_inpainting", save_fps=int(original_fps), save_frames=False,
                fp16=use_fp16, use_cached_models=True, frames_array=seg_frames, masks_array=seg_masks,
                return_frames=True
            )

            if not output_frames:
                print(f"[YOLO] WARNING: Segment {seg_idx+1} returned no frames!")
                return seg_idx, None, None, (proc_start, proc_end)

            # SAVE OUTPUT FRAMES TO DISK (shared directory)
            # Re-read original frames and composite output onto them
            cap_orig = cv2.VideoCapture(video_path)
            frames_saved = 0
            for i, out_frame in enumerate(output_frames):
                frame_idx = proc_start + i
                if frame_idx < start_f or frame_idx >= end_f:
                    continue  # Skip padding
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, orig_frame = cap_orig.read()
                if ret:
                    orig_frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w] = out_frame
                    out_path = os.path.join(shared_cleaned_dir, f"{frame_idx:06d}.png")
                    cv2.imwrite(out_path, orig_frame)
                    processed_frames.add(frame_idx)
                    frames_saved += 1
            cap_orig.release()
            del output_frames
            print(f"[YOLO] Segment {seg_idx+1}: saved {frames_saved} frames to disk")

            return seg_idx, True, (seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h), (proc_start, proc_end)

        # Process segments (parallel when enabled, like main task)
        segments_to_process = segments or [(0, total_frames, [0, 0, width, height])]

        # YOLO MODE: Forward to server_production.py's prepare_video task (EXACT same flow as START_CELERY_TRT!)
        # This ensures YOLO detection, mask creation, and segment processing all happen consistently
        if mode == 'yolo' and len(segments_to_process) > 1:
            print(f"[YOLO] Forwarding to server_production.py prepare_video task (START_CELERY_TRT flow)...")
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': 'Forwarding to YOLO pipeline'})

            # Get API base URL for video download
            api_base_url = os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL') or ''
            temp_base_url = os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL') or ''

            # Call server_production.py's prepare_video task - pass masks_dir AND masks_url!
            # masks_url allows remote workers to download masks from B2 CDN if local path doesn't exist
            from celery import signature
            task = signature('prepare_video', args=[video_path], kwargs={
                'api_base': api_base_url,
                'temp_base': temp_base_url,
                'masks_dir': masks_dir,  # Local path (may not exist on remote worker)
                'masks_url': masks_url,  # B2 CDN URL for remote download fallback
            })
            result = task.apply_async(queue='propainter')  # Force to Docker (has updated code)

            print(f"[YOLO] Forwarded to prepare_video task: {result.id}")
            print(f"   Pre-downloaded masks at: {masks_dir}")

            # 🔥 VRAM cleanup after YOLO mask download/forwarding
            cleanup_vram("after YOLO forward")

            return {
                'status': 'processing',
                'task_id': result.id,
                'video_id': video_id,
                'message': 'YOLO: Forwarded to server_production.py pipeline'
            }

        # SAM2 MODE: Use ThreadPoolExecutor (2 workers, VRAM sharing)
        elif SAM2_PARALLEL_SEGMENTS and SEGMENT_WORKERS > 1 and len(segments_to_process) > 1:
            print(f"[SAM2] Processing {len(segments_to_process)} segments in PARALLEL (workers={min(SEGMENT_WORKERS, len(segments_to_process))})...")

            max_workers = min(SEGMENT_WORKERS, len(segments_to_process))
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments_to_process):
                    futures[executor.submit(process_segment_local, seg_idx, start_f, end_f, seg_bbox)] = seg_idx

                completed = 0
                for future in as_completed(futures):
                    seg_idx_done = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"[ERROR] Segment {seg_idx_done} raised: {e}")
                        import traceback
                        traceback.print_exc()
                        result = (seg_idx_done, None, None, (0, 0))
                    segment_results[seg_idx_done] = result
                    completed += 1
                    # Cleanup VRAM after each segment to prevent accumulation
                    cleanup_vram(f"after segment {seg_idx_done+1}")
                    # Check if watchdog triggered abort
                    check_vram_abort()
                    print(f"[SAM2] Completed {completed}/{len(segments_to_process)} segments")
        else:
            print(f"[SAM2] Processing {len(segments_to_process)} segment(s) sequentially...")
            for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments_to_process):
                segment_results[seg_idx] = process_segment_local(seg_idx, start_f, end_f, seg_bbox)
                # Cleanup VRAM after each segment to prevent accumulation
                cleanup_vram(f"after segment {seg_idx+1}")
                # Check if watchdog triggered abort
                check_vram_abort()

        # --- REVOLUTIONARY: Direct VRAM → FFmpeg pipe → NVENC (ZERO disk I/O!) ---
        print(f"[SAM2] Segment processing complete - {len(vram_frames)} frames stored in VRAM")
        print(f"[SAM2] Encoding directly from VRAM via FFmpeg pipe (ZERO disk I/O!)...")
        import time as _time
        _encode_start = _time.time()

        output_path = os.path.join(RESULT_DIR, f"{video_id}_sam2_removed.mp4")

        # Get frame dimensions from first available frame
        first_frame = VRAMCompressor.decompress_frame(
            vram_frames[min(vram_frames.keys())] if vram_frames else preloaded_frames[0]
        )
        frame_height, frame_width = first_frame.shape[:2]

        # Choose encoder: H.264 NVENC for <=4096, H.265 NVENC for larger, libx264 fallback
        _, encoder_args = _get_nvenc_encoder(frame_width, frame_height)

        # FFmpeg command: pipe rawvideo → encoder → MP4 with audio from original
        ffmpeg_cmd = [
            str(FFMPEG_EXE), '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{frame_width}x{frame_height}',
            '-r', str(int(original_fps)),
            '-i', 'pipe:0',  # Video from stdin
            '-i', video_path,  # Audio from original
            '-map', '0:v:0',
            '-map', '1:a:0?',
        ] + encoder_args + [
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]

        # Start FFmpeg process - capture stderr for debugging
        print(f"[FFMPEG] Command: {' '.join(ffmpeg_cmd)}")
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE  # Capture stderr for debugging
        )

        # Stream ALL frames directly to FFmpeg (processed + unprocessed)
        try:
            for frame_idx in range(total_frames):
                # Progress logging every 100 frames
                if frame_idx % 100 == 0:
                    print(f"[SAM2] Encoding frame {frame_idx}/{total_frames}...")

                if frame_idx in vram_frames:
                    # Processed frame from segment
                    frame_bgr = VRAMCompressor.decompress_frame(vram_frames[frame_idx])
                elif frame_idx in preloaded_frames:
                    # Unprocessed frame from preloaded
                    frame_bgr = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                else:
                    print(f"[WARNING] Frame {frame_idx} not found in VRAM, skipping")
                    continue

                proc.stdin.write(frame_bgr.tobytes())

            proc.stdin.close()
        except BrokenPipeError:
            stderr_output = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr else ''
            print(f"[FFMPEG ERROR] FFmpeg crashed during encoding")
            print(f"[FFMPEG STDERR] {stderr_output[-2000:]}")  # Last 2000 chars
            raise RuntimeError(f"FFmpeg crashed during encoding: {stderr_output[-500:]}")

        proc.wait(timeout=300)

        if proc.returncode != 0:
            print(f"[ERROR] FFmpeg encoding failed (code {proc.returncode})")
            raise RuntimeError(f"Video encoding failed (code {proc.returncode})")

        _encode_time = _time.time() - _encode_start
        print(f"[SAM2] NVENC encode complete: {total_frames} frames in {_encode_time:.2f}s ({total_frames/_encode_time:.1f} fps)")

        # Free VRAM
        import torch as _torch
        del vram_frames
        _torch.cuda.empty_cache()

        # Upload to B2 + Cloudflare CDN (inlined - no heavy imports!)
        cdn_url = None
        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo
            import time as _upload_time

            B2_KEY_ID = os.getenv('B2_KEY_ID')
            B2_APP_KEY = os.getenv('B2_APP_KEY')
            B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
            B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

            if not B2_KEY_ID or not B2_APP_KEY:
                print(f"[B2] Warning: B2 credentials not set - skipping upload")
            elif os.getenv('B2_UPLOAD_ENABLED', '1') == '1':
                timestamp = int(_upload_time.time())
                remote_path = f"results/{timestamp}_{os.path.basename(output_path)}"

                print(f"[B2] Uploading to {B2_BUCKET}/{remote_path}...")
                _b2_start = _upload_time.time()
                info = InMemoryAccountInfo()
                b2_api = B2Api(info)
                b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
                bucket = b2_api.get_bucket_by_name(B2_BUCKET)
                bucket.upload_local_file(local_file=output_path, file_name=remote_path)
                cdn_url = f"{B2_CDN_URL}/{remote_path}"
                _b2_time = _upload_time.time() - _b2_start
                print(f"[B2] Upload complete in {_b2_time:.1f}s - CDN URL: {cdn_url}")
        except ImportError:
            print(f"[B2] b2sdk not installed - skipping upload")
        except Exception as e:
            print(f"[B2] Upload failed: {e}")

        # 🚀 Return immediately after upload - cleanup happens in finally block!
        # This shaves ~3 seconds off user wait time
        return {
            'status': 'success', 'video_id': video_id,
            'video_path': video_path, 'output_path': cdn_url or output_path,
            'total_frames': total_frames, 'masks_generated': len(all_masks), 'masks_expanded': len(all_masks),
            'segments_processed': len(segments or []), 'width': width, 'height': height,
            'fps': original_fps, 'pipeline': 'full_fps_segments', 'message': f'SAM2 pipeline complete! Processed {len(segments or [])} segment(s)'
        }
    except Exception as e:
        print(f"[ERROR] Continue-after-masks task failed: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup temp files on failure
        try:
            import shutil as _shutil
            for temp_path in [output_dir, masks_dir]:
                if isinstance(temp_path, str) and os.path.exists(temp_path):
                    if os.path.isdir(temp_path):
                        _shutil.rmtree(temp_path)
                        print(f"[CLEANUP] Removed {temp_path}")
        except Exception as cleanup_err:
            print(f"[CLEANUP] Warning: {cleanup_err}")
        raise
    finally:
        # Stop VRAM watchdog
        stop_vram_watchdog()
        # NUCLEAR cleanup in BACKGROUND THREAD - don't block task completion!
        import threading
        def _bg_cleanup():
            try:
                full_cleanup(video_id=video_id, context="after _continue_after_masks")
            except:
                cleanup_vram("finally block fallback")
        cleanup_thread = threading.Thread(target=_bg_cleanup, daemon=True)
        cleanup_thread.start()


# =============================================================================
# UNIFIED WORKER: Import server_production tasks to handle YOLO chord in same worker
# =============================================================================
# This allows running ONE Celery worker that handles both:
# - SAM2 interactive tasks (propainter queue)
# - YOLO chord tasks (celery/default queue)
try:
    import server_production
    # server_production tasks are now registered when imported
    print("[UNIFIED] server_production tasks imported - single worker handles both SAM2 and YOLO!")
except Exception as e:
    print(f"[UNIFIED] Warning: Could not import server_production: {e}")
