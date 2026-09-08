"""
Production Server for Watermark Removal SaaS
- Async queue processing with Celery + Redis
- GPU-optimized YOLO detection + ProPainter inpainting
- Keeps your PC usable while serving customers
- Designed for $1Mi/month scale
- ALL FILES STAY ON D DRIVE (inside watermarkz folder)
- B2/CDN fefefefef upload with Railway notification (2025-12-07)
"""

import sys
import os
import importlib
import shutil
from pathlib import Path
from email_utils import send_reset_email, send_verification_email

# Load environment variables from .env file (for Celery Redis configuration)
from dotenv import load_dotenv
load_dotenv()

# CRITICAL: Force ALL temp/cache to D drive (watermarkz folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detect Railway environment
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT_NAME') is not None or os.getenv('RAILWAY') is not None

# Use Railway volume (/data) for persistent storage in production
if IS_RAILWAY:
    DATA_DIR = '/data'
    TEMP_DIR = os.path.join(DATA_DIR, 'temp')
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')
    UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
    RESULT_DIR = os.path.join(DATA_DIR, 'results')
    STATIC_VIDEOS_DIR = os.path.join(DATA_DIR, 'static_videos')
    TRAINING_VIDEOS_DIR = os.path.join(DATA_DIR, 'training_videos')
else:
    DATA_DIR = SCRIPT_DIR
    TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
    CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
    UPLOAD_DIR = os.path.join(SCRIPT_DIR, 'uploads')
    RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')
    STATIC_VIDEOS_DIR = os.path.join(SCRIPT_DIR, 'web')
    TRAINING_VIDEOS_DIR = os.path.join(SCRIPT_DIR, 'videostotrain')

DEBUG_DIR = os.path.join(RESULT_DIR, 'debug_masks')
PYTHON_PACKAGES_DIR = os.path.join(SCRIPT_DIR, 'python_packages')
PROPAINTER_SCRIPT = os.path.join(SCRIPT_DIR, 'ProPainter', 'inference_propainter.py')
PROPAINTER_OUTPUT_ROOT = os.path.join(RESULT_DIR, 'propainter')
PROPAINTER_MASK_ROOT = os.path.join(TEMP_DIR, 'propainter_masks')
PROPAINTER_FLOW_BACKEND = os.getenv('PROPAINTER_FLOW_BACKEND', 'raft')  # 'raft' or 'fastflownet'
try:
    _segment_workers_env = int(os.getenv('SEGMENT_WORKERS', '2'))
except ValueError:
    _segment_workers_env = 2
SEGMENT_WORKERS = max(1, _segment_workers_env)

# Force parallel segmentation for multi-GPU/multi-worker distribution (only if YOLO fails)
os.environ.setdefault('MIN_SEGMENTS', '5')  # Fallback split if YOLO finds 0-1 segments
os.environ.setdefault('MIN_CHUNK_FRAMES', '60')  # Minimum frames per chunk (fallback when YOLO fails)

# Create directories
for directory in [TEMP_DIR, CACHE_DIR, UPLOAD_DIR, RESULT_DIR, DEBUG_DIR, PROPAINTER_OUTPUT_ROOT, PROPAINTER_MASK_ROOT]:
    os.makedirs(directory, exist_ok=True)

# Override ALL temp/cache environment variables
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['PIP_CACHE_DIR'] = os.path.join(SCRIPT_DIR, 'pip_cache')
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
os.environ['OPENCV_TEMP_PATH'] = TEMP_DIR


def _ensure_cuda_torch():
    """
    Make sure we end up with a CUDA-enabled torch build.
    Prefer the system install if it has CUDA; otherwise fall back to python_packages.
    """
    def _import_torch(disable_triton_retry: bool = False):
        """
        Import torch and, if we hit the duplicated TORCH_LIBRARY Triton error,
        retry exactly once with Triton disabled.
        """
        try:
            return importlib.import_module('torch')
        except RuntimeError as exc:
            message = str(exc)
            if (
                not disable_triton_retry
                and "Only a single TORCH_LIBRARY" in message
            ):
                os.environ['PYTORCH_DISABLE_TRITON'] = '1'
                for module_name in ['torch', 'triton', 'torch_triton', 'torchvision._cuda', 'torchvision._C']:
                    sys.modules.pop(module_name, None)
                importlib.invalidate_caches()
                return _import_torch(disable_triton_retry=True)
            raise

    try:
        _torch_test = _import_torch()
        if hasattr(_torch_test, 'cuda') and _torch_test.cuda.is_available():
            sys.modules['torch'] = _torch_test
            return
        # GPU not available – continue with the existing torch module (CPU fallback)
        sys.modules['torch'] = _torch_test
        print("[WARNING]  CUDA not detected in system torch; continuing with CPU mode.")
        return
    except Exception:
        # Drop whatever was imported and fall back to bundled packages
        sys.modules.pop('torch', None)
        sys.modules.pop('triton', None)

    if PYTHON_PACKAGES_DIR not in sys.path:
        sys.path.insert(0, PYTHON_PACKAGES_DIR)

    torch_cuda = _import_torch()
    if not hasattr(torch_cuda, 'cuda') or not torch_cuda.cuda.is_available():
        raise RuntimeError(
            "CUDA-enabled torch not available in python_packages. "
            "Reinstall dependencies or run INSTALL_WAVEMIX.bat."
        )
    sys.modules['torch'] = torch_cuda


# Try to initialize CUDA/torch (local workers only)
# Railway deployment runs API-only mode (no GPU needed)
GPU_AVAILABLE = False
try:
    _ensure_cuda_torch()
    GPU_AVAILABLE = True
    print("[OK] GPU/CUDA initialized successfully - worker mode enabled")
except Exception as e:
    #GPU init failed silently
    #Railway mode info suppressed
    pass

from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
from celery import Celery, chord

# Conditional imports for GPU processing (only needed on local workers)
try:
    import cv2
except ImportError:
    cv2 = None
    if not GPU_AVAILABLE:
        print("[INFO] cv2 not available (API-only mode)")

try:
    import numpy as np
except ImportError:
    np = None
    if not GPU_AVAILABLE:
        print("[INFO] numpy not available (API-only mode)")
import io
import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
import redis
import threading
import secrets
import hmac
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import smtplib
import ssl
from email.message import EmailMessage
from contextlib import contextmanager

# Stripe for billing (optional - gracefully handle if not installed)
try:
    import stripe
    STRIPE_ENABLED = True
    stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
except ImportError:
    STRIPE_ENABLED = False
    print("[WARNING] Stripe not installed - billing endpoints disabled")

# B2 + Cloudflare CDN for zero-egress file storage
B2_KEY_ID = os.getenv('B2_KEY_ID', '')
B2_APP_KEY = os.getenv('B2_APP_KEY', '')
B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')
B2_UPLOAD_ENABLED = os.getenv('B2_UPLOAD_ENABLED', '1') == '1'


try:
    from b2sdk.v2 import B2Api, InMemoryAccountInfo
    B2_ENABLED = bool(B2_KEY_ID and B2_APP_KEY)
    B2_IMPORT_ERROR = None
    if B2_ENABLED:
        print(f"[OK] B2 storage enabled - CDN: {B2_CDN_URL}")
    else:
        print("[WARNING] B2 credentials not set - using local storage")
except Exception as e:
    B2_ENABLED = False
    B2_IMPORT_ERROR = str(e)
    print(f"[WARNING] b2sdk import failed: {e}")

def upload_to_b2(local_path: str, remote_path: str) -> str:
    """Upload a file to B2 and return CDN URL. Returns None on failure."""
    if not B2_ENABLED or not B2_UPLOAD_ENABLED:
        return None
    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET)
        bucket.upload_local_file(local_file=local_path, file_name=remote_path)
        cdn_url = f"{B2_CDN_URL}/{remote_path}"
        print(f"[B2] Uploaded {local_path} -> {cdn_url}")
        return cdn_url
    except Exception as e:
        print(f"[B2] Upload failed: {e}")
        return None

def list_b2_files(prefix='uploads/'):
    """List files in B2 bucket with given prefix"""
    if not B2_ENABLED:
        return []

    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET)

        # List files with prefix
        files = []
        for file_info, _ in bucket.ls(folder_to_list=prefix, fetch_count=1000):
            files.append({
                'name': file_info.file_name,
                'upload_timestamp': file_info.upload_timestamp / 1000,  # Convert from ms to seconds
                'file_id': file_info.id_
            })

        return files
    except Exception as e:
        print(f"[B2-CLEANUP] Error listing files: {e}")
        return []

def delete_from_b2(file_name, file_id):
    """Delete a specific file from B2 by file_id"""
    if not B2_ENABLED:
        return False

    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)

        # Delete file version
        b2_api.delete_file_version(file_id, file_name)
        print(f"[B2-CLEANUP] Deleted: {file_name}")
        return True
    except Exception as e:
        print(f"[B2-CLEANUP] Error deleting {file_name}: {e}")
        return False

# Authentication and session management
from flask import session, redirect, url_for
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    from google_auth_oauthlib.flow import Flow
    import bcrypt
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    AUTH_ENABLED = True
    print("[OK] Authentication modules loaded")
except ImportError as e:
    AUTH_ENABLED = False
    print(f"[WARNING] Authentication disabled - missing dependencies: {e}")
    print("[INFO] Install with: pip install google-auth google-auth-oauthlib bcrypt psycopg2-binary")

# [INIT] FFmpeg/FFprobe path detection (check common locations, no download)
def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths - check common locations first, then PATH."""
    import platform

    # On Linux/Docker, use system ffmpeg first (avoid Windows .exe files from mounted volumes)
    if platform.system() != 'Windows':
        ffmpeg_path = shutil.which('ffmpeg')
        ffprobe_path = shutil.which('ffprobe')
        if ffmpeg_path and ffprobe_path:
            print(f"[OK] Using system FFmpeg (Linux): {ffmpeg_path}")
            return ffmpeg_path, ffprobe_path

    # Common FFmpeg install locations on Windows
    common_locations = [
        os.path.join(SCRIPT_DIR, 'ffmpeg'),  # Local project folder
        r'C:\ffmpeg\bin',                     # Common install location
        r'C:\ffmpeg',                         # Another common location
        r'C:\Program Files\ffmpeg\bin',
        r'C:\Program Files (x86)\ffmpeg\bin',
        os.path.expandvars(r'%LOCALAPPDATA%\Programs\ffmpeg\bin'),
        os.path.expandvars(r'%USERPROFILE%\ffmpeg\bin'),
    ]

    # Check common locations first
    for location in common_locations:
        ffmpeg_path = os.path.join(location, 'ffmpeg.exe')
        ffprobe_path = os.path.join(location, 'ffprobe.exe')
        if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
            print(f"[OK] Using FFmpeg from: {location}", flush=True)
            return ffmpeg_path, ffprobe_path

    # Try system PATH
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')

    if ffmpeg_path and ffprobe_path:
        print(f"[OK] Using system FFmpeg: {ffmpeg_path}")
        return ffmpeg_path, ffprobe_path

    # Skip static_ffmpeg download - it can hang indefinitely
    print("[WARNING] FFmpeg not found! Video encoding will fail.")
    print("[WARNING] Install FFmpeg to C:\\ffmpeg\\bin or D:\\watermarkz\\ffmpeg\\")
    return None, None

# Initialize FFmpeg paths at module level (before Celery workers start)
# FFmpeg only needed on local workers (video processing), not on Railway API
try:
    FFMPEG_EXE, FFPROBE_EXE = get_ffmpeg_executables()
except Exception as e:
    FFMPEG_EXE, FFPROBE_EXE = None, None
    if not GPU_AVAILABLE:
        print(f"[INFO] FFmpeg not available (API-only mode): {e}")

print("[DEBUG] FFmpeg init complete, creating Flask app...", flush=True)

# [HEVC] Async transcode executor (non-blocking)
hevc_transcode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="HEVCTranscode")

# Codecs a browser will actually play in a <video> element. Anything outside
# this set has to be transcoded, not just HEVC - mpeg4, wmv and mpeg2 are all
# "not HEVC" and all unplayable, which is what made uploads show a blank player.
BROWSER_SAFE_CODECS = {'h264', 'avc1', 'vp8', 'vp9', 'av1', 'av01'}

# [HEVC] Transcode HEVC to H.264 for browser preview
def check_video_codec(video_path):
    """Check if video is HEVC/H.265 codec using ffprobe"""
    if not FFPROBE_EXE:
        return None
    try:
        import subprocess
        import json
        cmd = [FFPROBE_EXE, '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    codec = stream.get('codec_name', '').lower()
                    return codec
    except Exception as e:
        print(f"[HEVC] Error checking codec: {e}")
    return None

def transcode_hevc_to_h264(input_path, output_path):
    """Transcode HEVC to H.264 for browser playback"""
    if not FFMPEG_EXE:
        return False
    try:
        import subprocess
        # Use GPU encoding if available, fallback to CPU
        if GPU_AVAILABLE:
            cmd = [
                FFMPEG_EXE, '-y', '-i', input_path,
                '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                output_path
            ]
        else:
            # CPU encode, so speed matters more than fidelity. This file is only
            # ever the editor preview - the person watches it and taps the object
            # they want gone - while the render still runs on the original at full
            # resolution. Capping the preview at 720p on a fast preset turns a
            # minute of encoding into a few seconds, which is the difference
            # between someone waiting and someone leaving.
            cmd = [
                FFMPEG_EXE, '-y', '-i', input_path,
                '-vf', "scale='min(1280,iw)':'min(1280,ih)'"
                       ":force_original_aspect_ratio=decrease:force_divisible_by=2",
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                '-c:a', 'aac', '-b:a', '96k',
                '-movflags', '+faststart',
                '-threads', '0',
                output_path
            ]
        print(f"[HEVC] Transcoding to H.264: {input_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"[HEVC] Transcode success: {output_path}")
            return True
        else:
            print(f"[HEVC] Transcode failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"[HEVC] Transcode error: {e}")
        return False


def transcode_hevc_pipeline(video_url, task_id):
    """
    Complete async pipeline: Download -> Check codec -> Transcode if HEVC -> Upload to B2 -> Store in Redis
    This runs in background thread, doesn't block server.

    OPTIMIZATION: Downloads only 1MB first to check codec. If not HEVC, skips full download!
    Saves ~99% bandwidth for H.264 videos (Android/Desktop uploads).
    """
    import requests
    temp_input = None
    temp_output = None
    partial_path = None

    try:
        # Connect to Redis
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)

        # Set initial status
        redis_client.setex(f"preview_status:{task_id}", 86400, "checking")
        print(f"[HEVC] Pipeline started for {task_id}: {video_url}")

        # 1. OPTIMIZATION: Download ONLY first 1MB to check codec (not full file!)
        partial_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_partial.mp4")
        print(f"[HEVC] Downloading 1MB sample to check codec...")

        try:
            # Use Range header to get only first 1MB
            headers = {'Range': 'bytes=0-1048576'}
            response = requests.get(video_url, headers=headers, timeout=30)
            # Accept both 200 (full file) and 206 (partial content)
            if response.status_code not in [200, 206]:
                response.raise_for_status()

            with open(partial_path, 'wb') as f:
                f.write(response.content)

            partial_size = os.path.getsize(partial_path) / 1024
            print(f"[HEVC] Downloaded {partial_size:.0f}KB sample")

        except Exception as range_err:
            print(f"[HEVC] Range request failed, falling back to full download: {range_err}")
            partial_path = None

        # 2. Check codec on partial file (ffprobe works with partial files!)
        if partial_path and os.path.exists(partial_path):
            codec = check_video_codec(partial_path)
            print(f"[HEVC] Detected codec from sample: {codec}")

            # Cleanup partial file
            try:
                os.remove(partial_path)
                partial_path = None
            except:
                pass

            if codec and codec in BROWSER_SAFE_CODECS:
                # Browser can play this as-is - skip the full download.
                redis_client.setex(f"preview_status:{task_id}", 86400, "original")
                redis_client.setex(f"preview_url:{task_id}", 86400, video_url)
                print(f"[PREVIEW] {codec} plays in-browser, skipping download - saved bandwidth!")
                return video_url
        else:
            codec = None

        # 3. HEVC detected (or couldn't check) - download full file
        temp_input = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_input.mp4")
        print(f"[PREVIEW] {codec or 'unknown codec'} needs transcoding, downloading full video to {temp_input}")

        response = requests.get(video_url, stream=True, timeout=120)
        response.raise_for_status()

        with open(temp_input, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(temp_input) / (1024 * 1024)
        print(f"[HEVC] Downloaded {file_size:.1f}MB")

        # 4. Verify codec if we couldn't check earlier
        if not codec:
            codec = check_video_codec(temp_input)
            print(f"[HEVC] Verified codec: {codec}")

            if codec in BROWSER_SAFE_CODECS:
                # Playable after all - no transcode needed
                redis_client.setex(f"preview_status:{task_id}", 86400, "original")
                redis_client.setex(f"preview_url:{task_id}", 86400, video_url)
                print(f"[PREVIEW] {codec} plays in-browser, using original URL")
                return video_url

        # 3. Transcode HEVC -> H.264
        redis_client.setex(f"preview_status:{task_id}", 86400, "transcoding")
        temp_output = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_preview.mp4")

        success = transcode_hevc_to_h264(temp_input, temp_output)

        if not success or not os.path.exists(temp_output):
            print(f"[HEVC] Transcode failed, falling back to original")
            redis_client.setex(f"preview_status:{task_id}", 86400, "original")
            redis_client.setex(f"preview_url:{task_id}", 86400, video_url)
            return video_url

        # 4. Upload transcoded file to B2
        preview_url = None
        if B2_ENABLED:
            try:
                info = InMemoryAccountInfo()
                b2_api = B2Api(info)
                b2_api.authorize_account('production', B2_KEY_ID, B2_APP_KEY)
                bucket = b2_api.get_bucket_by_name(B2_BUCKET)

                remote_path = f"previews/{task_id}_h264.mp4"

                with open(temp_output, 'rb') as f:
                    bucket.upload_bytes(
                        f.read(),
                        remote_path,
                        content_type='video/mp4'
                    )

                preview_url = f"{B2_CDN_URL}/{remote_path}"
                print(f"[HEVC] Uploaded preview to B2: {preview_url}")

            except Exception as b2_err:
                print(f"[HEVC] B2 upload failed: {b2_err}")
                # Fall back to original
                redis_client.setex(f"preview_status:{task_id}", 86400, "original")
                redis_client.setex(f"preview_url:{task_id}", 86400, video_url)
                return video_url
        else:
            print(f"[HEVC] B2 not enabled, falling back to original")
            redis_client.setex(f"preview_status:{task_id}", 86400, "original")
            redis_client.setex(f"preview_url:{task_id}", 86400, video_url)
            return video_url

        # 5. Store in Redis
        redis_client.setex(f"preview_status:{task_id}", 86400, "ready")
        redis_client.setex(f"preview_url:{task_id}", 86400, preview_url)
        print(f"[HEVC] Pipeline complete! Preview: {preview_url}")

        return preview_url

    except Exception as e:
        print(f"[HEVC] Pipeline error: {e}")
        # Always fallback to original on error
        try:
            redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
            redis_client.setex(f"preview_status:{task_id}", 86400, "original")
            redis_client.setex(f"preview_url:{task_id}", 86400, video_url)
        except:
            pass
        return video_url

    finally:
        # 6. CRITICAL: Cleanup temp files (including partial sample)
        for f in [temp_input, temp_output, partial_path]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"[HEVC] Cleaned up: {f}")
                except Exception as cleanup_err:
                    print(f"[HEVC] Cleanup failed: {cleanup_err}")


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
            # Log VRAM status
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] Cleanup {context}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        print(f"[VRAM] Cleanup error: {e}")

# [INIT] EXTREME SPEED: Global in-memory frame/mask cache
# Shared across all threads in Celery worker (threads pool)
# Stores frames/masks in RAM for instant access (no Redis, no disk!)
FRAME_CACHE = {}
FRAME_CACHE_LOCK = threading.Lock()

# Import faster-propainter modules for direct pipeline processing
# Note: These imports are moved to inside functions to avoid startup errors
# sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'faster-propainter-main'))
# from watermark import pipeline as faster_propainter_pipeline
# from mytimer import timer_decorator  
# from pre_post_process import crop_video_mask, merge_videos_with_mask

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web'))
# CORS: allow cross-origin calls to /api/* and accept the ngrok header if present
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
    expose_headers=["Content-Disposition"]
)
print("[DEBUG] Flask app + CORS created", flush=True)

# ----------------------------------------------------------------------------
# Database Connection Pool (for user authentication)
# ----------------------------------------------------------------------------
db_pool = None
print("[DEBUG] About to check database init...", flush=True)
# Skip database init for GPU workers (they don't need auth, and it can hang)
if AUTH_ENABLED and not GPU_AVAILABLE:
    try:
        DATABASE_URL = os.getenv('DATABASE_URL')
        if DATABASE_URL:
            # Railway provides postgres:// but psycopg2 needs postgresql://
            if DATABASE_URL.startswith('postgres://'):
                DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

            db_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DATABASE_URL
            )
            print("[OK] Database connection pool initialized")
        else:
            print("[WARNING] DATABASE_URL not set - authentication will not work")
            AUTH_ENABLED = False
    except Exception as e:
        print(f"[ERROR] Failed to initialize database pool: {e}")
        AUTH_ENABLED = False

print("[DEBUG] Database init block passed (skipped for GPU)", flush=True)

@contextmanager
def get_db():
    """Get database connection from pool."""
    if not db_pool:
        raise RuntimeError("Database pool not initialized")

    conn = db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)

# ----------------------------------------------------------------------------
# Authentication Decorators
# ----------------------------------------------------------------------------
from functools import wraps

def require_auth(f):
    """Decorator to require authentication for endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not AUTH_ENABLED:
            # If auth is disabled, allow access (for development)
            return f(*args, **kwargs)

        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'status': 'error',
                'error': 'Authentication required. Please sign in.',
                'signin_required': True
            }), 401

        return f(*args, **kwargs)
    return decorated_function

def require_credits(min_credits=1):
    """Decorator to check if user has sufficient credits."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not AUTH_ENABLED:
                # If auth is disabled, allow access (for development)
                return f(*args, **kwargs)

            user_id = session.get('user_id')
            if not user_id:
                return jsonify({
                    'status': 'error',
                    'error': 'Authentication required',
                    'signin_required': True
                }), 401

            # Check credit balance
            try:
                with get_db() as conn:
                    cur = conn.cursor()
                    cur.execute('SELECT credits FROM users WHERE id = %s', (user_id,))
                    result = cur.fetchone()

                    if not result:
                        return jsonify({
                            'status': 'error',
                            'error': 'User not found'
                        }), 404

                    credits = result[0]
                    if credits < min_credits:
                        return jsonify({
                            'status': 'error',
                            'error': 'Insufficient credits',
                            'required': min_credits,
                            'available': credits,
                            # The app has nothing to sell, so it must never be
                            # told to go buy something. The website still gets
                            # the wording that points at its own pricing.
                            'message': (
                                f"That's today's free videos used up. Two more tomorrow."
                                if _is_app_client() else
                                f'You need {min_credits} credit(s) but only have {credits}. Please purchase more credits.'
                            )
                        }), 402  # Payment Required
            except Exception as e:
                print(f"[ERROR] Credit check failed: {e}")
                # Allow processing to continue if database check fails (graceful degradation)
                pass

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def deduct_credit_on_completion(task_id):
    """
    Deduct credits when task completes successfully.
    Looks up user_id and credit amount from Redis (stored when task was created).
    Uses atomic check-and-set to prevent double-deduction.
    Returns new balance (int) on success, None on failure.
    """
    if not AUTH_ENABLED:
        return None

    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)

        # Look up user_id and credits amount from stored task data
        user_id = redis_client.get(f"task:{task_id}:user_id")
        credits_to_deduct = float(redis_client.get(f"task:{task_id}:credits") or 0.1)

        if not user_id:
            print(f"[CREDITS] No user_id found for task {task_id}")
            return None

        redis_key = f"credits_deducted:{task_id}"

        # Atomic check-and-set - only proceeds if key doesn't exist
        if redis_client.setnx(redis_key, "1"):
            # First time seeing completion - deduct credits
            redis_client.expire(redis_key, 86400 * 7)  # Expire after 7 days

            with get_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    'UPDATE users SET credits = credits - %s WHERE id = %s AND credits >= %s RETURNING credits',
                    (credits_to_deduct, user_id, credits_to_deduct)
                )
                deduct_result = cur.fetchone()
                if deduct_result:
                    # float(), not the raw Decimal. jsonify renders Decimal as
                    # a JSON *string* ("1.00"), and clients that type this as a
                    # number fail to decode the whole response - which happens
                    # only on the reply that reports completion, so a finished
                    # job looks like it is still running.
                    new_balance = float(deduct_result[0])
                    print(f"[CREDITS] Deducted {credits_to_deduct} credit(s) for user {user_id} on task {task_id}. New balance: {new_balance}")
                    return new_balance
                else:
                    print(f"[CREDITS] Deduction failed for user {user_id} - insufficient credits")
                    return None
        else:
            # Already deducted - return current balance
            print(f"[CREDITS] Already deducted for task {task_id}, returning current balance")
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute('SELECT credits FROM users WHERE id = %s', (user_id,))
                result = cur.fetchone()
                return float(result[0]) if result else None

    except Exception as e:
        print(f"[CREDITS] Error during deduction: {e}")
        return None


# ----------------------------------------------------------------------------
# Daily free tier (iOS app)
# ----------------------------------------------------------------------------
# The App Store build ships with no in-app purchases while the Paid Apps
# agreement is still pending, so app users get an allowance instead: a top-up
# to DAILY_FREE_CREDITS once per calendar day (server time, UTC on Railway).
#
# Deliberately a top-up *to* the cap rather than an addition, which is what
# stops it accumulating - going away for a week still leaves you with 2, not
# 14. The same rule is why a balance already above the cap is left completely
# alone: someone who bought a pack on the website keeps and spends what they
# paid for, and only drops back onto the free tier once it runs out.
#
# Granted to the app only. The website sells these same credits, so handing
# them out there would undercut it.

DAILY_FREE_CREDITS = float(os.getenv('DAILY_FREE_CREDITS', '2'))
APP_CLIENT_HEADER = 'X-Client'
APP_CLIENT_VALUE = 'objectremoverai-ios'


def _is_app_client():
    """True when the request came from the iOS app rather than the website."""
    try:
        return request.headers.get(APP_CLIENT_HEADER, '').strip().lower() == APP_CLIENT_VALUE
    except Exception:
        # No request context (startup, Celery). Not the app, so no grant.
        return False


def _ensure_daily_credit_table(cur):
    """One row per user recording the last day their allowance was claimed."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_daily_credits (
            user_id    INTEGER PRIMARY KEY,
            grant_date DATE NOT NULL,
            granted    NUMERIC NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)


def grant_daily_free_credits(cur, user_id):
    """Tops an app user up to the daily allowance.

    Returns the new balance, or None when nothing was granted (not the app,
    already claimed today, or already holding more than the cap).

    Claiming the day happens first and atomically: the balance UPDATE only runs
    for the request that actually moved grant_date forward, so two launches at
    the same moment cannot both pay out.
    """
    if not _is_app_client():
        return None

    try:
        _ensure_daily_credit_table(cur)

        cur.execute("""
            INSERT INTO app_daily_credits (user_id, grant_date, granted)
            VALUES (%s, CURRENT_DATE, 0)
            ON CONFLICT (user_id) DO UPDATE
                SET grant_date = CURRENT_DATE, updated_at = NOW()
                WHERE app_daily_credits.grant_date < CURRENT_DATE
            RETURNING user_id
        """, (user_id,))

        if not cur.fetchone():
            return None  # someone already claimed today for this user

        # Top up TO the cap, never past it. The `credits <` guard is the whole
        # reason a web purchase is safe here - without it this would reset a
        # 60-credit balance down to 2.
        cur.execute(
            'UPDATE users SET credits = %s WHERE id = %s AND credits < %s RETURNING credits',
            (DAILY_FREE_CREDITS, user_id, DAILY_FREE_CREDITS)
        )
        row = cur.fetchone()
        if not row:
            print(f"[FREE-TIER] User {user_id} already above the cap, nothing granted")
            return None

        new_balance = float(row[0])
        cur.execute(
            'UPDATE app_daily_credits SET granted = %s WHERE user_id = %s',
            (new_balance, user_id)
        )
        print(f"[FREE-TIER] User {user_id} topped up to {new_balance} for today")
        return new_balance

    except Exception as exc:
        # A failed grant must never block sign-in.
        print(f"[FREE-TIER] Grant failed for user {user_id}: {exc}")
        return None


print("[DEBUG] Decorator definitions complete", flush=True)
print("[DEBUG] About to check REDIS_URL...", flush=True)
# ----------------------------------------------------------------------------
# Redis URL Definition (needed for Session and Celery)
# ----------------------------------------------------------------------------
REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL:
    print("[WARNING] REDIS_URL not found in environment. Session and Celery may fail.")
else:
    print(f"[OK] Using REDIS_URL from environment for Session config: {REDIS_URL}")


# ----------------------------------------------------------------------------
# Flask Session Configuration (using Redis for multi-worker stability)
# ----------------------------------------------------------------------------
# Skip Flask-Session for GPU workers (they don't need web sessions)
if not GPU_AVAILABLE:
    try:
        from flask_session import Session
        app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.getenv('SECRET_KEY', secrets.token_hex(32)))
        app.config['SESSION_TYPE'] = 'redis'
        app.config['SESSION_PERMANENT'] = True
        app.config['SESSION_USE_SIGNER'] = True  # Encrypt session cookie
        app.config['SESSION_REDIS'] = redis.from_url(REDIS_URL)
        app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
        app.config['SESSION_COOKIE_SECURE'] = True      # Required for HTTPS
        app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'   # Allow OAuth redirects
        Session(app)
        print("[OK] Flask-Session initialized with Redis")
    except ImportError:
        print("[WARNING] flask_session not installed - sessions disabled")
else:
    print("[OK] Skipping Flask-Session (GPU worker mode)")

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI', '')
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# ----------------------------------------------------------------------------
# Simple access logging (Waitress doesn't emit per‑request access logs by default)
# ----------------------------------------------------------------------------
import sys
import logging

# Toggle per-request access logs with ACCESS_LOGS=1 (default disabled)
ENABLE_ACCESS_LOGS = str(os.getenv('ACCESS_LOGS', '0')).lower() in ('1', 'true', 'yes', 'on')
LOG_LEVEL = logging.INFO if ENABLE_ACCESS_LOGS else logging.WARNING
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)

from flask import g

@app.before_request
def _log_request_start():
    if not ENABLE_ACCESS_LOGS:
        return
    try:
        g._req_start = time.time()
        g._req_id = secrets.token_hex(4)
        ip = request.headers.get('X-Forwarded-For', request.remote_addr) or '-'
        qs = request.query_string.decode('utf-8', errors='ignore')
        path_qs = request.path + (('?' + qs) if qs else '')
        logging.info(f"--> {g._req_id} {request.method} {path_qs} from {ip}")
    except Exception:
        pass

@app.after_request
def _log_request_end(response):
    if ENABLE_ACCESS_LOGS:
        try:
            rid = getattr(g, '_req_id', '-')
            dur_ms = int((time.time() - getattr(g, '_req_start', time.time())) * 1000)
            length = response.calculate_content_length() or 0
            logging.info(f"<-- {rid} {response.status_code} {length}b {dur_ms}ms {request.method} {request.path}")
        except Exception:
            pass
    return response

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return ('', 204)
    """Basic health endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'message': 'Flask API server running - workers handle processing',
        'b2_enabled': B2_ENABLED,
        'b2_key_id_set': bool(B2_KEY_ID),
        'b2_app_key_set': bool(B2_APP_KEY),
        'b2_key_id_len': len(B2_KEY_ID) if B2_KEY_ID else 0,
        'b2_app_key_len': len(B2_APP_KEY) if B2_APP_KEY else 0,
        'b2_cdn': B2_CDN_URL if B2_ENABLED else None,
        'b2_import_error': B2_IMPORT_ERROR
    })

# ----------------------------------------------------------------------------
# Email Verification
# ----------------------------------------------------------------------------

# Brevo SMTP configuration
SMTP_SERVER = "smtp-relay.brevo.com"
SMTP_PORT = 587
SMTP_USERNAME = "9c3ca4001@smtp-brevo.com"
SMTP_PASSWORD = "xsmtpsib-798b8db725baf9ad346d32e4ccae5bdcfc64f7c03d059b5aef8e6f82fa5da30b-IB5SjcNk2AV6NZ1f"
SMTP_FROM = "markremoverai@gmail.com"

# send_verification_email is now imported from email_utils.py (uses Brevo API)

# ----------------------------------------------------------------------------
# Authentication Routes (Google OAuth + Email/Password)
# ----------------------------------------------------------------------------

def _external_callback_uri():
    """Railway terminates TLS at the proxy, so request.host_url reports http://.
    Google rejects any non-https redirect URI, so force the scheme."""
    host_url = request.host_url
    if host_url.startswith('http://'):
        host_url = 'https://' + host_url[len('http://'):]
    return f"{host_url}auth/google/callback"


@app.route('/auth/google')
def auth_google():
    """Initiate Google OAuth flow."""
    if not AUTH_ENABLED or not GOOGLE_CLIENT_ID:
        return jsonify({'error': 'Google OAuth not configured'}), 503

    try:
        native = request.args.get('native') == '1'
        # GOOGLE_REDIRECT_URI pins the callback to markremoverai.com, which is
        # exactly the host filtered Wi-Fi refuses to resolve. The app reaches
        # Railway directly, so send the callback back to whichever host it
        # actually used. That URI must also be registered in Google Cloud.
        redirect_uri = (
            _external_callback_uri() if native
            else (GOOGLE_REDIRECT_URI or _external_callback_uri())
        )

        # Create flow instance
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [redirect_uri]
                }
            },
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']
        )

        flow.redirect_uri = redirect_uri

        # Generate authorization URL with state token (CSRF protection)
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='select_account'
        )

        # Store state in session for verification
        session['oauth_state'] = state
        # The iOS app starts this same flow inside a web view. Remember that so
        # the callback hands back a one-time code instead of rendering the site.
        session['native_auth'] = request.args.get('native') == '1'
        # Store PKCE code_verifier so the callback can complete the token
        # exchange (google-auth-oauthlib enables PKCE by default; without this
        # Google rejects the exchange with "Missing code verifier").
        session['code_verifier'] = flow.code_verifier

        return redirect(authorization_url)

    except Exception as e:
        print(f"[ERROR] Google OAuth initiation failed: {e}")
        return jsonify({'error': 'Failed to initiate Google login'}), 500


@app.route('/auth/google/callback')
def auth_google_callback():
    """Handle Google OAuth callback."""
    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication not enabled'}), 503

    try:
        # Verify state token (CSRF protection)
        state = session.get('oauth_state')
        if not state or state != request.args.get('state'):
            return jsonify({'error': 'Invalid state parameter'}), 400

        # Must match the URI used to start the flow exactly.
        native = session.get('native_auth', False)
        redirect_uri = (
            _external_callback_uri() if native
            else (GOOGLE_REDIRECT_URI or _external_callback_uri())
        )

        # Create flow instance with same config
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [redirect_uri]
                }
            },
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
            state=state
        )

        flow.redirect_uri = redirect_uri
        # Restore the PKCE code_verifier saved when the flow was initiated.
        flow.code_verifier = session.get('code_verifier')

        # Exchange authorization code for tokens
        # Fix: request.url may have http:// behind reverse proxy, force https://
        auth_response = request.url.replace('http://', 'https://')
        flow.fetch_token(authorization_response=auth_response)

        # Get user info from ID token
        credentials = flow.credentials
        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Extract user information
        google_id = id_info['sub']
        email = id_info['email']
        name = id_info.get('name', email.split('@')[0])

        # Store or update user in database
        with get_db() as conn:
            cur = conn.cursor()

            # Check if user exists by google_id first
            cur.execute('SELECT id, credits FROM users WHERE google_id = %s', (google_id,))
            user = cur.fetchone()

            if user:
                user_id, credits = user
                print(f"[AUTH] Existing user logged in: {email}")
            else:
                # Check if user exists by email (registered with email/password)
                cur.execute('SELECT id, credits FROM users WHERE email = %s', (email,))
                user = cur.fetchone()

                if user:
                    # Link Google account to existing email user
                    user_id, credits = user
                    # COALESCE-style guard: a later sign-in that carries no name
                    # must not blank out the one already stored.
                    cur.execute(
                        'UPDATE users SET google_id = %s, name = COALESCE(NULLIF(%s, %s), name) WHERE id = %s',
                        (google_id, _display_name(name, email) if name else '', '', user_id)
                    )
                    print(f"[AUTH] Linked Google account to existing user: {email}")
                else:
                    # New user - give 2 free credits
                    cur.execute(
                        'INSERT INTO users (google_id, email, name, credits) VALUES (%s, %s, %s, %s) RETURNING id',
                        (google_id, email, _display_name(name, email), 2)
                    )
                    user_id = cur.fetchone()[0]
                    credits = 2
                    print(f"[AUTH] New user registered via Google: {email} (2 free credits)")

            # Create session
            session['user_id'] = user_id
            session['email'] = email
            session['name'] = name
            session.permanent = True

        # The app cannot read the cookie this web view just received, so give
        # it a short-lived code it can trade for a session of its own.
        if session.pop('native_auth', False):
            code = _mint_native_auth_code(user_id)
            if code:
                return redirect(f'{NATIVE_AUTH_SCHEME}://auth?code={code}')
            return redirect(f'{NATIVE_AUTH_SCHEME}://auth?error=code_unavailable')

        # Redirect to main page with user data for frontend localStorage
        from urllib.parse import urlencode
        params = urlencode({
            'auth_success': '1',
            'user_id': user_id,
            'email': email,
            'name': name,
            'credits': credits
        })
        return redirect(f'/?{params}')

    except Exception as e:
        import traceback
        print(f"[ERROR] Google OAuth callback failed: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Authentication failed: {str(e)}'}), 500


@app.route('/api/auth/register', methods=['POST', 'OPTIONS'])
def auth_register():
    """Register new user with email/password."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication not enabled'}), 503

    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()

        # Validation
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400

        # Basic email validation
        if '@' not in email or '.' not in email.split('@')[1]:
            return jsonify({'error': 'Invalid email address'}), 400

        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Store user in database
        with get_db() as conn:
            cur = conn.cursor()

            # Check if email already exists
            cur.execute('SELECT id FROM users WHERE email = %s', (email,))
            if cur.fetchone():
                return jsonify({'error': 'Email already registered'}), 409

            # Generate verification token
            verification_token = secrets.token_urlsafe(32)
            token_expires = datetime.utcnow() + timedelta(hours=24)

            # Create user with 2 free credits and verification token
            cur.execute(
                '''INSERT INTO users (email, password_hash, name, credits, email_verified, verification_token, verification_token_expires)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id, created_at''',
                (email, password_hash, name or email.split('@')[0], 2, False, verification_token, token_expires)
            )
            user_id, created_at = cur.fetchone()

            # Create session
            session['user_id'] = user_id
            session['email'] = email
            session['name'] = name or email.split('@')[0]
            session.permanent = True

            # Send verification email (async to not block response)
            base_url = os.getenv('SITE_URL', 'https://markremoverai.com')
            verify_url = f"{base_url}/api/auth/verify?token={verification_token}"
            threading.Thread(target=send_verification_email, args=(email, verify_url)).start()

            print(f"[AUTH] New user registered: {email} (5 free credits, verification email sent)")

            return jsonify({
                'status': 'success',
                'message': 'Please check your email to verify your account',
                'email_verification_required': True,
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': session['name'],
                    'credits': 5,
                    'email_verified': False,
                    'created_at': created_at.isoformat() if created_at else None
                }
            })

    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def auth_login():
    """Login with email/password."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication not enabled'}), 503

    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        with get_db() as conn:
            cur = conn.cursor()

            # Get user
            cur.execute('SELECT id, password_hash, name, credits, email_verified, created_at FROM users WHERE email = %s', (email,))
            user = cur.fetchone()

            if not user:
                return jsonify({'error': 'Invalid email or password'}), 401

            user_id, password_hash, name, credits, email_verified, created_at = user

            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                return jsonify({'error': 'Invalid email or password'}), 401

            # Check if email is verified
            if not email_verified:
                return jsonify({
                    'error': 'Please verify your email before logging in. Check your inbox for the verification link.',
                    'needs_verification': True,
                    'email': email
                }), 403

            # Create session
            session['user_id'] = user_id
            session['email'] = email
            session['name'] = name
            session.permanent = True

            # The app trusts this response and does not re-fetch, so the
            # allowance has to be applied before the balance is reported.
            granted = grant_daily_free_credits(cur, user_id)
            if granted is not None:
                credits = granted

            print(f"[AUTH] User logged in: {email}")

            return jsonify({
                'status': 'success',
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': name,
                    'credits': float(credits or 0),
                    'email_verified': email_verified or False,
                    'created_at': created_at.isoformat() if created_at else None
                }
            })

    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/verify', methods=['GET'])
def auth_verify():
    """Verify email address from verification link."""
    token = request.args.get('token', '')

    if not token:
        return redirect('/login?error=missing_token')

    try:
        with get_db() as conn:
            cur = conn.cursor()

            # Find user by verification token
            cur.execute(
                'SELECT id, email, verification_token_expires FROM users WHERE verification_token = %s',
                (token,)
            )
            user = cur.fetchone()

            if not user:
                print(f"[AUTH] Invalid verification token attempted")
                return redirect('/login?error=invalid_token')

            user_id, email, token_expires = user

            # Check if token has expired
            if token_expires and datetime.utcnow() > token_expires:
                print(f"[AUTH] Expired verification token for {email}")
                return redirect('/login?error=token_expired')

            # Mark email as verified and clear the token
            cur.execute(
                '''UPDATE users
                   SET email_verified = TRUE, verification_token = NULL, verification_token_expires = NULL
                   WHERE id = %s''',
                (user_id,)
            )

            # Create session so user is logged in after verification
            session['user_id'] = user_id
            session['email'] = email
            session.permanent = True

            print(f"[AUTH] Email verified for {email}")

            # Redirect to home page
            return redirect('/')

    except Exception as e:
        print(f"[ERROR] Email verification failed: {e}")
        return redirect('/login?error=verification_failed')


@app.route('/api/auth/resend-verification', methods=['POST', 'OPTIONS'])
def auth_resend_verification():
    """Resend verification email."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication not enabled'}), 503

    # Support both logged-in users and users who can't log in due to unverified
    # email. An address in the body wins over the session: someone signed in on
    # one account and trying to verify a second one was being answered about
    # the account they were signed into, which reported "already verified"
    # while the account they actually named stayed unverified and locked out.
    data = request.get_json(silent=True) or {}
    email_from_request = (data.get('email') or '').strip().lower()
    user_id = None if email_from_request else session.get('user_id')

    if not user_id and not email_from_request:
        return jsonify({'error': 'Email is required'}), 400

    try:
        with get_db() as conn:
            cur = conn.cursor()

            # Get user info - either by user_id or by email
            if user_id:
                cur.execute(
                    'SELECT id, email, email_verified, verification_token_expires FROM users WHERE id = %s',
                    (user_id,)
                )
            else:
                cur.execute(
                    'SELECT id, email, email_verified, verification_token_expires FROM users WHERE email = %s',
                    (email_from_request,)
                )
            user = cur.fetchone()

            if not user:
                # Don't reveal if email exists or not for security
                return jsonify({'status': 'success', 'message': 'If an account exists, a verification email will be sent'}), 200

            user_id, email, email_verified, last_token_expires = user

            if email_verified:
                return jsonify({'error': 'Email already verified'}), 400

            # Rate limit: don't send if token was created less than 60 seconds ago
            if last_token_expires:
                token_created = last_token_expires - timedelta(hours=24)
                if datetime.utcnow() < token_created + timedelta(seconds=60):
                    return jsonify({'error': 'Please wait before requesting another email'}), 429

            # Generate new token
            verification_token = secrets.token_urlsafe(32)
            token_expires = datetime.utcnow() + timedelta(hours=24)

            cur.execute(
                'UPDATE users SET verification_token = %s, verification_token_expires = %s WHERE id = %s',
                (verification_token, token_expires, user_id)
            )

            # Send email
            base_url = os.getenv('SITE_URL', 'https://markremoverai.com')
            verify_url = f"{base_url}/api/auth/verify?token={verification_token}"
            threading.Thread(target=send_verification_email, args=(email, verify_url)).start()

            print(f"[AUTH] Resent verification email to {email}")

            return jsonify({
                'status': 'success',
                'message': 'Verification email sent'
            })

    except Exception as e:
        print(f"[ERROR] Resend verification failed: {e}")
        return jsonify({'error': 'Failed to resend verification email'}), 500


@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Check if user is logged in and get user info."""
    if not AUTH_ENABLED:
        return jsonify({'authenticated': False})

    try:
        user_id = session.get('user_id')

        if not user_id:
            return jsonify({'authenticated': False})

        # Get current user data from database
        with get_db() as conn:
            cur = conn.cursor()
            # Before reading the balance, not after: the app calls this on every
            # launch, so this is what actually hands out the daily allowance.
            grant_daily_free_credits(cur, user_id)
            cur.execute('SELECT email, name, credits, email_verified, created_at FROM users WHERE id = %s', (user_id,))
            user = cur.fetchone()

            if not user:
                session.clear()
                return jsonify({'authenticated': False})

            email, name, credits, email_verified, created_at = user

            return jsonify({
                'authenticated': True,
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': name,
                    'credits': credits,
                    'email_verified': email_verified or False,
                    'created_at': created_at.isoformat() if created_at else None
                }
            })

    except Exception as e:
        print(f"[ERROR] Auth status check failed: {e}")
        return jsonify({'authenticated': False})


@app.route('/api/auth/logout', methods=['POST', 'OPTIONS'])
def auth_logout():
    """Logout user."""
    if request.method == 'OPTIONS':
        return ('', 204)

    session.clear()
    return jsonify({'status': 'success'})


@app.route('/api/auth/delete-account', methods=['POST', 'OPTIONS'])
@require_auth
def delete_account():
    """
    Delete user's own account and all associated data.

    SECURITY: Users can ONLY delete their OWN account.
    Deletes: user record, uploads, results, clears session.
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        user_id = session.get('user_id')
        user_email = session.get('email')

        if not user_id:
            return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401

        print(f"[ACCOUNT DELETE] Starting deletion for user {user_id} ({user_email})")

        with get_db() as conn:
            cur = conn.cursor()

            # Verify user exists and get their data
            cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()

            if not user:
                return jsonify({'status': 'error', 'message': 'User not found'}), 404

            # Double-check: only allow deletion of own account
            if user[0] != user_id:
                print(f"[SECURITY] User {user_id} tried to delete different account!")
                return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403

            # Delete user from database
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))

            print(f"[ACCOUNT DELETE] Deleted user {user_id} ({user_email}) from database")

        # Clear session
        session.clear()

        # Note: User's uploaded files are stored with task_ids (UUIDs), not user IDs
        # They will be cleaned up by the automatic cleanup routine
        # If you want immediate deletion, you'd need to track user->file associations

        return jsonify({
            'status': 'success',
            'message': 'Account deleted successfully'
        })

    except Exception as e:
        print(f"[ERROR] Account deletion failed: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to delete account'}), 500


@app.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    """Request password reset email."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication disabled'}), 500

    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()

        if not email:
            return jsonify({'error': 'Email is required'}), 400

        with get_db() as conn:
            cur = conn.cursor()

            # Check if user exists
            cur.execute('SELECT id, name FROM users WHERE email = %s', (email,))
            user = cur.fetchone()

            # Always return success to prevent email enumeration
            # But only send email if user exists
            if user:
                import datetime

                # Generate reset token
                token = secrets.token_urlsafe(32)
                expires_at = datetime.datetime.now() + datetime.timedelta(hours=1)

                # Store token in database
                cur.execute('''
                    INSERT INTO password_reset_tokens (user_id, token, expires_at)
                    VALUES (%s, %s, %s)
                ''', (user[0], token, expires_at))

                # Generate reset URL and send email
                base_url = os.getenv('TUNNEL_URL', request.host_url.rstrip('/'))
                # Must be the route, not the file name. /reset-password.html
                # redirects to /reset-password and drops the query string with
                # it, so the page loaded with no token and just showed the
                # "request a reset" form again.
                reset_url = f"{base_url}/reset-password?token={token}"

                try:
                    send_reset_email(email, reset_url)
                    print(f"[AUTH] Password reset email sent to {email}")
                except Exception as mail_err:
                    print(f"[WARNING] Email failed for {email}: {mail_err}")
                    # Still return success to prevent email enumeration

        return jsonify({
            'status': 'success',
            'message': 'If an account exists with that email, a password reset link has been sent.'
        })

    except Exception as e:
        print(f"[ERROR] Forgot password failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to process request'}), 500


@app.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    """Reset password using token."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication disabled'}), 500

    try:
        data = request.get_json()
        token = data.get('token', '')
        new_password = data.get('newPassword', '')

        if not token or not new_password:
            return jsonify({'error': 'Token and new password are required'}), 400

        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400

        with get_db() as conn:
            cur = conn.cursor()

            # Find valid token
            cur.execute('''
                SELECT user_id, expires_at
                FROM password_reset_tokens
                WHERE token = %s AND used = FALSE
            ''', (token,))

            reset_token = cur.fetchone()

            if not reset_token:
                return jsonify({'error': 'Invalid or expired reset token'}), 400

            # Check if token expired
            import datetime
            if reset_token[1] < datetime.datetime.now():
                return jsonify({'error': 'Reset token has expired'}), 400

            # Hash new password
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Update password. Verifying the address here is the point: the
            # token only ever reached them by email, so completing this flow is
            # proof they control it. Without this, anyone who signed up but
            # never clicked the original verification link resets their
            # password successfully and is then still refused at login for
            # being unverified, with no way out - the verification token has
            # usually long expired by then.
            cur.execute('''
                UPDATE users
                SET password_hash = %s,
                    email_verified = TRUE,
                    verification_token = NULL,
                    verification_token_expires = NULL
                WHERE id = %s
            ''', (password_hash, reset_token[0]))

            # Mark token as used
            cur.execute('''
                UPDATE password_reset_tokens
                SET used = TRUE
                WHERE token = %s
            ''', (token,))

            print(f"[AUTH] Password reset successful for user_id {reset_token[0]}")

        return jsonify({'status': 'success', 'message': 'Password reset successfully'})

    except Exception as e:
        print(f"[ERROR] Password reset failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to reset password'}), 500


# ----------------------------------------------------------------------------
# End Authentication Routes
# ----------------------------------------------------------------------------

@app.route('/api/debug/files', methods=['GET'])
def debug_files():
    """Debug endpoint to check what files exist in web/ folder"""
    try:
        import glob
        web_files = []
        static_folder = app.static_folder

        # List files in web/ directory
        if os.path.exists(static_folder):
            for root, dirs, files in os.walk(static_folder):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), static_folder)
                    web_files.append(rel_path)

        return jsonify({
            'static_folder': static_folder,
            'static_folder_exists': os.path.exists(static_folder),
            'cwd': os.getcwd(),
            'script_dir': os.path.dirname(os.path.abspath(__file__)),
            'web_files_count': len(web_files),
            'web_files': web_files[:50],  # First 50 files
            'config_exists': os.path.exists(os.path.join(static_folder, 'config.js')),
            'index_exists': os.path.exists(os.path.join(static_folder, 'index.html')),
            'emblem_exists': os.path.exists(os.path.join(static_folder, 'emblem.png'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/celery', methods=['GET'])
def debug_celery():
    """Debug endpoint to check Celery/Redis configuration"""
    try:
        # Mask sensitive passwords in URLs
        def mask_url(url):
            if not url:
                return None
            import re
            return re.sub(r'(redis://[^:]+:)([^@]+)(@)', r'\1****\3', url)

        return jsonify({
            'redis_url_from_file': mask_url(REDIS_URL),
            'celery_broker_env': mask_url(os.getenv('CELERY_BROKER_URL')),
            'celery_backend_env': mask_url(os.getenv('CELERY_RESULT_BACKEND')),
            'celery_broker_config': mask_url(app.config.get('broker_url')),
            'celery_backend_config': mask_url(app.config.get('result_backend')),
            'celery_broker_actual': mask_url(celery.conf.broker_url),
            'celery_backend_actual': mask_url(celery.conf.result_backend),
            'redis_url_file_exists': os.path.exists(os.path.join(SCRIPT_DIR, 'redis_url.txt')),
            'env_file_exists': os.path.exists(os.path.join(SCRIPT_DIR, '.env'))
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/admin/users', methods=['GET'])
def admin_view_users():
    """View latest users - protected by secret key"""
    secret = request.args.get('key')
    if secret != 'markremover2024admin':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute('''
            SELECT id, email, name, credits, email_verified, created_at, google_id
            FROM users
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        users = cur.fetchall()
        cur.close()
        db_pool.putconn(conn)

        return jsonify({
            'users': [{
                'id': u[0],
                'email': u[1],
                'name': u[2],
                'credits': u[3],
                'verified': u[4],
                'created': str(u[5]),
                'google_id': u[6]
            } for u in users]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    # Prevent MIME sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    # XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Content Security Policy (allow Cloudflare, Fonts, data URIs for videos)
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://static.cloudflareinsights.com https://markz.humblewoslayer.workers.dev; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: https:; media-src 'self' data: blob: https://markz.humblewoslayer.workers.dev; connect-src 'self' https:; frame-src 'self' https://www.youtube.com https://youtube.com;"
    # Remove server header
    response.headers.pop('Server', None)
    return response

# ============================================================================
# Configuration - ALL ON D DRIVE
# ============================================================================


# Redis configuration (for queue + caching) - NO LOCALHOST FALLBACK
# Priority: 1) redis_url.txt (local dev), 2) REDIS_URL env var (Railway)
REDIS_URL = None
# SCRIPT_DIR is already defined above as os.path.dirname(os.path.abspath(__file__))
redis_url_file = os.path.join(SCRIPT_DIR, 'redis_url.txt')
if os.path.exists(redis_url_file):
    with open(redis_url_file, 'r') as f:
        REDIS_URL = f.read().strip()
    print(f"[OK] Auto-loaded Redis URL from {redis_url_file}")
    print(f"   Using: {REDIS_URL}")
else:
    # Fallback to environment variable (Railway deployment)
    REDIS_URL = os.getenv('REDIS_URL')
    if REDIS_URL:
        print(f"[OK] Using REDIS_URL from environment")
        print(f"   Using: {REDIS_URL}")
    else:
        print(f"[ERROR] No Redis URL found!")
        print(f"   - redis_url.txt not found at {redis_url_file}")
        print(f"   - REDIS_URL environment variable not set")
        print(f"   Redis connection will fail - set REDIS_URL in Railway or create redis_url.txt")

app.config['broker_url'] = REDIS_URL
app.config['result_backend'] = REDIS_URL  # Store results in Redis
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR  # D drive only!
app.config['TEMP_FOLDER'] = TEMP_DIR  # D drive only!

# Rate limiting - prevent abuse
UPLOAD_RATE_LIMIT = {}  # IP -> (count, timestamp)
USER_VIDEO_RATE_LIMIT = {}  # user_id -> list of timestamps

# Video processing limits
MAX_VIDEO_DURATION_SECONDS = 90  # 1 min 30 sec max
MAX_VIDEO_FPS = 60  # 60fps max

# Input validation
def sanitize_filename(filename):
    """Remove dangerous characters from filenames"""
    import re
    # Remove path traversal attempts
    filename = os.path.basename(filename)
    # Only allow alphanumeric, dots, dashes, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename


# ============================================
# FILE SECURITY - Magic Bytes Validation
# ============================================
# Validates file type by checking actual file content (magic bytes)
# NOT just the extension - prevents malicious file uploads

ALLOWED_FILE_SIGNATURES = {
    # Video formats
    'mp4': [
        (4, b'ftyp'),           # MP4/M4V - ftyp at offset 4
        (4, b'ftypmp4'),        # MP4
        (4, b'ftypisom'),       # MP4 ISO
        (4, b'ftypMSNV'),       # MP4 Sony
        (4, b'ftypavc1'),       # MP4 AVC
    ],
    'mov': [
        (4, b'ftypqt'),         # QuickTime
        (4, b'ftyp'),           # MOV also uses ftyp
        (4, b'moov'),           # Old MOV format
        (0, b'moov'),           # Alternative MOV
    ],
    'avi': [
        (0, b'RIFF'),           # AVI starts with RIFF
    ],
    'webm': [
        (0, b'\x1a\x45\xdf\xa3'),  # WebM/Matroska
    ],
    'mkv': [
        (0, b'\x1a\x45\xdf\xa3'),  # MKV/Matroska
    ],
    # Image formats
    'jpg': [
        (0, b'\xff\xd8\xff'),   # JPEG
    ],
    'jpeg': [
        (0, b'\xff\xd8\xff'),   # JPEG
    ],
    'png': [
        (0, b'\x89PNG'),        # PNG
    ],
    'gif': [
        (0, b'GIF87a'),         # GIF87
        (0, b'GIF89a'),         # GIF89
    ],
    'webp': [
        (0, b'RIFF'),           # WebP (also uses RIFF, check for WEBP later)
    ],
}

def validate_file_magic_bytes(file_stream, claimed_extension):
    """
    Validate that file content matches claimed extension using magic bytes.
    Returns (is_valid, detected_type, error_message)

    This is a SECURITY measure to prevent:
    - Uploading executables disguised as videos
    - Malware hidden in fake media files
    - Script injection via file uploads
    """
    try:
        # Read first 32 bytes for signature check
        file_stream.seek(0)
        header = file_stream.read(32)
        file_stream.seek(0)  # Reset for later use

        if len(header) < 8:
            return False, None, "File too small to validate"

        # Normalize extension
        ext = claimed_extension.lower().lstrip('.')

        # Check if extension is in our whitelist
        if ext not in ALLOWED_FILE_SIGNATURES:
            return False, None, f"File type '.{ext}' not allowed. Allowed: mp4, mov, avi, webm, mkv, jpg, png, gif"

        # Check magic bytes for this extension
        signatures = ALLOWED_FILE_SIGNATURES[ext]
        for offset, signature in signatures:
            if offset + len(signature) <= len(header):
                if header[offset:offset + len(signature)] == signature:
                    # Additional check for AVI vs WebP (both use RIFF)
                    if ext == 'avi' and b'AVI ' not in header[:16]:
                        continue
                    if ext == 'webp' and b'WEBP' not in header[:16]:
                        continue
                    return True, ext, None

        # Signature didn't match - could be malicious
        # Log what we actually found for debugging
        detected = "unknown"
        if header[:4] == b'RIFF':
            if b'AVI ' in header[:16]:
                detected = "avi"
            elif b'WEBP' in header[:16]:
                detected = "webp"
        elif header[4:8] == b'ftyp':
            detected = "mp4/mov"
        elif header[:3] == b'\xff\xd8\xff':
            detected = "jpg"
        elif header[:4] == b'\x89PNG':
            detected = "png"
        elif header[:4] == b'GIF8':
            detected = "gif"
        elif header[:2] == b'MZ':
            detected = "EXECUTABLE (BLOCKED!)"
        elif header[:4] == b'PK\x03\x04':
            detected = "ZIP/JAR (BLOCKED!)"
        elif b'<script' in header.lower() or b'<?php' in header.lower():
            detected = "SCRIPT (BLOCKED!)"

        return False, detected, f"File content doesn't match '.{ext}' extension. Detected: {detected}"

    except Exception as e:
        return False, None, f"Error validating file: {str(e)}"


def is_path_safe(filepath, allowed_directory):
    """
    Check if filepath is safely within allowed_directory.
    Prevents path traversal attacks like ../../etc/passwd
    """
    # Resolve to absolute paths
    filepath = os.path.abspath(filepath)
    allowed_directory = os.path.abspath(allowed_directory)

    # Check if filepath starts with allowed directory
    return filepath.startswith(allowed_directory + os.sep) or filepath == allowed_directory

def get_tunnel_url():
    """Get tunnel URL from environment variable or file"""
    # Check environment variable first
    env_url = os.getenv('TUNNEL_URL')
    if env_url:
        return env_url.strip()

    # Check tunnel_output.txt (localtunnel format: "your url is: https://...")
    tunnel_file = os.path.join(SCRIPT_DIR, 'tunnel_output.txt')
    if os.path.exists(tunnel_file):
        try:
            with open(tunnel_file, 'r') as f:
                for line in f:
                    if 'your url is:' in line:
                        url = line.split('your url is:')[1].strip()
                        if url:
                            return url
        except Exception:
            pass

    # Check web/tunnel_url.txt (alternative format)
    tunnel_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
    if os.path.exists(tunnel_file):
        try:
            with open(tunnel_file, 'r') as f:
                url = f.read().strip()
                if url:
                    return url
        except Exception:
            pass

    return None

def validate_url(url):
    """Validate and sanitize URLs to prevent SSRF attacks"""
    from urllib.parse import urlparse

    if not url or not isinstance(url, str):
        return False

    # Basic length check
    if len(url) > 2048:
        return False

    try:
        parsed = urlparse(url)

        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False

        # Only allow http/https
        if parsed.scheme not in ['http', 'https']:
            return False

        # Block localhost/internal IPs to prevent SSRF
        blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
        if any(blocked in parsed.netloc.lower() for blocked in blocked_hosts):
            return False

        # Block private IP ranges
        import ipaddress
        try:
            # Extract hostname without port
            hostname = parsed.netloc.split(':')[0]
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            # Not an IP address, that's fine (it's a domain)
            pass

        return True
    except Exception:
        return False


def get_dynamic_subvideo_length(width, height):
    """
    Calculate optimal subvideo length based on video resolution for memory efficiency.
    Higher resolution videos need smaller chunks to fit in GPU memory.
    """
    resolution = width * height
    
    if resolution <= 640 * 480:        # 640p and below
        return 100, 12  # subvideo_length, short_clip_len
    elif resolution <= 1280 * 720:     # 720p
        return 80, 8
    elif resolution <= 1920 * 1080:    # 1080p  
        return 60, 6
    elif resolution <= 2560 * 1440:    # 1440p
        return 40, 4
    else:                              # 4K and above
        return 20, 2

def clear_gpu_memory():
    """
    Strategic GPU memory clearing for optimal memory management.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Removed torch.cuda.synchronize() - was creating hard barrier between parallel segments
            print(f"[CLEANUP] GPU memory cleared: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")
    except Exception as e:
        print(f"[WARNING]  GPU memory clear failed: {e}")
        pass

def performance_checkpoint(stage_name, start_time=None):
    """
    Performance checkpoint logging for identifying bottlenecks.
    """
    import time
    current_time = time.perf_counter()
    
    if start_time is not None:
        elapsed = current_time - start_time
        print(f"[TIME]  {stage_name} completed in {elapsed:.2f} seconds")
    else:
        print(f"[RUNNING] {stage_name} started")
    
    return current_time


def _init_gpu_worker(gpu_id):
    """
    Initialize worker process with specific GPU assignment.
    Each worker gets exclusive access to one GPU.
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Worker initialized with GPU {gpu_id}")


def _process_propainter_segment(seg_idx, total_segments, segment, context):
    """
    Process a single smart-cropped segment with faster-propainter.
    Returns frames directory ready for encoding.
    Runs in separate process with assigned GPU via CUDA_VISIBLE_DEVICES.

    Returns a tuple (seg_idx, seg_cleaned_dir, crop_info, temp_dirs_to_cleanup).
    """
    from crop_utils import calculate_crop_region, estimate_speedup

    start_frame, end_frame, seg_bbox = segment
    seg_duration = end_frame - start_frame + 1

    base_name = context['base_name']
    unique_suffix = context['unique_suffix']
    width = context['width']
    height = context['height']
    mask_dir = context['mask_dir']
    original_frames_dir = context['original_frames_dir']

    seg_label = f"{seg_idx + 1}/{total_segments}"
    print(f"\n[SEGMENT] Processing segment {seg_label}: frames {start_frame}-{end_frame} ({seg_duration} frames)")

    crop_x, crop_y, crop_w, crop_h = calculate_crop_region(seg_bbox, width, height, padding_ratio=0.2, min_size=128)
    speedup = estimate_speedup((width, height), (crop_w, crop_h))
    print(f"   [CROP] Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h} (estimated {speedup:.1f}x speedup)")

    seg_prefix = f"{base_name}_{unique_suffix}_seg{seg_idx}"
    seg_frames_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_frames")
    seg_cropped_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cropped")
    seg_mask_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_masks")
    seg_output_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_output")
    seg_cleaned_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cleaned")

    for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
        os.makedirs(path, exist_ok=True)

    try:
        frames_copied = 0
        for frame_idx in range(start_frame, end_frame + 1):
            src = os.path.join(original_frames_dir, f"{frame_idx:04d}.png")
            dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
            if not os.path.exists(src):
                print(f"[WARNING]  Warning: Frame {frame_idx} ({frame_idx:04d}.png) not found, skipping")
                continue
            shutil.copy2(src, dst)
            frames_copied += 1

        if frames_copied == 0:
            raise RuntimeError(f"No frames copied for segment {seg_idx}")

        print(f"   [CROP]  Cropping {frames_copied} frames to {crop_w}x{crop_h}...")
        for frame_idx in range(frames_copied):
            frame_file = f"{frame_idx:04d}.png"
            frame_path = os.path.join(seg_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[WARNING]  Warning: Could not read frame {frame_file}, skipping crop")
                continue
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            cv2.imwrite(os.path.join(seg_cropped_dir, frame_file), cropped)

        masks_copied = 0
        for frame_idx in range(start_frame, end_frame + 1):
            mask_file = f"{frame_idx:04d}.png"
            mask_src = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_src):
                mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    cv2.imwrite(os.path.join(seg_mask_dir, f"{masks_copied:04d}.png"), cropped_mask)
                    masks_copied += 1

        print(f"   [PAINT] Running faster-propainter pipeline on cropped segment...")
        try:
            # Use cached ProPainter pipeline (pre-loaded at worker startup)
            faster_propainter_pipeline = get_propainter_pipeline()

            import torch
            use_fp16 = torch.cuda.is_available()

            print(f"   [RUNNING] Direct pipeline: segment {seg_idx+1}, resolution={crop_w}x{crop_h}, neighbor_length=10, ref_stride=10, subvideo_length=120, raft_iter=20, FP16={use_fp16}")

            # Each process gets its own CUDA context for true parallel processing
            faster_propainter_pipeline(
                video=seg_cropped_dir,
                mask=seg_mask_dir,
                output=seg_output_dir,
                resize_ratio=1.0,
                mask_dilation=4,
                ref_stride=10,
                neighbor_length=10,
                subvideo_length=120,
                raft_iter=20,
                mode="video_inpainting",
                save_frames=True,
                fp16=use_fp16
            )

            print(f"   [OK] faster-propainter segment {seg_idx+1} completed")

        except Exception as exc:
            print(f"[ERROR] faster-propainter failed on segment {seg_idx}: {exc}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"faster-propainter failed on segment {seg_idx}: {exc}") from exc

        seg_propainter_frames = os.path.join(seg_output_dir, os.path.basename(seg_cropped_dir), 'frames')
        if not os.path.exists(seg_propainter_frames):
            raise RuntimeError(f"ProPainter output frames not found for segment {seg_idx}")

        print(f"   🔗 Merging cleaned region back to full frames (GPU-accelerated)...")

        # Try GPU-accelerated merge first, fallback to CPU if needed
        try:
            import torch
            if torch.cuda.is_available():
                # GPU-accelerated batch processing
                print(f"   [RUNNING] Using GPU for frame merging...")
                for frame_idx in range(seg_duration):
                    frame_file = f"{frame_idx:04d}.png"
                    orig = cv2.imread(os.path.join(seg_frames_dir, frame_file))
                    clean = cv2.imread(os.path.join(seg_propainter_frames, frame_file))

                    if clean is not None and orig is not None:
                        # Convert to GPU tensors (BGR to RGB not needed for merging)
                        orig_gpu = torch.from_numpy(orig).cuda()
                        clean_gpu = torch.from_numpy(clean).cuda()

                        # Merge on GPU
                        orig_gpu[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = clean_gpu

                        # Copy back to CPU and save
                        result_frame = orig_gpu.cpu().numpy()
                        cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), result_frame)
                    elif orig is not None:
                        cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), orig)

                print(f"   [OK] GPU merge completed")
            else:
                raise RuntimeError("CUDA not available")
        except Exception as e:
            # Fallback to CPU merge
            print(f"   [WARNING]  GPU merge failed ({e}), falling back to CPU...")
            original_frames = []
            cleaned_frames = []
            for frame_idx in range(seg_duration):
                frame_file = f"{frame_idx:04d}.png"
                orig = cv2.imread(os.path.join(seg_frames_dir, frame_file))
                clean = cv2.imread(os.path.join(seg_propainter_frames, frame_file))
                original_frames.append(orig)
                cleaned_frames.append(clean)

            # Merge in memory and write in one pass
            for frame_idx, (original, cleaned_crop) in enumerate(zip(original_frames, cleaned_frames)):
                frame_file = f"{frame_idx:04d}.png"
                if cleaned_crop is not None and original is not None:
                    result_frame = original.copy()
                    result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop
                    cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), result_frame)
                elif original is not None:
                    cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), original)

        # Return cleaned frames directory for later encoding
        temp_dirs = [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir]
        return seg_idx, seg_cleaned_dir, {'fps': context['fps']}, temp_dirs

    except Exception as e:
        # Cleanup on error
        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
            shutil.rmtree(path, ignore_errors=True)
        raise


def _encode_segment(seg_idx, total_segments, seg_cleaned_dir, context, temp_dirs_to_cleanup):
    """
    Encode a segment's cleaned frames to video (GPU-accelerated with NVENC).

    Returns a tuple (seg_idx, seg_video_path).
    """
    import subprocess

    base_name = context['base_name']
    unique_suffix = context['unique_suffix']
    fps = context['fps']

    seg_prefix = f"{base_name}_{unique_suffix}_seg{seg_idx}"
    seg_video_path = os.path.join(TEMP_DIR, f"{seg_prefix}.mp4")

    seg_label = f"{seg_idx + 1}/{total_segments}"
    print(f"   [ENCODE]  Encoding segment {seg_label} to video (GPU NVENC)...")

    try:
        # Try faster preset p1 for maximum speed, fallback to p4 if fails
        encode_cmd = [
            FFMPEG_EXE, '-y',
            '-framerate', str(fps),
            '-i', os.path.join(seg_cleaned_dir, '%04d.png'),
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',  # Balanced quality/speed (was p1)
            '-b:v', '15M',  # Higher bitrate for better quality
            '-bufsize', '16M',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            seg_video_path
        ]
        result = subprocess.run(encode_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback to p4 if p5 fails
            print(f"   [WARNING]  p5 preset failed, trying p4...")
            encode_cmd[encode_cmd.index('-preset') + 1] = 'p4'
            subprocess.run(encode_cmd, capture_output=True, check=True)

        print(f"   [OK] Segment {seg_label} encoded successfully")
        return seg_idx, seg_video_path

    finally:
        # Cleanup temp directories after encoding
        for path in temp_dirs_to_cleanup + [seg_cleaned_dir]:
            shutil.rmtree(path, ignore_errors=True)


def save_detection_debug(image, mask, detections, prefix):
    """
    Save a debug visualization showing YOLO detections and the inpainting mask.

    Args:
        image: Original frame (H, W, 3)
        mask:  Binary mask aligned with image (H, W)
        detections: List of detection dicts with 'bbox'
        prefix: Filename prefix (str)
    """
    if not detections or image is None or mask is None:
        return None

    try:
        overlay = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if mask.shape[:2] != image.shape[:2]:
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        else:
            mask_resized = mask

        mask_color = np.zeros_like(overlay)
        mask_color[:, :, 2] = np.clip(mask_resized, 0, 255)

        debug_image = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
        filename = f"{prefix}.png"
        output_path = os.path.join(DEBUG_DIR, filename)
        cv2.imwrite(output_path, debug_image)
        print(f"🧪 Detection debug saved: {output_path}")
        return output_path
    except Exception as exc:
        print(f"[WARNING]  Failed to save detection debug image: {exc}")
        return None

# Initialize Celery - NO LOCALHOST FALLBACK
# Priority: CELERY_BROKER_URL env var -> REDIS_URL env var -> redis_url.txt (already loaded into REDIS_URL)
broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL") or REDIS_URL
backend = os.getenv("CELERY_RESULT_BACKEND") or broker

if not broker:
    print("[ERROR] No Celery broker URL configured!")
    print("   Set CELERY_BROKER_URL or REDIS_URL environment variable")
    raise ValueError("Celery broker URL is required - no localhost fallback allowed")

print(f"[DEBUG] Celery Broker = {broker}")
print(f"[DEBUG] Celery Backend = {backend}")

celery = Celery(app.name, broker=broker, backend=backend)
celery.conf.update(app.config)

# Celery configuration for production
celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minute timeout
    worker_prefetch_multiplier=4,  # Prefetch up to 4 tasks for TRUE parallel execution
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (prevent memory leaks)
    result_expires=3600,  # Results expire after 1 hour
    broker_connection_retry_on_startup=True,
    # Fix connection hanging and task pickup blocking
    broker_pool_limit=10,  # Celery default - stable connection pool (was 1 = too restrictive, None = connection churn)
    broker_connection_timeout=10,  # 10 second timeout for broker connection (increased from 3)
    broker_transport_options={
        'visibility_timeout': 300,  # 5 minutes (default 3600) - tasks become visible again after 5min if worker crashes
    },
    result_backend_transport_options={'socket_connect_timeout': 10},
    task_ignore_result=False,  # We need results for status tracking
    task_acks_late=True,
    worker_disable_rate_limits=True,  # Disable rate limiting to prevent task pickup delays
)

# ============================================================================
# BACKGROUND ENCODER THREAD (Real-Time Continuous Encoding)
# ============================================================================
# Workers signal when segments ready → Background thread encodes immediately
# Encoding happens continuously as segments complete (not all at once at end)
# Everything stays on ONE GPU for extreme speed

import threading

encoder_thread = None

def start_background_encoder(**kwargs):
    """
    Start background encoding thread when worker process initializes.
    This runs ONCE per worker process.
    """
    global encoder_thread

    print("[BACKGROUND ENCODER INIT] Worker process starting, initializing background encoder...")

    if encoder_thread is None or not encoder_thread.is_alive():
        print("[BACKGROUND ENCODER] Starting real-time encoding thread...")
        encoder_thread = threading.Thread(
            target=background_encoder_worker,
            daemon=True,  # Dies with worker process
            name="BackgroundEncoder"
        )
        encoder_thread.start()
        print("[BACKGROUND ENCODER] Thread active - will encode segments as they complete!")
    else:
        print("[BACKGROUND ENCODER] Thread already running")


def background_encoder_worker():
    """
    Background thread that encodes segments in real-time as they complete.
    Uses Redis pub/sub to receive segment completion notifications.
    PARALLEL ENCODING: Encodes multiple segments simultaneously using ThreadPoolExecutor.
    """
    import json
    import traceback
    import time
    from concurrent.futures import ThreadPoolExecutor

    # Track videos being processed
    video_futures = {}  # {video_id: {seg_idx: future, ...}}
    video_metadata = {}  # {video_id: {'total_segments': N, 'redis_client': client}}

    while True:  # Auto-reconnect loop
        try:
            redis_client = celery.backend.client

            # Configure pubsub with NO timeout
            # NOTE: socket_keepalive_options disabled - causes "Invalid argument" error in WSL2/Docker
            pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
            pubsub.connection_pool.connection_kwargs['socket_timeout'] = None
            pubsub.connection_pool.connection_kwargs['socket_keepalive'] = False  # Disabled for WSL2 compatibility

            pubsub.subscribe('segment_ready')

            print("[BACKGROUND ENCODER] Listening for segment completion signals...")
            print("[BACKGROUND ENCODER] Socket keepalive enabled - NO timeout!")
            print("[BACKGROUND ENCODER] UNIFIED MODE: All encoding happens at finalization (no segment boundaries!)")

            # Create thread pool for parallel encoding (4 matches SEGMENT_WORKERS)
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="EncoderThread") as executor:
                for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            video_id = data['video_id']
                            seg_idx = data['seg_idx']
                            total_segments = data['total_segments']

                            # Initialize tracking for new video
                            if video_id not in video_futures:
                                video_futures[video_id] = {}
                                video_metadata[video_id] = {
                                    'total_segments': total_segments,
                                    'redis_client': redis_client
                                }

                            print(f"[BACKGROUND ENCODER] Segment {seg_idx+1}/{total_segments} ready for video {video_id} - submitting to parallel encoder!")

                            # Submit encoding to thread pool (NON-BLOCKING!)
                            future = executor.submit(encode_segment_background, redis_client, data)
                            video_futures[video_id][seg_idx] = future

                            # Check if all segments have been submitted for this video
                            if len(video_futures[video_id]) == total_segments:
                                print(f"[BACKGROUND ENCODER] All {total_segments} segments submitted for {video_id} - waiting for completion...")

                                # Wait for all encoding futures to complete
                                completed_count = 0
                                failed_segments = []

                                for seg_idx, future in video_futures[video_id].items():
                                    try:
                                        future.result()  # Block until this segment completes
                                        completed_count += 1
                                        print(f"[BACKGROUND ENCODER] ✓ Segment {seg_idx+1} encoded! Progress: {completed_count}/{total_segments}")
                                    except Exception as e:
                                        print(f"[BACKGROUND ENCODER ERROR] Segment {seg_idx+1} failed: {e}")
                                        traceback.print_exc()
                                        failed_segments.append(seg_idx)

                                # Finalize if all segments succeeded
                                if not failed_segments:
                                    print(f"[BACKGROUND ENCODER] All {total_segments} segments encoded for {video_id}! Triggering finalization...")
                                    trigger_finalization(redis_client, video_id, total_segments)
                                    print(f"[BACKGROUND ENCODER] ✓ Video {video_id} finalized!")
                                else:
                                    print(f"[BACKGROUND ENCODER ERROR] Video {video_id} had {len(failed_segments)} failed segments: {failed_segments}")

                                # Cleanup tracking
                                del video_futures[video_id]
                                del video_metadata[video_id]

                        except Exception as e:
                            print(f"[BACKGROUND ENCODER ERROR] Failed to process segment: {e}")
                            traceback.print_exc()

        except Exception as e:
            print(f"[BACKGROUND ENCODER] Connection lost: {e}")
            traceback.print_exc()
            print("[BACKGROUND ENCODER] Reconnecting in 2 seconds...")
            time.sleep(2)
            # Loop continues - auto-reconnect!


def encode_segment_background(redis_client, data):
    """
    Mark segment as ready for finalization.

    FIX: No longer encodes per-segment videos! All encoding happens in
    trigger_finalization() which encodes the ENTIRE video in one pass.
    This eliminates segment boundary flickering.
    """
    video_id = data['video_id']
    seg_idx = data['seg_idx']
    total_segments = data['total_segments']

    # Get segment metadata from Redis to verify it exists
    segment_key = f"video:{video_id}:segment:{seg_idx}"
    segment_info_raw = redis_client.hgetall(segment_key)

    if not segment_info_raw:
        raise RuntimeError(f"Segment metadata not found in Redis: {segment_key}")

    # Decode bytes to strings (Redis returns bytes)
    segment_info = {
        k.decode() if isinstance(k, bytes) else k:
        v.decode() if isinstance(v, bytes) else v
        for k, v in segment_info_raw.items()
    }

    cleaned_dir = segment_info.get('cleaned_dir')
    start_frame = int(segment_info.get('start_frame', 0))
    end_frame = int(segment_info.get('end_frame', 0))

    if not cleaned_dir or not os.path.exists(cleaned_dir):
        raise RuntimeError(f"Cleaned frames directory not found: {cleaned_dir}")

    # 🔥 SKIP per-segment encoding - finalization encodes entire video at once!
    # Just mark segment as ready
    print(f"[ENCODER] Segment {seg_idx} ready: frames {start_frame}-{end_frame} (encoding deferred to finalization)")
    redis_client.hset(segment_key, 'status', 'ready_for_finalization')


def trigger_finalization(redis_client, video_id, total_segments):
    """
    Encode entire video at once from shared buffer, then merge audio.
    Called automatically when all segments complete.

    FIX: Encodes all frames in one pass instead of concatenating per-segment videos.
    This eliminates segment boundary flickering caused by encoder state reset.
    """
    import subprocess
    import time as encode_time

    print(f"\n[FINALIZE] Starting finalization for video {video_id}")

    # Get shared_cleaned_dir and frame range from segments
    segment_key = f"video:{video_id}:segment:0"
    shared_cleaned_dir_raw = redis_client.hget(segment_key, 'cleaned_dir')
    if not shared_cleaned_dir_raw:
        raise RuntimeError(f"No cleaned_dir found for video {video_id}")
    shared_cleaned_dir = shared_cleaned_dir_raw.decode() if isinstance(shared_cleaned_dir_raw, bytes) else shared_cleaned_dir_raw

    fps_raw = redis_client.hget(segment_key, 'fps')
    fps = float(fps_raw.decode() if isinstance(fps_raw, bytes) else fps_raw) if fps_raw else 30.0

    # Find the full frame range across all segments
    min_frame = float('inf')
    max_frame = 0
    for seg_idx in range(total_segments):
        seg_key = f"video:{video_id}:segment:{seg_idx}"
        start_raw = redis_client.hget(seg_key, 'start_frame')
        end_raw = redis_client.hget(seg_key, 'end_frame')
        if start_raw:
            start_frame = int(start_raw.decode() if isinstance(start_raw, bytes) else start_raw)
            min_frame = min(min_frame, start_frame)
        if end_raw:
            end_frame = int(end_raw.decode() if isinstance(end_raw, bytes) else end_raw)
            max_frame = max(max_frame, end_frame)

    if min_frame == float('inf'):
        min_frame = 0

    total_frames = max_frame - min_frame + 1
    print(f"[FINALIZE] Shared buffer: {shared_cleaned_dir}")
    print(f"[FINALIZE] Frame range: {min_frame}-{max_frame} ({total_frames} frames) @ {fps} fps")

    # Get video metadata (decode bytes)
    base_name_raw = redis_client.get(f"video:{video_id}:base_name")
    base_name = base_name_raw.decode() if isinstance(base_name_raw, bytes) else (base_name_raw or 'video')

    video_path_raw = redis_client.get(f"video:{video_id}:video_path")
    video_path = video_path_raw.decode() if isinstance(video_path_raw, bytes) else video_path_raw

    # 🔥 ENCODE ENTIRE VIDEO AT ONCE - no segment concatenation!
    # This eliminates segment boundary flickering by keeping encoder state continuous
    file_list_path = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_all_frames.txt")
    frames_found = 0
    with open(file_list_path, 'w') as f:
        for global_idx in range(min_frame, max_frame + 1):
            frame_path = os.path.join(shared_cleaned_dir, f"{global_idx:04d}.png")
            if os.path.exists(frame_path):
                abs_path = os.path.abspath(frame_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {1/fps}\n")
                frames_found += 1
        # Last frame for proper duration
        last_frame_path = os.path.join(shared_cleaned_dir, f"{max_frame:04d}.png")
        if os.path.exists(last_frame_path):
            abs_path = os.path.abspath(last_frame_path).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    print(f"[FINALIZE] Found {frames_found}/{total_frames} frames in shared buffer")

    temp_processed = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_processed.mp4")

    # Encode with NVENC in ONE PASS - continuous encoder state = no flicker!
    encode_start = encode_time.time()
    encode_cmd = [
        FFMPEG_EXE, '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c:v', 'h264_nvenc',
        '-preset', 'p5',
        '-b:v', '15M',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'main',
        temp_processed
    ]

    print(f"[FINALIZE] Encoding entire video with NVENC (single pass, no segment boundaries)...")
    subprocess.run(encode_cmd, capture_output=True, check=True, text=True, timeout=600)
    encode_duration = encode_time.time() - encode_start
    encode_fps = frames_found / encode_duration if encode_duration > 0 else 0
    encoded_size_mb = os.path.getsize(temp_processed) / (1024 * 1024)
    print(f"[FINALIZE] ✓ Encoded: {encoded_size_mb:.2f} MB in {encode_duration:.2f}s ({encode_fps:.1f} fps)")

    # Merge audio from original
    final_output = os.path.join(RESULT_DIR, f"{base_name}_propainter.mp4")

    if video_path and os.path.exists(video_path):
        # Check if original has audio
        check_audio_cmd = [
            FFPROBE_EXE, '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
        has_audio = 'audio' in has_audio_check.stdout

        if has_audio:
            print(f"[FINALIZE] Merging audio from original...")
            merge_cmd = [
                FFMPEG_EXE, '-y',
                '-i', temp_processed,
                '-i', video_path,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-shortest',
                final_output
            ]
            subprocess.run(merge_cmd, capture_output=True, check=True, text=True, timeout=300)
            if os.path.exists(temp_processed):
                os.remove(temp_processed)
            print(f"[FINALIZE] ✓ Audio merged")
        else:
            os.rename(temp_processed, final_output)
            print(f"[FINALIZE] No audio in original")
    else:
        os.rename(temp_processed, final_output)
        print(f"[FINALIZE] Using processed video only")

    # Cleanup
    if os.path.exists(file_list_path):
        os.remove(file_list_path)

    # 🔥 SHARED BUFFER: Cleanup shared frame directory after finalization
    if shared_cleaned_dir and os.path.exists(shared_cleaned_dir):
        print(f"[FINALIZE] Cleaning up shared frame buffer: {shared_cleaned_dir}")
        shutil.rmtree(shared_cleaned_dir, ignore_errors=True)

    final_size_mb = os.path.getsize(final_output) / (1024 * 1024)
    print(f"[FINALIZE] ✓ Final video ready: {final_output} ({final_size_mb:.2f} MB)")

    # Upload to B2 + Cloudflare CDN (replaced Railway HTTP POST upload)
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
            remote_path = f"results/{timestamp}_{os.path.basename(final_output)}"

            print(f"[FINALIZE] Uploading to B2: {B2_BUCKET}/{remote_path}")
            _b2_start = _upload_time.time()
            info = InMemoryAccountInfo()
            b2_api = B2Api(info)
            b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
            bucket = b2_api.get_bucket_by_name(B2_BUCKET)
            bucket.upload_local_file(local_file=final_output, file_name=remote_path)
            cdn_url = f"{B2_CDN_URL}/{remote_path}"
            _b2_time = _upload_time.time() - _b2_start
            print(f"[FINALIZE] B2 upload complete in {_b2_time:.1f}s - CDN URL: {cdn_url}")

            # Notify Railway server of the CDN URL
            tunnel = os.getenv('TUNNEL_URL') or os.getenv('API_BASE_URL')
            if tunnel:
                try:
                    import requests
                    notify_url = tunnel.rstrip('/') + '/api/notify-result'
                    resp = requests.post(notify_url, json={'video_id': video_id, 'cdn_url': cdn_url}, timeout=10)
                    if resp.ok:
                        print(f"[FINALIZE] ✅ Notified Railway server of CDN URL")
                    else:
                        print(f"[FINALIZE] ⚠️ Railway notification failed: HTTP {resp.status_code}")
                except Exception as notify_err:
                    print(f"[FINALIZE] ⚠️ Railway notification error: {notify_err}")
    except ImportError:
        print(f"[FINALIZE] b2sdk not installed - skipping B2 upload")
    except Exception as e:
        print(f"[FINALIZE] B2 upload failed: {e}")
        import traceback
        traceback.print_exc()

    # Store CDN URL (or local path as fallback) in Redis
    redis_client.set(f"video:{video_id}:final_path", cdn_url or final_output)
    redis_client.set(f"video:{video_id}:status", "complete")

    # Track processing time for queue ETA calculation
    try:
        import time as time_module
        start_time_raw = redis_client.get(f"video:{video_id}:start_time")
        if start_time_raw:
            start_time = float(start_time_raw if isinstance(start_time_raw, str) else start_time_raw.decode())
            processing_duration = time_module.time() - start_time
            # Store in recent processing times list (keep last 20)
            redis_client.lpush('processing_times:recent', str(processing_duration))
            redis_client.ltrim('processing_times:recent', 0, 19)
            print(f"[FINALIZE] Processing time recorded: {processing_duration:.1f}s for video {video_id}")
    except Exception as e:
        print(f"[FINALIZE] Could not record processing time: {e}")

    # 🔥 FIX: Update distributed tracking to mark all segments complete
    # Status endpoint checks segments:{video_id} to see progress
    # Without this, frontend shows "Segment 0/X complete" forever
    tracking_key = f"segments:{video_id}"
    total_segments_bytes = redis_client.get(f"{tracking_key}:total")
    if total_segments_bytes:
        redis_client.set(tracking_key, int(total_segments_bytes))  # Mark all segments complete
        print(f"[FINALIZE] ✅ Marked all {int(total_segments_bytes)} segments complete in Redis tracking")

# ============================================================================
# REDIS VIDEO DOWNLOAD POLLING (for multi-PC parallel downloads)
# ============================================================================

def start_redis_download_poller():
    """
    Background thread that polls Redis for video download signals.
    When prepare_video sets 'video_download:{video_id}' key, this thread
    detects it and downloads the video to cache immediately.

    This enables ALL workers to download in parallel instead of sitting idle.
    """
    import threading
    import time
    import requests

    def poll_redis():
        print("[POLL] Starting Redis video download poller...")
        redis_client = celery.backend.client
        checked_videos = set()  # Track videos we've already processed

        while True:
            try:
                # Scan for all video_download:* keys
                keys = redis_client.keys('video_download:*')

                for key in keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    video_id = key_str.replace('video_download:', '')

                    # Skip if we've already downloaded this video
                    if video_id in checked_videos:
                        continue

                    # Get download URL from Redis
                    download_url = redis_client.get(key)
                    if not download_url:
                        continue

                    download_url = download_url.decode('utf-8') if isinstance(download_url, bytes) else download_url

                    # Check if we already have this video cached
                    cache_dir = os.path.join(TEMP_DIR, 'video_cache')
                    os.makedirs(cache_dir, exist_ok=True)
                    cached_video = os.path.join(cache_dir, f"{video_id}.mp4")

                    if os.path.exists(cached_video):
                        print(f"[OK] Worker {os.getpid()}: Video {video_id} already cached (skip)")
                        checked_videos.add(video_id)
                        continue

                    # Download video to cache
                    print(f"📥 Worker {os.getpid()}: Detected download signal for {video_id}")
                    print(f"   [DOWNLOAD]  Downloading: {download_url}")

                    try:
                        r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=120)
                        r.raise_for_status()

                        with open(cached_video, 'wb') as f:
                            f.write(r.content)

                        file_size = os.path.getsize(cached_video) / (1024 * 1024)
                        print(f"   [OK] Worker {os.getpid()}: Downloaded and cached {video_id} ({file_size:.1f}MB)")
                        checked_videos.add(video_id)

                    except Exception as download_error:
                        print(f"   [ERROR] Worker {os.getpid()}: Download failed for {video_id}: {download_error}")
                        # Don't add to checked_videos so we can retry

                # Clean up checked_videos if it gets too large (memory leak prevention)
                if len(checked_videos) > 100:
                    checked_videos.clear()

            except Exception as e:
                print(f"[WARNING]  Redis poller error: {e}")

            # Poll every 1 second
            time.sleep(1)

    # Start polling thread
    poller_thread = threading.Thread(target=poll_redis, daemon=True)
    poller_thread.start()
    print("[OK] Redis video download poller started")


# Start poller when Celery worker is ready
from celery.signals import worker_ready

@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    """Called when Celery worker finishes initialization - pre-warm models"""
    print("[INIT] Worker ready - warming up models...")

    # ============================================================================
    # ENVIRONMENT VARIABLE VALIDATION - Critical for performance!
    # ============================================================================
    print("=" * 70)
    print("WORKER ENVIRONMENT VALIDATION")
    print("=" * 70)

    # Check critical performance environment variables
    env_vars_status = {
        'USE_NEUFLOW': os.environ.get('USE_NEUFLOW', '0'),
        'FORCE_TRT_RAFT': os.environ.get('FORCE_TRT_RAFT', '0'),
        'ENABLE_FLASH_ATTENTION': os.environ.get('ENABLE_FLASH_ATTENTION', '0'),
        'ENABLE_FP8_TRANSFORMER': os.environ.get('ENABLE_FP8_TRANSFORMER', '1'),  # Default ON
        'RFCNET_TORCHTRT': os.environ.get('RFCNET_TORCHTRT', '0'),
    }

    print("Critical Performance Variables:")
    for var, value in env_vars_status.items():
        is_enabled = value in ('1', 'true', 'yes', 'on')
        status_icon = "✓" if is_enabled else "✗"
        print(f"  {status_icon} {var:<25} = {value}")

    # Check if NeuFlow model file exists
    neuflow_path = os.path.join(os.getcwd(), 'faster-propainter-main', 'models', 'neuflow_things.onnx')
    neuflow_exists = os.path.exists(neuflow_path)
    print()
    print("Model Files:")
    print(f"  {'✓' if neuflow_exists else '✗'} NeuFlow v2 ONNX: {neuflow_path}")

    # Performance warnings
    print()
    print("Performance Analysis:")
    if env_vars_status['USE_NEUFLOW'] == '1':
        if neuflow_exists:
            print("  ✓ NeuFlow v2 enabled - OPTIMAL PERFORMANCE (10-70x faster optical flow)")
        else:
            print("  ✗ WARNING: USE_NEUFLOW=1 but model file missing!")
            print("    → Download from: https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases")
    else:
        print("  ⚠ Using PyTorch RAFT - SLOW! (4-5s per segment)")
        print("    → Set USE_NEUFLOW=1 for 10-70x speedup")

    if env_vars_status['ENABLE_FLASH_ATTENTION'] != '1':
        print("  ⚠ Flash Attention disabled - missing 3-5x transformer speedup")

    if env_vars_status['ENABLE_FP8_TRANSFORMER'] == '1':
        print("  ✓ FP8 Transformer quantization enabled - RTX 4090 Ada (5-10x speedup!)")
    else:
        print("  ⚠ FP8 Transformer disabled - missing 1.3-1.5x speedup on Linear layers")

    print("=" * 70)
    print()

    # Start background threads
    start_redis_download_poller()
    start_background_encoder()

    # Pre-load YOLO detector (saves 6-7s on first task)
    try:
        get_detector()
        print("[OK] YOLO detector pre-loaded")
    except Exception as e:
        print(f"[WARNING]  Failed to pre-load YOLO: {e}")

    # Pre-load ProPainter pipeline (saves 1-2s per segment)
    try:
        get_propainter_pipeline()
        print("[OK] ProPainter pipeline pre-loaded")
    except Exception as e:
        print(f"[WARNING]  Failed to pre-load ProPainter: {e}")

    print("[RUNNING] Worker fully initialized and ready!")


# Global model instances (lazy loaded)
detector = None
propainter_ready = False

# Use multiprocessing for true parallel GPU processing
# Each process gets its own CUDA context and model instances
from concurrent.futures import ProcessPoolExecutor

# File cleanup - delete files older than configured age
def cleanup_old_files():
    """Delete local files older than configured age (B2 cleanup handled by lifecycle rules)"""
    import time
    current_time = time.time()

    # Local file max age (configurable via env var)
    local_max_age = int(os.getenv('LOCAL_CLEANUP_MAX_AGE_SECONDS', '1800'))  # 30 minutes
    # Note: B2 file cleanup handled by B2 lifecycle rules (zero Class C transaction costs!)

    # Clean LOCAL files
    for directory in [UPLOAD_DIR, RESULT_DIR, TEMP_DIR]:
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > local_max_age:
                        os.remove(file_path)
                        print(f"[CLEANUP] Cleaned up old local file: {filename} (age: {file_age/60:.1f} min)")
        except Exception as e:
            print(f"[WARNING] Cleanup error in {directory}: {e}")

    # B2 cleanup handled by lifecycle rules - no code needed!
    # Files in uploads/, results/, masks/ are auto-deleted after configured period
    # Configure via B2 web UI: Bucket Settings → Lifecycle Rules
    # This eliminates Class C transaction costs from list/delete operations

# Schedule cleanup to run every configurable interval
import threading
def schedule_cleanup():
    cleanup_old_files()
    interval = int(os.getenv('CLEANUP_INTERVAL_SECONDS', '3600'))  # 1 hour (3600 seconds)
    threading.Timer(interval, schedule_cleanup).start()

# Start cleanup scheduler
threading.Thread(target=schedule_cleanup, daemon=True).start()
cleanup_interval = int(os.getenv('CLEANUP_INTERVAL_SECONDS', '3600'))
print(f"[CLEANUP] File cleanup scheduler started (runs every {cleanup_interval}s / {cleanup_interval/60:.1f} min)")

# ============================================================================
# Model Loading (Shared across workers)
# ============================================================================

def _check_propainter_assets() -> bool:
    """
    Verify required ProPainter assets are present.
    Returns True when everything looks good.
    """
    global propainter_ready

    if propainter_ready:
        return True

    # Support both Windows (weights/) and Docker (faster-propainter-main/weights/) paths
    weights_dir = os.path.join(SCRIPT_DIR, 'weights')
    if not os.path.exists(weights_dir):
        weights_dir = os.path.join(SCRIPT_DIR, 'faster-propainter-main', 'weights')

    required_paths = [
        os.path.join(SCRIPT_DIR, 'faster-propainter-main', 'watermark.py'),
        os.path.join(weights_dir, 'ProPainter.pth'),
        os.path.join(weights_dir, 'raft-things.pth'),
        os.path.join(weights_dir, 'recurrent_flow_completion.pth'),
    ]

    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        print("[ERROR] ProPainter assets missing:")
        for path in missing:
            print(f"   - {path}")
        print("   Download weights or copy them into the paths above.")
        propainter_ready = False
        return False

    # Check TensorRT DCNv4 files if TensorRT mode is enabled
    if os.getenv('FORCE_TRT_RFCNET', '0') == '1':
        # Support both Windows (.dll) and Docker (.so) plugin paths
        plugin_path = os.path.join(SCRIPT_DIR, 'dcnv4_tensorrt_plugin', 'build', 'Release', 'dcnv4_plugin.dll')
        if not os.path.exists(plugin_path):
            plugin_path = os.path.join(SCRIPT_DIR, 'libdcnv4_plugin.so')  # Docker path

        trt_paths = [
            os.path.join(SCRIPT_DIR, 'engines', 'rfcnet', 'rfcnet_dcnv4_fp16.engine'),
            plugin_path,
        ]
        trt_missing = [path for path in trt_paths if not os.path.exists(path)]
        if trt_missing:
            print("[ERROR] FORCE_TRT_RFCNET=1 but TensorRT DCNv4 files missing:")
            for path in trt_missing:
                print(f"   - {path}")
            print("   Build the engine with: python build_rfcnet_trt_engine.py --fp16")
            print("   Or set FORCE_TRT_RFCNET=0 to use PyTorch fallback")
            propainter_ready = False
            return False

    propainter_ready = True
    return True


def get_detector():
    """
    Lazy load the YOLO detector (TensorRT if available).
    Note: This is used by Celery workers on cloud machines, not the local Flask server.
    """
    global detector

    if detector is None:
        print("=" * 60)
        print("Loading YOLO detector...")
        print("=" * 60)
        from yolo_detector import YOLOWatermarkDetector
        import numpy as np
        # Use env var to control TensorRT requirement (default: True for speed)
        require_trt = os.getenv('YOLO_REQUIRE_TENSORRT', '1') == '1'
        detector = YOLOWatermarkDetector(require_tensorrt=require_trt)

        # WARMUP: Initialize TensorRT context (eliminates cold start overhead!)
        print("[WARMUP] Running warmup inference to initialize TensorRT context...")
        dummy_batch = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(64)]
        _ = detector.detect_batch(dummy_batch, confidence_threshold=0.15, batch_size=64)
        print("[OK] YOLO warmed up! TensorRT context ready for max speed.")

        print("=" * 60)
        print("[OK] YOLO detector ready!")
        print("=" * 60)

    # Check ProPainter assets (only matters on cloud workers)
    _check_propainter_assets()
    return detector


# Global ProPainter pipeline cache
_propainter_pipeline = None

def get_propainter_pipeline():
    """
    Cached ProPainter pipeline loader.
    Loads the pipeline once and reuses it across all tasks in the same worker.
    """
    global _propainter_pipeline

    if _propainter_pipeline is None:
        print("=" * 60)
        print("Loading ProPainter pipeline...")
        print("=" * 60)
        faster_propainter_path = os.path.join(SCRIPT_DIR, 'faster-propainter-main')
        if faster_propainter_path not in sys.path:
            sys.path.insert(0, faster_propainter_path)

        from watermark import pipeline as faster_propainter_pipeline
        _propainter_pipeline = faster_propainter_pipeline
        print("=" * 60)
        print("[OK] ProPainter pipeline loaded!")
        print("=" * 60)

        # Warmup SageAttention INT8 kernels (compiles once per worker)
        if os.getenv("ENABLE_SAGE_ATTENTION", "1") == "1":
            try:
                from watermark import warmup_sageattention
                import torch
                if torch.cuda.is_available():
                    # Shape: (B, T, H, W, num_heads, head_dim)
                    # Typical ProPainter transformer: batch=1, temporal=16, spatial=64x64, heads=8, dim=64
                    warmup_sageattention(device='cuda', q_shape=(1, 16, 64, 64, 8, 64))
            except Exception as e:
                print(f"[WARNING] SageAttention warmup failed: {e}")

        # Note: TensorRT RFC Net context will be created on first use (lazy init)
        # Expected performance: 9-12ms/frame after first segment warm-up
        if os.getenv('FORCE_TRT_RFCNET', '0') == '1':
            print("[INFO] TensorRT RFC Net enabled (FORCE_TRT_RFCNET=1)")
            print("[INFO] Context will be created on first inference (expect ~500ms warm-up)")
            print("[INFO] Performance after warm-up: ~9-12ms/frame (consistent)")

    return _propainter_pipeline


# ============================================================================
# Celery Tasks (Background Processing)
# ============================================================================

# ============================================================================
# DISTRIBUTED VIDEO PROCESSING TASKS
# These tasks enable multiple workers across different machines to collaborate
# on processing one video together by distributing segments across all GPUs
# ============================================================================

@celery.task(bind=True, name='watermark.broadcast_download')
def broadcast_video_download(self, video_id, video_url, upload_filename):
    """
    Broadcast task: All workers download video simultaneously
    This runs on ALL workers in parallel to pre-cache the video
    """
    try:
        print(f"\n📡 Broadcast download for video {video_id}")

        # Create cache directory
        cache_dir = os.path.join(TEMP_DIR, 'video_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cached_video = os.path.join(cache_dir, f"{video_id}.mp4")

        # Skip if already cached
        if os.path.exists(cached_video):
            print(f"   [OK] Video already cached")
            return {'status': 'cached', 'path': cached_video}

        # Download video
        print(f"   [DOWNLOAD]  Downloading: {video_url}")
        import requests
        r = requests.get(video_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=120)
        r.raise_for_status()

        # Save to cache
        with open(cached_video, 'wb') as f:
            f.write(r.content)

        file_size = os.path.getsize(cached_video) / (1024 * 1024)
        print(f"   [OK] Video downloaded and cached ({file_size:.2f} MB)")

        return {'status': 'downloaded', 'path': cached_video, 'size_mb': file_size}

    except Exception as e:
        print(f"   [ERROR] Broadcast download failed: {e}")
        return {'status': 'failed', 'error': str(e)}


@celery.task(bind=True, name='prepare_video')
def prepare_video_task(self, video_path, api_base=None, temp_base=None, video_id=None, masks_dir=None, masks_url=None):
    """
    Phase 1: Prepare video for distributed processing
    - Download video if needed
    - Run YOLO detection on all frames (centralized) OR use pre-downloaded masks
    - Generate masks
    - Detect segments (handles moving watermarks)
    - Workers will use these masks or regenerate if not available

    Args:
        masks_dir: Optional pre-downloaded masks directory (from _continue_after_masks)
                   If provided, skip YOLO detection and use these masks

    Returns: dict with video_id, segments, metadata for distributed processing
    """
    try:
        import json
        import time as time_module

        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Preparing video'})

        # Track processing start time for queue ETA calculation
        if video_id:
            try:
                redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                redis_client.set(f"video:{video_id}:start_time", str(time_module.time()))
            except:
                pass

        detector = get_detector()
        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing")

        # Download video if on remote worker OR if not found locally
        # First extract the local filename (whether video_path is URL or local path)
        from pathlib import PureWindowsPath
        base_name = PureWindowsPath(video_path).name if '\\' in video_path else os.path.basename(video_path)
        local_video_path = os.path.join(UPLOAD_DIR, base_name)

        # Smart check: If file exists locally (same PC workers share uploads/), skip download!
        if os.path.exists(local_video_path):
            print(f"[OK] Video already exists locally: {local_video_path} (skip download)")
            video_path = local_video_path
        elif not os.path.exists(video_path):
            # File not found locally - download from remote
            tunnel = api_base or os.getenv('TUNNEL_URL')
            if tunnel:
                try:
                    from urllib.parse import urljoin
                    import requests
                    download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                    print(f"🌐 Downloading video: {download_url}")
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                    r.raise_for_status()
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    with open(local_video_path, 'wb') as f:
                        f.write(r.content)
                    print(f"[OK] Video downloaded: {local_video_path}")
                    video_path = local_video_path
                except Exception as e:
                    raise Exception(f"Failed to download video: {e}")
            else:
                raise Exception(f"Video not found: {video_path}")

        print(f"📹 Preparing video: {video_path} ({os.path.getsize(video_path) / (1024 * 1024):.2f} MB)")

        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Analyzing video'})

        # NVDEC hardware decoder - NO FALLBACKS! (1.16x faster than CPU)
        use_nvdec = os.getenv("ENABLE_NVDEC", "0") == "1"
        nvdec_loader = None
        cap = None

        if use_nvdec:
            # NVDEC REQUIRED - NO CPU FALLBACK!
            from nvdec_video_loader import NVDECVideoLoader
            nvdec_loader = NVDECVideoLoader(video_path, device_id=0)
            props = nvdec_loader.get_properties()
            fps = int(props['fps'])
            width = props['width']
            height = props['height']
            total_frames = props['total_frames']
            print(f"[OK] NVDEC hardware decoder: {width}x{height} @ {fps} fps ({total_frames} frames)")
        else:
            # CPU decoder (if NVDEC disabled)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {video_path}")

            fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            print(f"[OK] CPU decoder: {width}x{height} @ {fps} fps ({total_frames} frames)")

        base_name = Path(video_path).stem
        # Use provided video_id (from upload task_id) for cache consistency, fallback to Celery task ID
        video_id = video_id or (self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8])

        # 🔒 DEDUPLICATION: Check if this video is already being prepared by another worker
        try:
            redis_client = celery.backend.client
            lock_key = f'prepare_lock:{base_name}'

            # Check if another worker is already processing this video
            if redis_client.exists(lock_key):
                print(f"[SKIP]  Video '{base_name}' already being prepared by another worker - skipping duplicate task")
                # Clean up video decoder (NO FALLBACKS!)
                if nvdec_loader is not None:
                    nvdec_loader.close()
                if cap is not None:
                    cap.release()
                return {
                    'status': 'skipped',
                    'message': f'Video already being processed',
                    'video_id': base_name
                }

            # Acquire lock for 5 minutes (YOLO detection + frame extraction time)
            redis_client.setex(lock_key, 300, self.request.id if hasattr(self.request, 'id') else 'unknown')
            print(f"🔒 Acquired processing lock for video '{base_name}'")
        except Exception as e:
            print(f"[WARNING]  Deduplication check failed: {e} - proceeding anyway")

        # [RUNNING] REDIS SIGNAL: Tell ALL workers to download this video in parallel!
        # This allows idle workers to start downloading immediately instead of waiting
        try:
            redis_client = celery.backend.client
            # Construct download URL for workers to use
            tunnel = api_base or temp_base or os.getenv('TUNNEL_URL') or os.getenv('API_BASE_URL')
            if tunnel:
                from urllib.parse import urljoin
                video_filename = os.path.basename(video_path)
                download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{video_filename}')
                # Set Redis key with 5 minute expiry (workers will poll for this)
                redis_client.setex(f'video_download:{video_id}', 300, download_url)
                print(f"📡 Redis signal sent: video_download:{video_id} = {download_url}")
        except Exception as e:
            print(f"[WARNING]  Failed to send Redis download signal: {e}")

        # Create shared directories for distributed access
        # If masks_dir provided (from _continue_after_masks), use it directly!
        print(f"[DEBUG] masks_dir received: {masks_dir}")
        print(f"[DEBUG] masks_url received: {masks_url}")
        print(f"[DEBUG] os.path.isdir(masks_dir): {os.path.isdir(masks_dir) if masks_dir else 'N/A'}")
        if masks_dir:
            print(f"[DEBUG] os.path.exists(masks_dir): {os.path.exists(masks_dir)}")
            if os.path.exists(masks_dir):
                try:
                    contents = os.listdir(masks_dir)
                    print(f"[DEBUG] Directory contents ({len(contents)} items): {contents[:5] if len(contents) > 5 else contents}...")
                except Exception as e:
                    print(f"[DEBUG] Failed to list directory: {e}")

        if masks_dir and os.path.isdir(masks_dir):
            shared_mask_dir = masks_dir
            print(f"[OK] Using pre-downloaded masks from: {masks_dir}")
            mask_count = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
            print(f"   Found {mask_count} masks - skipping YOLO detection!")
        else:
            print(f"[WARNING] masks_dir check FAILED - will create new directory and run YOLO")
            shared_mask_dir = os.path.join(PROPAINTER_MASK_ROOT, f"{base_name}_{video_id}")
            os.makedirs(shared_mask_dir, exist_ok=True)
        shared_frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_originals")
        os.makedirs(shared_frames_dir, exist_ok=True)

        # [RUNNING] CACHE CHECK: Skip YOLO if segments already cached from previous request
        segments = None
        frames_processed = total_frames
        cache_key = f'segments_cache:{base_name}'

        try:
            # Check if segments were already detected for this video
            cached_data = redis_client.get(cache_key)
            if cached_data:
                import json
                import time
                cached = json.loads(cached_data)

                # Verify frames/masks still exist (not cleaned up)
                frame_count = len([f for f in os.listdir(shared_frames_dir) if f.endswith('.png')]) if os.path.exists(shared_frames_dir) else 0

                if frame_count > 0:
                    segments = cached['segments']
                    frames_processed = cached.get('total_frames', total_frames)
                    frames_with_watermark = cached.get('frames_with_watermark', 0)  # Load from cache
                    cache_age = time.time() - cached.get('cached_at', 0)
                    print(f"[OK] Using cached YOLO results for '{base_name}' (age: {cache_age:.1f}s, {frame_count} frames)")
                    print(f"   [SKIP]  Skipping YOLO detection - reusing {len(segments)} cached segments")
                    # Convert cached segments back to tuples (JSON converts tuples to lists)
                    segments = [tuple(seg) for seg in segments]
                else:
                    print(f"[WARNING]  Cache found but frames missing - will re-run YOLO")
                    segments = None
        except Exception as e:
            print(f"[WARNING]  Cache check failed: {e} - will run YOLO")
            segments = None

        # If masks_dir provided (from _continue_after_masks), detect segments from masks instead of YOLO
        if segments is None and masks_dir and os.path.isdir(masks_dir):
            print(f"[OK] Detecting segments from pre-downloaded masks...")
            self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Detecting segments from masks'})
            try:
                from segment_detector import detect_segments_from_masks
                segments = detect_segments_from_masks(
                    masks_dir,
                    position_tolerance=int(os.getenv('SEGMENT_POS_TOLERANCE', '50')),
                    min_segment_length=int(os.getenv('SEGMENT_MIN_LEN_FULL', '3'))
                )
                print(f"[OK] Detected {len(segments)} segments from masks")
                # Count frames with masks for watermark coverage stat
                frames_with_watermark = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
            except Exception as e:
                print(f"[WARNING] Failed to detect segments from masks: {e}")
                segments = None

        # Run YOLO only if not cached AND no pre-downloaded masks
        if segments is None:
            print(f"[REGEN] Running YOLO detection on {total_frames} frames...")
            self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Detecting watermarks'})

            # [INIT] EXTREME SPEED: Load all frames to memory for batch processing
            print(f"📥 Loading {total_frames} frames to memory (batch processing)...")

            import time
            decode_start = time.time()

            if use_nvdec:
                # NVDEC hardware decoder - NO FALLBACK! (1.16x faster)
                all_frames = nvdec_loader.load_all_frames(to_numpy=True, color_format='BGR')
                frames_processed = len(all_frames)
                decode_time = time.time() - decode_start
                print(f"[OK] NVDEC decoded {frames_processed} frames: {decode_time:.3f}s ({decode_time/frames_processed*1000:.2f}ms/frame)")
                nvdec_loader.close()
            else:
                # CPU decoder (only if NVDEC disabled)
                all_frames = []
                all_frames_reserve = [None] * int(total_frames)  # Reserve memory
                frames_loaded = 0

                # Fast frame loading (no prints in loop!)
                while frames_loaded < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_frames.append(frame)
                    frames_loaded += 1

                cap.release()
                frames_processed = len(all_frames)
                decode_time = time.time() - decode_start
                print(f"[OK] CPU decoded {frames_processed} frames: {decode_time:.3f}s ({decode_time/frames_processed*1000:.2f}ms/frame)")

            # [RUNNING] BATCH DETECTION (EXTREME SPEED - 1-2ms per frame!)
            print(f"[RUNNING] Running BATCH detection on {frames_processed} frames (EXTREME SPEED!)...")
            self.update_state(state='PROCESSING', meta={'progress': 15, 'status': f'Batch detection (1-2ms/frame)'})

            import time
            batch_start = time.time()
            # Batch detect ALL frames at once! (batch_size=64 - FASTEST per benchmark: 748 fps!)
            all_detections = detector.detect_batch(all_frames, confidence_threshold=0.15, padding=0, batch_size=64)
            batch_duration = time.time() - batch_start
            ms_per_frame = (batch_duration / max(frames_processed, 1)) * 1000
            print(f"[OK] Batch detection complete: {batch_duration:.3f}s ({ms_per_frame:.2f}ms per frame)")

            # [INIT] EXTREME SPEED: Process detections and store in Redis (in-memory!)
            print(f"⚡ Creating masks and storing in Redis (in-memory)...")
            zero_mask = np.zeros((height, width), dtype=np.uint8)
            bboxes_per_frame = []
            frames_with_watermark = 0
            last_valid_bbox = None
            all_masks = []

            # Track bboxes for segmentation
            for i, detections in enumerate(all_detections):
                if detections:
                    frames_with_watermark += 1
                    last_valid_bbox = detections[0]['bbox']
                    bboxes_per_frame.append(last_valid_bbox)
                elif last_valid_bbox:
                    bboxes_per_frame.append(last_valid_bbox)
                else:
                    bboxes_per_frame.append(None)

            # Create all masks with GPU batch processing (10-17x faster than CPU loop!)
            # OR use SAM2 for temporal consistency (BEST QUALITY!)
            use_sam2 = os.getenv('USE_SAM2_TRACKING', '0') == '1'

            if use_sam2 and frames_with_watermark > 0:
                print(f"⚡ Using SAM2-Tiny for temporal mask tracking (44ms/frame)...")
                mask_start = time.time()

                # Import SAM2 tracker
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'faster-propainter-main'))
                from sam2_tracker import SAM2Tracker

                # Get first bbox from YOLO detection
                first_bbox = None
                for bbox in bboxes_per_frame:
                    if bbox is not None:
                        first_bbox = bbox
                        break

                if first_bbox is None:
                    print("[WARNING] No bbox found - falling back to GPU masks")
                    if hasattr(detector, 'use_gpu_masks') and detector.use_gpu_masks:
                        all_masks = detector.create_masks_batch_gpu(all_frames, all_detections)
                    else:
                        all_masks = [
                            detector.create_mask(frame, dets) if dets else zero_mask
                            for frame, dets in zip(all_frames, all_detections)
                        ]
                else:
                    # Initialize SAM2-Tiny tracker
                    tracker = SAM2Tracker(device="cuda")

                    # Convert YOLO bbox [x1, y1, x2, y2] format for SAM2
                    bbox_xyxy = [int(first_bbox[0]), int(first_bbox[1]),
                                 int(first_bbox[2]), int(first_bbox[3])]

                    # Track watermark across all frames
                    masks_bool = tracker.track_from_box(
                        video_frames=all_frames,
                        bbox=bbox_xyxy,
                        frame_idx=0,
                        video_path=video_path
                    )

                    # Convert boolean masks to uint8 [0, 255]
                    all_masks = [(mask * 255).astype(np.uint8) for mask in masks_bool]

                    print(f"[OK] SAM2 tracked {len(all_masks)} frames with temporal consistency")

                mask_duration = time.time() - mask_start
                print(f"   SAM2 tracking: {mask_duration:.3f}s ({mask_duration/frames_processed*1000:.2f}ms/frame)")
            else:
                print(f"⚡ Creating {frames_processed} masks (GPU batch)...")
                mask_start = time.time()

                # Compatibility: Check if GPU masks are available (older commits may not have this)
                if hasattr(detector, 'use_gpu_masks') and detector.use_gpu_masks:
                    # GPU batch processing - ALL masks at once!
                    all_masks = detector.create_masks_batch_gpu(all_frames, all_detections)
                else:
                    # Fallback to CPU sequential (if Kornia not available)
                    all_masks = [
                        detector.create_mask(frame, dets) if dets else zero_mask
                        for frame, dets in zip(all_frames, all_detections)
                    ]

                mask_duration = time.time() - mask_start
                print(f"   Mask creation: {mask_duration:.3f}s ({frames_processed/mask_duration:.1f} masks/sec)")

            # [INIT] EXTREME SPEED: Store in global memory (INSTANT!)
            print(f"⚡ Storing {frames_processed} frames/masks in memory (INSTANT!)...")
            mem_start = time.time()

            # Store in global FRAME_CACHE (just storing Python references - INSTANT!)
            cache_key = f"video_data:{base_name}"
            with FRAME_CACHE_LOCK:
                FRAME_CACHE[cache_key] = {
                    'frames': all_frames,         # List of numpy arrays (already in RAM!)
                    'masks': all_masks,            # List of numpy arrays
                    'bboxes': bboxes_per_frame,   # For segmentation
                    'timestamp': time.time(),      # For cleanup
                    'video_id': video_id,
                    'base_name': base_name
                }

            mem_duration = time.time() - mem_start
            mem_ms_per_frame = (mem_duration / max(frames_processed, 1)) * 1000
            print(f"[OK] Memory storage complete: {mem_duration:.4f}s ({mem_ms_per_frame:.3f}ms per frame) - INSTANT!")

            # OPTIONAL: Also write to disk for backup/remote workers
            # Disabled by default for EXTREME SPEED (memory-only)
            if os.getenv('WRITE_FRAMES_TO_DISK', '0') == '1':
                print(f"💾 Also writing to disk (backup for remote workers)...")
                for i in range(frames_processed):
                    mask_path = os.path.join(shared_mask_dir, f"{i:05d}.png")
                    cv2.imwrite(mask_path, all_masks[i])
                    frame_path = os.path.join(shared_frames_dir, f"{i:05d}.png")
                    cv2.imwrite(frame_path, all_frames[i])
            else:
                print(f"⚡ Skipping disk writes (pure memory for EXTREME SPEED!)")

            if frames_processed == 0:
                raise RuntimeError("No frames processed - video may be corrupted")

            print(f"[OK] Detection complete: {frames_processed} frames, {frames_with_watermark} with watermarks")
            print(f"[OK] Frames saved to shared storage: {shared_frames_dir}")

            # Detect segments (by watermark position changes)
            from segment_detector import detect_segments, merge_adjacent_segments
            segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
            if segments:
                segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
                print(f"[INFO] Detected {len(segments)} segments for distributed processing")
                for i, (start, end, bbox) in enumerate(segments):
                    print(f"   Segment {i+1}: frames {start}-{end} ({end-start+1} frames), bbox={bbox}")
            else:
                # No segments detected - treat entire video as one segment
                segments = [(0, frames_processed-1, last_valid_bbox if last_valid_bbox else [0,0,width,height])]
                print("[INFO] No segments detected - processing entire video as one segment")

            # 💾 Cache segment results for future duplicate requests (1 hour TTL)
            try:
                import json
                import time
                cache_data = {
                    'segments': segments,  # Will be converted to list by JSON
                    'total_frames': frames_processed,
                    'frames_with_watermark': frames_with_watermark,  # Include for accurate stats
                    'cached_at': time.time()
                }
                redis_client.setex(cache_key, 3600, json.dumps(cache_data))
                print(f"💾 Cached YOLO results for '{base_name}' (1 hour TTL)")
            except Exception as e:
                print(f"[WARNING]  Failed to cache results: {e}")
        else:
            # Using cached segments - release video decoder (NO FALLBACKS!)
            if nvdec_loader is not None:
                nvdec_loader.close()
            if cap is not None:
                cap.release()
            # frames_with_watermark should be loaded from cache, but add fallback
            if 'frames_with_watermark' not in locals():
                frames_with_watermark = 0  # Fallback if not in cache

        # Optional: force time-based splitting to ensure multi-GPU distribution
        try:
            import math
            min_segments = int(os.getenv('MIN_SEGMENTS', '0'))
            min_chunk_frames = int(os.getenv('MIN_CHUNK_FRAMES', '60'))
        except Exception:
            min_segments = 0
            min_chunk_frames = 60

        # Force-split ONLY when YOLO fails (0-1 segments detected)
        # If YOLO found multiple segments, preserve them - they represent distinct watermark regions
        if min_segments and len(segments) <= 1 and frames_processed >= min_chunk_frames:
            # YOLO failed - split video into time-based chunks for parallel processing
            base_seg = segments[0] if segments else (0, frames_processed-1, last_valid_bbox if last_valid_bbox else [0,0,width,height])
            s0, e0, bb = base_seg
            duration = e0 - s0 + 1
            num_chunks = min_segments
            chunk = max(min_chunk_frames, math.ceil(duration / num_chunks))
            new_segments = []
            cur = s0
            while cur <= e0:
                end = min(e0, cur + chunk - 1)
                new_segments.append((cur, end, bb if bb else [0,0,width,height]))
                cur = end + 1
                if len(new_segments) >= num_chunks and end < e0:
                    # If more frames remain after hitting desired count, extend last chunk
                    new_segments[-1] = (new_segments[-1][0], e0, new_segments[-1][2])
                    break
            segments = new_segments
            print(f"🪓 Force-split enabled (YOLO fallback): created {len(segments)} time chunks (chunk≈{chunk} frames)")

        # Provide a base URL so OTHER workers can fetch frames/masks from this host
        temp_base_url = temp_base or os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL')
        if temp_base_url:
            print(f"🌐 Shared temp base set for workers: {temp_base_url.rstrip('/')}/temp/")
        else:
            print("[WARNING]  No TEMP_BASE_URL/TUNNEL_URL set; only preparing worker can read local frames.")

        # Prepare segment data for distribution
        segment_tasks_data = []
        api_base_url = api_base or os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL')
        for seg_idx, (start_frame, end_frame, bbox) in enumerate(segments):
            segment_tasks_data.append({
                'seg_idx': seg_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'bbox': bbox,
                'video_id': video_id,
                'base_name': base_name,
                'width': width,
                'height': height,
                'fps': fps,
                'video_path': video_path,  # For background encoder audio merge
                'shared_mask_dir': shared_mask_dir,
                'shared_frames_dir': shared_frames_dir,
                'masks_url': masks_url,  # B2 CDN URL for remote download fallback
                'video_url': f"{api_base_url.rstrip('/')}/uploads/{os.path.basename(video_path)}",
                'temp_base_url': temp_base_url,
                'api_base': api_base_url,
                'upload_filename': os.path.basename(video_path),
                'total_frames': frames_processed,  # For clamping padded_end on last segment
            })

        result = {
            'video_id': video_id,
            'base_name': base_name,
            'video_path': video_path,
            'segments': segment_tasks_data,
            'total_segments': len(segments),
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': frames_processed,
            'frames_with_watermark': frames_with_watermark,
            'shared_mask_dir': shared_mask_dir,
            'shared_frames_dir': shared_frames_dir,
            'api_base': api_base or os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL'),
            'temp_base_url': temp_base_url,
        }

        print(f"[OK] Video prepared for distributed processing: {len(segments)} segments ready")

        # 🔒 REDIS LOCK: Only ONE worker should dispatch segments (others exit here)
        # When multiple workers do YOLO in parallel, they all find the same segments
        # Use Redis lock to ensure only the first one dispatches
        redis_client = celery.backend.client
        dispatch_lock_key = f"dispatch_lock:{video_id}"

        # Try to acquire lock (set if not exists, 60 second expiry)
        lock_acquired = redis_client.set(dispatch_lock_key, self.request.id, nx=True, ex=60)

        if not lock_acquired:
            # Another worker already dispatching segments - this worker's job is done!
            print(f"🔓 Another worker already dispatching segments - exiting early (YOLO parallelization worked!)")
            return {
                'chord_id': f'distributed_{video_id}',
                'status': 'processing',
                'message': f'Parallel YOLO worker (segments being dispatched by another worker)'
            }

        # We got the lock! This worker will dispatch segments
        print(f"🔒 Lock acquired - this worker will dispatch segments")

        # ⚡ OPTIMIZATION: Use Celery chord to dispatch segments in parallel
        # Chord automatically waits for ALL segments to complete, then triggers finalize
        print(f"[INIT] Creating chord: {len(segments)} segment tasks → finalize callback")
        self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Dispatching {len(segments)} parallel tasks'})

        # Add total_segments to each segment data
        for seg in segment_tasks_data:
            seg['total_segments'] = len(segments)

        # Create task signatures (delayed execution) for each segment
        segment_sigs = [process_segment_task.s(seg_data) for seg_data in segment_tasks_data]

        # Create chord: segments run in parallel, finalize runs when ALL complete
        # The chord returns a finalize task ID that we can track
        workflow = chord(segment_sigs)(finalize_video_task.s(prepare_result=result))

        # Store tracking for status endpoint (distributed_ pattern compatibility)
        tracking_key = f"segments:{video_id}"
        celery.backend.set(f"{tracking_key}:total", len(segments))
        celery.backend.set(f"{tracking_key}:prepare_result", json.dumps(result))

        print(f"[OK] Chord dispatched! Segments will run in parallel across workers")
        print(f"   Finalize callback ID: {workflow.id}")
        print(f"   Finalize will auto-trigger when all {len(segments)} segments complete")

        # 🔓 Release processing lock - we're done with preparation
        try:
            redis_client = celery.backend.client
            lock_key = f'prepare_lock:{base_name}'
            redis_client.delete(lock_key)
            print(f"🔓 Released processing lock for video '{base_name}'")
        except Exception as e:
            print(f"[WARNING]  Failed to release lock: {e}")

        # Return distributed task ID for frontend tracking (status endpoint recognizes this pattern)
        return {
            'chord_id': f'distributed_{video_id}',
            'status': 'processing',
            'message': f'Chord workflow: {len(segments)} segments → finalize callback'
        }

    except Exception as e:
        # 🔓 Release processing lock on error
        try:
            redis_client = celery.backend.client
            lock_key = f'prepare_lock:{base_name}'
            redis_client.delete(lock_key)
            print(f"🔓 Released processing lock for video '{base_name}' (error cleanup)")
        except:
            pass  # Ignore cleanup errors

        print(f"[ERROR] Error preparing video: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.process_segment')
def process_segment_task(self, segment_data):
    """
    Phase 2: Process one segment (distributed across all available workers/GPUs)
    - Download required frames and masks
    - Run ProPainter on this segment
    - Upload cleaned frames back

    This task can run on ANY available worker with a GPU
    """
    try:
        seg_idx = segment_data['seg_idx']
        total_segments = segment_data.get('total_segments', '?')
        start_frame = segment_data['start_frame']
        end_frame = segment_data['end_frame']
        bbox = segment_data['bbox']
        video_id = segment_data['video_id']
        base_name = segment_data['base_name']
        width = segment_data['width']
        height = segment_data['height']
        fps = segment_data['fps']
        video_path = segment_data['video_path']  # For background encoder audio merge

        print(f"\n[SEGMENT] Worker processing segment {seg_idx+1}/{total_segments}: frames {start_frame}-{end_frame}")
        self.update_state(state='STARTED', meta={'progress': 0, 'status': f'Processing segment {seg_idx+1}'})

        # Import required modules
        import subprocess
        from crop_utils import calculate_crop_region

        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"   [CROP] Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

        seg_duration = end_frame - start_frame + 1

        # Create local temp directories for this segment
        seg_prefix = f"{base_name}_{video_id}_seg{seg_idx}"
        seg_frames_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_frames")
        seg_cropped_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cropped")
        seg_mask_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_masks")
        seg_output_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_output")

        # 🔥 SHARED FRAME BUFFER FIX: All segments merge onto same directory
        # This allows multiple segments to cooperatively edit the same frames
        shared_cleaned_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_all_frames_cleaned")
        seg_cleaned_dir = shared_cleaned_dir  # Backwards compatibility alias

        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
            os.makedirs(path, exist_ok=True)

        # Download frames from shared storage (smart: try local first, then individual frames, then video as fallback)
        origin_base = segment_data.get('temp_base_url') or os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL')
        shared_mask_dir = segment_data.get('shared_mask_dir')
        shared_frames_dir = segment_data.get('shared_frames_dir')

        # Debug logging for mask path investigation
        print(f"   [DEBUG] Segment {seg_idx} - shared_mask_dir from segment_data: {shared_mask_dir}")
        print(f"   [DEBUG] Segment {seg_idx} - shared_mask_dir exists: {os.path.exists(shared_mask_dir) if shared_mask_dir else 'N/A'}")
        if shared_mask_dir and os.path.exists(shared_mask_dir):
            try:
                mask_files = os.listdir(shared_mask_dir)
                print(f"   [DEBUG] Segment {seg_idx} - masks in directory: {len(mask_files)} files")
            except Exception as e:
                print(f"   [DEBUG] Segment {seg_idx} - failed to list masks: {e}")

        # Multi-PC optimization: Skip frame sharing, download video directly in parallel
        # Smart detection: Check if shared dirs are locally accessible (same PC = fast file copy)
        is_local_worker = shared_frames_dir and os.path.exists(shared_frames_dir)
        is_multi_pc = not is_local_worker

        import requests
        frames_copied = 0

        # 🔥 TEMPORAL CONTEXT FIX: Define padding BEFORE if/else so both paths have access
        # neighbor_length=10 in ProPainter needs ±10 frames external padding for proper temporal context
        neighbor_padding = 10
        padded_start = max(0, start_frame - neighbor_padding)
        # 🔥 CLAMP padded_end to video length (fixes last segment artifacts!)
        total_frames_video = segment_data.get('total_frames', end_frame + neighbor_padding + 1)
        padded_end = min(end_frame + neighbor_padding, total_frames_video - 1)

        # Initialize memory arrays (used by both paths, empty for multi-PC disk-based mode)
        segment_frames_memory = []
        segment_masks_memory = []

        # Cache key for FRAME_CACHE lookup (defined here so both paths can access)
        cache_key = f"video_data:{base_name}"

        if is_multi_pc:
            # Multi-PC mode: Each worker downloads video directly (faster, no tunnel congestion)
            print(f"   📦 Multi-PC mode: downloading video directly for frames {start_frame}-{end_frame}...")
            self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Downloading video'})

            api_base = os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL') or origin_base
            upload_filename = segment_data.get('upload_filename')
            video_url = segment_data.get('video_url')
            if (api_base or video_url) and upload_filename:
                try:
                    # Check cache first to avoid re-downloading for multiple segments
                    cache_dir = os.path.join(TEMP_DIR, 'video_cache')
                    os.makedirs(cache_dir, exist_ok=True)
                    cached_video = os.path.join(cache_dir, f"{video_id}.mp4")

                    if os.path.exists(cached_video):
                        print(f"   [OK] Using cached video (skip download)")
                        local_video = cached_video
                    else:
                        video_url = video_url or f"{api_base.rstrip('/')}/uploads/{upload_filename}"
                        print(f"   [DOWNLOAD]  Downloading: {video_url}")
                        r = requests.get(video_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                        r.raise_for_status()
                        # Save to cache for future segments
                        with open(cached_video, 'wb') as f:
                            f.write(r.content)
                        print(f"   💾 Video cached for future segments")
                        local_video = cached_video

                    cap2 = cv2.VideoCapture(local_video)
                    if not cap2.isOpened():
                        raise RuntimeError("Video open failed")
                    current_frame = 0
                    while True:
                        ret, frame = cap2.read()
                        if not ret:
                            break
                        # 🔥 FIX: Use padded ranges for neighbor context (same as single-PC mode)
                        if current_frame > padded_end:
                            break
                        if current_frame >= padded_start:
                            dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
                            cv2.imwrite(dst, frame)
                            frames_copied += 1
                        current_frame += 1
                    cap2.release()
                    # Don't delete - keep cached for other segments!
                except Exception as e:
                    raise RuntimeError(f"Video download failed: {e}")
            else:
                raise RuntimeError(f"No video URL available for segment {seg_idx}")
        else:
            # Single-PC mode: Try frame sharing first
            print(f"   [DOWNLOAD]  Loading frames {start_frame}-{end_frame}...")
            self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Loading frames'})

            # [INIT] EXTREME SPEED: Try FRAME_CACHE first (pure memory, INSTANT!)
            # cache_key and segment_frames_memory already initialized before if/else block
            memory_hits = 0

            if cache_key in FRAME_CACHE:
                # INSTANT access to frames in memory!
                print(f"   ⚡ Loading from memory cache WITH NEIGHBOR PADDING (ZERO disk I/O!)...")
                with FRAME_CACHE_LOCK:
                    cached = FRAME_CACHE[cache_key]
                    cached_frames = cached['frames']
                    total_frames = len(cached_frames)  # Update total_frames from cache

                    # Recalculate padded_end with correct total_frames
                    padded_end = min(total_frames - 1, end_frame + neighbor_padding)

                    # Extract frames WITH PADDING for temporal context
                    print(f"   [CONTEXT] Extracting frames {padded_start}-{padded_end} (core: {start_frame}-{end_frame}, padding: ±{neighbor_padding})")
                    for frame_idx in range(padded_start, padded_end + 1):
                        if frame_idx < len(cached_frames):
                            frame = cached_frames[frame_idx]
                            segment_frames_memory.append(frame)
                            frames_copied += 1
                            memory_hits += 1

                if memory_hits > 0:
                    print(f"   [OK] Loaded {memory_hits} frames from memory (including ±{neighbor_padding} neighbor padding for temporal context!)")

            # Fallback: Try other sources if memory cache incomplete
            if frames_copied < (padded_end - padded_start + 1):  # Use PADDED count!
                for frame_idx in range(padded_start, padded_end + 1):  # Use PADDED range!
                    # Skip if already copied from memory
                    if frame_idx - padded_start < memory_hits:  # Use padded_start!
                        continue

                    frame_file = f"{frame_idx:05d}.png"  # 5-digit to match WSL format!
                    local_idx = frame_idx - padded_start  # Local index for segment

                    # Priority 2: Local filesystem (if on same machine as prepare task)
                    local_frame = os.path.join(shared_frames_dir, frame_file) if shared_frames_dir else None
                    if local_frame and os.path.exists(local_frame):
                        dst = os.path.join(seg_frames_dir, f"{local_idx:04d}.png")
                        shutil.copy2(local_frame, dst)
                        frames_copied += 1
                    elif origin_base:
                        # Download individual frame from origin host (serving /temp/)
                        try:
                            frame_url = f"{origin_base.rstrip('/')}/temp/{base_name}_{video_id}_originals/{frame_file}"
                            r = requests.get(frame_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=10)
                            if r.ok:
                                dst = os.path.join(seg_frames_dir, f"{local_idx:04d}.png")
                                with open(dst, 'wb') as f:
                                    f.write(r.content)
                                frames_copied += 1
                        except Exception as e:
                            print(f"[WARNING]  Failed to download frame {frame_idx}: {e}")

            if frames_copied == 0:
                # Fallback: fetch original video from API uploads and extract only needed frames
                print(f"   [WARNING]  No frames found in shared storage, falling back to video download...")
                api_base = os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL') or origin_base
                upload_filename = segment_data.get('upload_filename')
                video_url = segment_data.get('video_url')
                if (api_base or video_url) and upload_filename:
                    try:
                        import requests
                        video_url = video_url or f"{api_base.rstrip('/')}/uploads/{upload_filename}"
                        print(f"   [DOWNLOAD]  Fallback: downloading original video for local extraction: {video_url}")
                        r = requests.get(video_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                        r.raise_for_status()
                        local_video = os.path.join(TEMP_DIR, f"seg_{seg_idx}_{upload_filename}")
                        with open(local_video, 'wb') as f:
                            f.write(r.content)
                        cap2 = cv2.VideoCapture(local_video)
                        if not cap2.isOpened():
                            raise RuntimeError("Fallback video open failed")
                        current_frame = 0
                        while True:
                            ret, frame = cap2.read()
                            if not ret:
                                break
                            if current_frame > padded_end:  # Use PADDED end!
                                break
                            if current_frame >= padded_start:  # Use PADDED start!
                                local_idx = current_frame - padded_start  # Proper local index
                                dst = os.path.join(seg_frames_dir, f"{local_idx:04d}.png")
                                cv2.imwrite(dst, frame)
                                frames_copied += 1
                            current_frame += 1
                        cap2.release()
                        os.remove(local_video)
                    except Exception as e:
                        raise RuntimeError(f"Fallback frame extraction failed: {e}")
                else:
                    raise RuntimeError(f"No frames available for segment {seg_idx}")

        if frames_copied == 0:
            raise RuntimeError(f"No frames extracted for segment {seg_idx}")

        print(f"   [OK] Loaded {frames_copied} frames ({start_frame}-{end_frame})")

        # Try to get masks from shared location (same PC) or download/regenerate them (different PC)
        masks_downloaded = False

        # [INIT] EXTREME SPEED: Try FRAME_CACHE first (pure memory, INSTANT!)
        print(f"   [MASKS] Loading masks...")
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Loading masks'})

        # 🔥 TEMPORAL CONTEXT FIX: Load masks WITH PADDING (same range as frames)
        masks_needed = list(range(padded_start, padded_end + 1))
        masks_succeeded = 0
        memory_mask_hits = 0
        segment_masks_memory = []  # Store masks in memory (skip disk!)

        # Priority 1: Memory cache (INSTANT!)
        if cache_key in FRAME_CACHE:
            print(f"   ⚡ Loading masks from memory WITH NEIGHBOR PADDING (ZERO disk I/O!)...")
            with FRAME_CACHE_LOCK:
                cached = FRAME_CACHE[cache_key]
                cached_masks = cached['masks']

                print(f"   [CONTEXT] Extracting masks {padded_start}-{padded_end} (core: {start_frame}-{end_frame}, padding: ±{neighbor_padding})")
                for abs_frame_idx in masks_needed:
                    if abs_frame_idx < len(cached_masks):
                        mask = cached_masks[abs_frame_idx]
                        segment_masks_memory.append(mask)
                        masks_succeeded += 1
                        memory_mask_hits += 1

        # Priority 2: Local filesystem (if memory incomplete)
        if masks_succeeded < len(masks_needed):
            try:
                for abs_frame_idx in masks_needed:
                    # Skip if already got from memory
                    if abs_frame_idx - padded_start < memory_mask_hits:
                        continue

                    if shared_mask_dir:
                        mask_filename = f"{abs_frame_idx:05d}.png"
                        shared_mask_path = os.path.join(shared_mask_dir, mask_filename)

                        if os.path.exists(shared_mask_path):
                            local_idx = abs_frame_idx - padded_start  # Use padded_start for correct local indexing!
                            local_mask_path = os.path.join(seg_mask_dir, f"{local_idx:04d}.png")
                            shutil.copy2(shared_mask_path, local_mask_path)
                            masks_succeeded += 1
                        else:
                            # Missing mask - break and regenerate all
                            break
            except Exception as copy_err:
                print(f"   [WARNING]  Mask loading failed: {copy_err} - will regenerate")

        if masks_succeeded == len(masks_needed):
            if memory_mask_hits > 0:
                print(f"   [OK] Loaded {masks_succeeded} masks from memory (ZERO disk I/O!)")
            else:
                print(f"   [OK] Copied {masks_succeeded} masks from local storage (fast!)")
            masks_downloaded = True
        else:
            print(f"   [WARNING]  Only {masks_succeeded}/{len(masks_needed)} masks found - will regenerate")

        # Try local filesystem first (same PC - FAST, direct file copy)
        if not masks_downloaded and shared_mask_dir and os.path.exists(shared_mask_dir):
            print(f"   [MASKS] Fallback: Copying masks from local shared directory...")
            self.update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Copying masks'})

            try:
                masks_needed = list(range(padded_start, padded_end + 1))  # Use PADDED range for temporal context!
                masks_succeeded = 0

                for abs_frame_idx in masks_needed:
                    mask_filename = f"{abs_frame_idx:05d}.png"
                    shared_mask_path = os.path.join(shared_mask_dir, mask_filename)

                    if os.path.exists(shared_mask_path):
                        # Save to local segment mask dir with LOCAL index (0000, 0001, etc.)
                        local_idx = abs_frame_idx - padded_start  # Use padded_start for correct local indexing!
                        local_mask_path = os.path.join(seg_mask_dir, f"{local_idx:04d}.png")
                        shutil.copy2(shared_mask_path, local_mask_path)
                        masks_succeeded += 1
                    else:
                        # Missing mask - break and regenerate all
                        break

                if masks_succeeded == len(masks_needed):
                    print(f"   [OK] Copied {masks_succeeded} masks from local storage (fast!)")
                    masks_downloaded = True
                else:
                    print(f"   [WARNING]  Only {masks_succeeded}/{len(masks_needed)} masks found locally - will regenerate")
            except Exception as copy_err:
                print(f"   [WARNING]  Local mask copy failed: {copy_err} - will regenerate")

        # Fallback: HTTP download (different PC - SLOW but necessary for distributed workers)
        elif not masks_downloaded and origin_base and shared_mask_dir:
            print(f"   📥 Downloading masks from remote location...")
            self.update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Downloading masks'})

            try:
                import requests
                masks_needed = list(range(padded_start, padded_end + 1))  # Use PADDED range for temporal context!
                masks_succeeded = 0

                for abs_frame_idx in masks_needed:
                    mask_filename = f"{abs_frame_idx:05d}.png"
                    mask_url = f"{origin_base.rstrip('/')}/temp/{os.path.basename(shared_mask_dir)}/{mask_filename}"

                    try:
                        r = requests.get(mask_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=10)
                        r.raise_for_status()

                        # Save to local segment mask dir with LOCAL index (0000, 0001, etc.)
                        local_idx = abs_frame_idx - padded_start  # Use padded_start for correct local indexing!
                        local_mask_path = os.path.join(seg_mask_dir, f"{local_idx:04d}.png")
                        with open(local_mask_path, 'wb') as f:
                            f.write(r.content)
                        masks_succeeded += 1
                    except Exception as mask_err:
                        # Silently skip - we'll regenerate if needed
                        break

                if masks_succeeded == len(masks_needed):
                    print(f"   [OK] Downloaded {masks_succeeded} masks from remote location")
                    masks_downloaded = True
                else:
                    print(f"   [WARNING]  Only {masks_succeeded}/{len(masks_needed)} masks downloaded - will regenerate")
            except Exception as download_err:
                print(f"   [WARNING]  Mask download failed: {download_err} - will regenerate")

        # Fallback: Download entire mask ZIP from B2 CDN (for distributed workers)
        if not masks_downloaded:
            masks_url = segment_data.get('masks_url')
            print(f"   [DEBUG] masks_url from segment_data: {masks_url}")
            if masks_url:
                print(f"   [B2] Downloading masks from B2 CDN: {masks_url}")
                self.update_state(state='PROCESSING', meta={'progress': 25, 'status': f'Downloading masks from B2'})
                try:
                    import zipfile
                    zip_path = os.path.join(TEMP_DIR, f"{video_id}_masks_seg{seg_idx}.zip")
                    r = requests.get(masks_url, timeout=120)
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        f.write(r.content)
                    print(f"   [B2] Downloaded {len(r.content) / 1024 / 1024:.1f} MB")

                    # Extract to shared_mask_dir (create if needed)
                    os.makedirs(shared_mask_dir, exist_ok=True)
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(shared_mask_dir)
                    os.remove(zip_path)

                    # Now copy masks for this segment
                    masks_succeeded = 0
                    for abs_frame_idx in masks_needed:
                        mask_filename = f"{abs_frame_idx:05d}.png"
                        shared_mask_path = os.path.join(shared_mask_dir, mask_filename)
                        if os.path.exists(shared_mask_path):
                            local_idx = abs_frame_idx - padded_start
                            local_mask_path = os.path.join(seg_mask_dir, f"{local_idx:04d}.png")
                            shutil.copy2(shared_mask_path, local_mask_path)
                            masks_succeeded += 1

                    if masks_succeeded == len(masks_needed):
                        print(f"   [OK] Downloaded and extracted {masks_succeeded} masks from B2 CDN")
                        masks_downloaded = True
                    else:
                        print(f"   [WARNING] Only {masks_succeeded}/{len(masks_needed)} masks found in B2 zip")
                except Exception as b2_err:
                    print(f"   [ERROR] B2 download failed: {b2_err}")

        # Define memory pipeline flag (check if we have all frames/masks in memory)
        using_memory_pipeline = (len(segment_frames_memory) == frames_copied and
                                 len(segment_masks_memory) == frames_copied)

        # Masks MUST be downloaded - no fallback detection!
        if not masks_downloaded:
            raise RuntimeError(f"Masks not found! B2 download failed for segment. Cannot proceed without masks.")
        else:
            # Masks were downloaded - need to extract bbox from them
            print(f"   [INFO] Using downloaded masks - extracting bbox info...")
            last_valid_bbox = bbox  # Use bbox from segment_data (from centralized detection)
            frames_with_watermark = frames_copied  # Assume all frames have watermarks (already filtered)

        # Update crop region based on detected bbox (calculate_crop_region handles min sizing)
        if last_valid_bbox:
            # Recalculate crop region with detected bbox (includes min_size=128 expansion)
            crop_x, crop_y, crop_w, crop_h = calculate_crop_region(last_valid_bbox, width, height, padding_ratio=0.2, min_size=128)
            print(f"   [CROP] Detected crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        else:
            print(f"   [INFO]  No watermark detected in this chunk - will skip ProPainter")

        # Crop frames to detected watermark region AND create masks on cropped frames
        # [INIT] OPTIMIZATION: Skip disk-based cropping if we have in-memory frames/masks
        if last_valid_bbox and not using_memory_pipeline:
            # Disk-based pipeline (fallback for remote workers or when memory cache unavailable)
            print(f"   [CROP]  Cropping frames to watermark region (disk-based)...")
            for frame_idx in range(frames_copied):
                frame_file = f"{frame_idx:04d}.png"
                frame_path = os.path.join(seg_frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    cv2.imwrite(os.path.join(seg_cropped_dir, frame_file), cropped)

            # Create masks on CROPPED frames (not full frames!)
            print(f"   [CREATE] Creating masks on cropped frames...")
            if not masks_downloaded and detector:
                # Calculate bbox relative to cropped region
                x1, y1, x2, y2 = last_valid_bbox
                # Translate bbox from full frame coords to crop coords
                crop_bbox_x1 = max(0, x1 - crop_x)
                crop_bbox_y1 = max(0, y1 - crop_y)
                crop_bbox_x2 = min(crop_w, x2 - crop_x)
                crop_bbox_y2 = min(crop_h, y2 - crop_y)
                crop_bbox = [crop_bbox_x1, crop_bbox_y1, crop_bbox_x2, crop_bbox_y2]

                # Create masks on cropped frames
                for frame_idx in range(frames_copied):
                    frame_file = f"{frame_idx:04d}.png"
                    cropped_frame_path = os.path.join(seg_cropped_dir, frame_file)
                    cropped_frame = cv2.imread(cropped_frame_path)
                    if cropped_frame is not None:
                        # Create mask on cropped frame with relative bbox
                        mask = detector.create_mask(cropped_frame, [{'bbox': crop_bbox}])
                        mask_path = os.path.join(seg_mask_dir, frame_file)
                        cv2.imwrite(mask_path, mask)
            else:
                # Masks were downloaded - crop them to the region
                print(f"   [CROP]  Cropping downloaded masks to region...")
                for frame_idx in range(frames_copied):
                    mask_file = f"{frame_idx:04d}.png"
                    mask_path = os.path.join(seg_mask_dir, mask_file)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        cv2.imwrite(mask_path, cropped_mask)
        elif using_memory_pipeline:
            print(f"   ⚡ Skipping disk-based cropping - using in-memory pipeline (saves ~600ms!)")

        print(f"   [OK] Prepared {frames_copied} frames and masks (including ±{neighbor_padding} neighbor padding)")

        # VALIDATION: Check first mask to catch bugs early
        if last_valid_bbox and frames_with_watermark > 0:
            test_mask = None

            # Get mask from memory or disk
            if using_memory_pipeline and len(segment_masks_memory) > 0:
                test_mask = segment_masks_memory[0]
            else:
                first_mask_path = os.path.join(seg_mask_dir, "0000.png")
                if os.path.exists(first_mask_path):
                    test_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)

            if test_mask is not None:
                white_pixels = np.sum(test_mask == 255)
                total_pixels = test_mask.size
                white_pct = (white_pixels / total_pixels) * 100

                print(f"   [INFO] Mask validation: {white_pct:.1f}% white pixels (target region)")

                # Warn if mask seems wrong
                if white_pct > 50:
                    print(f"   [WARNING]  WARNING: Mask is {white_pct:.1f}% white - suspicious! May cause black output.")
                elif white_pct < 0.1:
                    print(f"   [WARNING]  WARNING: Mask is {white_pct:.1f}% white - almost empty! No inpainting will occur.")

        # CONDITIONAL: Only run ProPainter if watermark was detected
        if not last_valid_bbox or frames_with_watermark == 0:
            print(f"   [SKIP]  No watermark detected - skipping ProPainter, encoding original frames")
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'No watermark - encoding original'})

            # Copy original frames to cleaned dir (no processing needed)
            # 🔥 SHARED BUFFER: Use global frame indices
            if using_memory_pipeline:
                # Write from memory (only core segment, skip padding)
                # 🔥 TEMPORAL CONTEXT FIX: Calculate padding offset
                padding_offset = start_frame - padded_start
                core_frames = segment_frames_memory[padding_offset:padding_offset + seg_duration]
                print(f"   💾 Writing {len(core_frames)} core segment frames from memory to shared buffer (skipping {padding_offset} padding frames)...")
                for local_idx, frame in enumerate(core_frames):
                    # Use GLOBAL frame index for shared buffer
                    global_frame_idx = start_frame + local_idx
                    frame_file = f"{global_frame_idx:04d}.png"
                    dst = os.path.join(shared_cleaned_dir, frame_file)

                    # Only write if not already written by another segment
                    if not os.path.exists(dst):
                        cv2.imwrite(dst, frame)
            else:
                # Copy from disk
                for local_idx in range(frames_copied):
                    # Use GLOBAL frame index for shared buffer
                    global_frame_idx = start_frame + local_idx
                    frame_file = f"{global_frame_idx:04d}.png"
                    src = os.path.join(seg_frames_dir, f"{local_idx:04d}.png")  # Local source
                    dst = os.path.join(shared_cleaned_dir, frame_file)  # Global destination

                    # Only write if not already written by another segment
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)

        else:
            # Run ProPainter on this segment - watermark detected!
            print(f"   [PAINT] Running ProPainter on {frames_with_watermark} watermarked frames...")
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Running ProPainter'})

            try:
                # Use cached ProPainter pipeline (pre-loaded at worker startup)
                faster_propainter_pipeline = get_propainter_pipeline()

                import torch
                use_fp16 = torch.cuda.is_available()

                # [INIT] EXTREME SPEED: Use already-loaded memory frames (skip FRAME_CACHE re-access!)
                frames_array = None
                masks_array = None

                if using_memory_pipeline:
                    print(f"   ⚡ Cropping {len(segment_frames_memory)} frames/masks in memory (ZERO disk I/O!)")
                    import time
                    crop_start = time.time()

                    # Crop to watermark region in memory (no disk I/O!)
                    frames_array = []
                    masks_array = []

                    for frame in segment_frames_memory:
                        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        frames_array.append(cropped)

                    for mask in segment_masks_memory:
                        cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        masks_array.append(cropped_mask)

                    crop_duration = time.time() - crop_start
                    print(f"   [OK] Cropped {len(frames_array)} frames + {len(masks_array)} masks in memory: {crop_duration:.3f}s")
                    print(f"   [RUNNING] Eliminated ~1000ms+ of disk I/O - pure in-memory pipeline!")

                if frames_array is not None and masks_array is not None:
                    print(f"   [GPU] GPU available: {use_fp16}, Running ProPainter with IN-MEMORY arrays...")
                else:
                    print(f"   [GPU] GPU available: {use_fp16}, Running ProPainter with disk paths (fallback)...")

                faster_propainter_pipeline(
                    video=seg_cropped_dir,
                    mask=seg_mask_dir,
                    output=seg_output_dir,
                    resize_ratio=1.0,
                    mask_dilation=4,
                    ref_stride=15,
                    neighbor_length=10,
                    subvideo_length=120,
                    raft_iter=20,
                    mode="video_inpainting",
                    save_frames=True,
                    fp16=use_fp16,
                    frames_array=frames_array,
                    masks_array=masks_array
                )

                print(f"   [OK] ProPainter complete for segment {seg_idx+1}")
            except Exception as e:
                print(f"[ERROR] ProPainter failed on segment {seg_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                clear_gpu_memory()

            # Merge cleaned region back to full frames
            print(f"   🔗 Merging cleaned region (in-memory)...")
            self.update_state(state='PROCESSING', meta={'progress': 80, 'status': f'Merging results'})

            seg_propainter_frames = os.path.join(seg_output_dir, os.path.basename(seg_cropped_dir), 'frames')
            if not os.path.exists(seg_propainter_frames):
                raise RuntimeError(f"ProPainter output not found for segment {seg_idx}")

            # Load all frames into memory first (faster than disk I/O in loop)
            # [INIT] EXTREME SPEED: Use in-memory frames if available (no disk reads!)
            original_frames = []
            original_masks = []  # 🔥 MASK COMPOSITING: Need masks for alpha blending
            cleaned_frames = []

            # 🔥 TEMPORAL CONTEXT FIX: Calculate padding offset
            # ProPainter processed frames WITH padding, we only need core segment frames
            padding_offset = start_frame - padded_start  # How many padding frames before core segment
            print(f"   [CONTEXT] ProPainter processed {len(segment_frames_memory)} frames (padding offset: {padding_offset})")

            if using_memory_pipeline:
                # Use already-loaded memory frames (ZERO disk I/O!)
                # Extract ONLY core segment frames (skip padding)
                print(f"   ⚡ Extracting core segment frames + masks from memory (frames {padding_offset} to {padding_offset + seg_duration})...")
                original_frames = segment_frames_memory[padding_offset:padding_offset + seg_duration]
                original_masks = segment_masks_memory[padding_offset:padding_offset + seg_duration]
            else:
                # Disk-based fallback
                print(f"   [MASKS] Loading {seg_duration} original frames + masks from disk...")
                for frame_idx in range(seg_duration):
                    frame_file = f"{frame_idx:04d}.png"
                    orig = cv2.imread(os.path.join(seg_frames_dir, frame_file))
                    original_frames.append(orig)
                    # Load corresponding mask
                    mask = cv2.imread(os.path.join(seg_mask_dir, frame_file), cv2.IMREAD_GRAYSCALE)
                    original_masks.append(mask)

            # Load cleaned frames from ProPainter output (skip padding frames)
            print(f"   [OUTPUT] Extracting core segment from ProPainter output (frames {padding_offset} to {padding_offset + seg_duration})...")
            for frame_idx in range(padding_offset, padding_offset + seg_duration):
                frame_file = f"{frame_idx:04d}.png"
                clean = cv2.imread(os.path.join(seg_propainter_frames, frame_file))
                cleaned_frames.append(clean)

            # 🔥 SHARED BUFFER MERGE: Load existing frame state, apply this segment's edit, save back
            # This allows multiple segments to cooperatively edit the same frames
            print(f"   🔗 Merging to shared frame buffer with mask-based alpha compositing...")
            for local_idx, (original, cleaned_crop, segment_mask) in enumerate(zip(original_frames, cleaned_frames, original_masks)):
                # Use GLOBAL frame index for shared buffer
                global_frame_idx = start_frame + local_idx
                frame_file = f"{global_frame_idx:04d}.png"
                shared_frame_path = os.path.join(shared_cleaned_dir, frame_file)

                # Load existing state (may have edits from other segments) or original
                if os.path.exists(shared_frame_path):
                    # Another segment already edited this frame - load current state
                    result_frame = cv2.imread(shared_frame_path)
                    if result_frame is None:
                        result_frame = original.copy() if original is not None else np.zeros((height, width, 3), dtype=np.uint8)
                elif original is not None:
                    result_frame = original.copy()
                else:
                    continue

                # 🔥 MASK COMPOSITING: Use mask to blend only the inpainted region
                # This preserves other segments' work in non-masked areas
                if cleaned_crop is not None and result_frame is not None and segment_mask is not None:
                    # Crop mask to ROI
                    cropped_mask = segment_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

                    # 🔥 DEFENSIVE CHECK: Skip if cropped mask is empty or wrong shape
                    if cropped_mask.size == 0 or cropped_mask.shape[0] == 0 or cropped_mask.shape[1] == 0:
                        # Empty mask - just paste cleaned crop directly
                        result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop
                    elif cropped_mask.shape[:2] != cleaned_crop.shape[:2]:
                        # Shape mismatch - resize mask to match
                        cropped_mask = cv2.resize(cropped_mask, (cleaned_crop.shape[1], cleaned_crop.shape[0]))
                        mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0 if len(cropped_mask.shape) == 2 else cropped_mask.astype(float) / 255.0
                        roi = result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].astype(float)
                        cleaned_crop_float = cleaned_crop.astype(float)
                        blended = (cleaned_crop_float * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
                        result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = blended
                    else:
                        # Convert mask to 3-channel float [0, 1] for alpha blending
                        if len(cropped_mask.shape) == 2:
                            # Grayscale mask - convert to 3 channels
                            mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
                        else:
                            mask_3ch = cropped_mask.astype(float) / 255.0

                        # Alpha composite: blend cleaned region using mask as alpha
                        roi = result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].astype(float)
                        cleaned_crop_float = cleaned_crop.astype(float)
                        blended = (cleaned_crop_float * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

                        # Paste blended result back
                        result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = blended

                # Save back to shared buffer
                cv2.imwrite(shared_frame_path, result_frame)

            print(f"   [OK] Merged {len(cleaned_frames)} frames to shared buffer: {shared_cleaned_dir}")

        # ⚡ PARALLEL ENCODING: Encode segment MP4 immediately (Blackwell NVENC!)
        print(f"   [OK] Cleaned frames ready in: {seg_cleaned_dir}")
        # ⚡ BACKGROUND ENCODING OPTIMIZATION: Signal encoder thread instead of blocking!
        # Worker returns immediately and starts next segment while encoding happens in background
        self.update_state(state='PROCESSING', meta={'progress': 85, 'status': f'Segment complete - signaling encoder'})

        # Count cleaned frames
        cleaned_frame_count = len([f for f in os.listdir(seg_cleaned_dir) if f.endswith('.png')])
        if cleaned_frame_count == 0:
            raise RuntimeError(f"No cleaned frames found in {seg_cleaned_dir}")

        print(f"   [OK] {cleaned_frame_count} cleaned frames ready - signaling background encoder!")

        # Store segment metadata in Redis for background encoder
        redis_client = celery.backend.client
        segment_key = f"video:{video_id}:segment:{seg_idx}"
        redis_client.hset(segment_key, 'cleaned_dir', seg_cleaned_dir)
        redis_client.hset(segment_key, 'fps', str(fps))
        redis_client.hset(segment_key, 'frame_count', str(cleaned_frame_count))
        redis_client.hset(segment_key, 'base_name', base_name)
        redis_client.hset(segment_key, 'status', 'ready_for_encoding')
        # 🔥 SHARED BUFFER: Store frame range for encoder
        redis_client.hset(segment_key, 'start_frame', str(start_frame))
        redis_client.hset(segment_key, 'end_frame', str(end_frame))

        # Set video-level metadata (idempotent - safe to set multiple times)
        redis_client.set(f"video:{video_id}:total_segments", total_segments)
        redis_client.set(f"video:{video_id}:base_name", base_name)
        redis_client.set(f"video:{video_id}:video_path", video_path)  # For audio merge later

        # Signal background encoder thread via Redis pub/sub (REAL-TIME!)
        redis_client.publish('segment_ready', json.dumps({
            'video_id': video_id,
            'seg_idx': seg_idx,
            'total_segments': total_segments
        }))

        print(f"[OK] Segment {seg_idx+1}/{total_segments} signaled to background encoder - worker returning immediately!")

        # Cleanup temp directories EXCEPT cleaned_dir (encoder needs it!)
        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir]:
            shutil.rmtree(path, ignore_errors=True)

        result = {
            'seg_idx': seg_idx,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frames_processed': seg_duration,
            'status': 'ready_for_encoding',  # Background encoder will encode this
        }

        print(f"[OK] Segment {seg_idx+1}/{total_segments} complete - worker free to process next segment!")

        # Note: Don't call self.update_state(state='SUCCESS') - it overrides the return value!
        # Celery automatically sets state to SUCCESS when the task returns normally
        # Chord will automatically trigger finalize when all segments complete

        # 🔥 VRAM cleanup after segment processing
        cleanup_vram(f"after segment {seg_idx+1}/{total_segments}")

        return result

    except Exception as e:
        print(f"[ERROR] Error processing segment {seg_idx}: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.finalize_video')
def finalize_video_task(self, segment_results, prepare_result):
    """
    Phase 3: Return immediately - encoding happens in background encoder thread!

    NOTE: This task returns immediately. The background encoder handles all encoding + finalization asynchronously.
    Check Redis for final video status: video:{video_id}:status and video:{video_id}:final_path
    """
    try:
        video_id = prepare_result['video_id']
        base_name = prepare_result['base_name']
        total_segments = prepare_result['total_segments']

        print(f"\n[FINALIZE TASK] All segments complete! Background encoder handling finalization for: {base_name}")
        print(f"[FINALIZE TASK] Encoding + concatenation happening asynchronously in background (parallel NVENC)")

        # Return immediately - don't block Celery worker!
        # Background encoder will update Redis when complete:
        # - video:{video_id}:status = "complete"
        # - video:{video_id}:final_path = <path to final video>

        result = {
            'status': 'background_encoding',
            'message': f'Segments complete. Encoding + finalization in progress (background thread)',
            'video_id': video_id,
            'total_segments': total_segments,
            'check_status_key': f'video:{video_id}:status',
            'final_path_key': f'video:{video_id}:final_path'
        }

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Background encoding in progress'})
        print(f"[FINALIZE TASK] Returned immediately - worker free! Check Redis keys for completion.")
        return result

        # Timeout
        raise RuntimeError(f"Background encoder did not complete within {max_wait_seconds}s")

    except Exception as e:
        print(f"[ERROR] Error finalizing video: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.coordinate_segments')
def coordinate_segments_task(self, segment_task_ids, prepare_result):
    """
    Coordinator task that waits for all segment tasks to complete,
    then triggers finalize_video_task.

    This replaces chord functionality with reliable manual coordination.
    """
    try:
        from celery.result import AsyncResult

        total_segments = len(segment_task_ids)
        print(f"[INFO] Coordinator: Waiting for {total_segments} segment tasks to complete...")
        self.update_state(state='STARTED', meta={'progress': 0, 'status': f'Waiting for {total_segments} segments'})

        # Wait for all segment tasks to complete
        segment_results = []
        for i, task_id in enumerate(segment_task_ids):
            print(f"   ⏳ Waiting for segment task {i+1}/{total_segments}: {task_id}")
            task_result = AsyncResult(task_id, app=celery)

            # Wait for this segment to complete (blocks, but doesn't block workers from processing)
            result = task_result.get(timeout=3600)  # 1 hour timeout per segment
            segment_results.append(result)

            progress = int((i + 1) / total_segments * 80)
            self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'{i+1}/{total_segments} segments complete'})
            print(f"   [OK] Segment {i+1}/{total_segments} complete!")

        print(f"[OK] All {total_segments} segments complete! Launching finalize task...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Finalizing video'})

        # All segments done - now finalize
        final_result = finalize_video_task.apply_async(args=[segment_results, prepare_result])

        # Wait for finalize to complete
        final_output = final_result.get(timeout=600)  # 10 minute timeout

        print(f"[OK] Video processing complete!")
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})

        return final_output

    except Exception as e:
        print(f"[ERROR] Coordinator failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark._launch_segments')
def launch_segments_task(self, prepare_result):
    """Create a chord of segment tasks and launch them.
    Returns the chord result for tracking.
    """
    try:
        from celery import chord, group

        if not prepare_result or 'segments' not in prepare_result:
            raise RuntimeError("Video preparation failed")

        segments = prepare_result['segments']
        total_segments = len(segments)

        # Add total_segments to each segment data
        for seg in segments:
            seg['total_segments'] = total_segments

        print(f"[INIT] Launching chord: {total_segments} segments in parallel across all workers...")

        # Create the chord
        header = group([process_segment_task.s(seg) for seg in segments])
        callback = finalize_video_task.s(prepare_result)

        # Apply the chord asynchronously - this dispatches tasks immediately
        chord_result = chord(header)(callback)

        print(f"[OK] Chord dispatched: {total_segments} segment tasks queued")
        print(f"   Workers will pick up tasks and process in parallel")

        # Wait for the chord to complete and return the final result
        # This blocks this task but doesn't block workers from processing segments
        final_result = chord_result.get(timeout=3600)

        print(f"[OK] All segments processed and finalized!")
        return final_result

    except Exception as e:
        print(f"[ERROR] Segment launch failed: {e}")
        import traceback; traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.remove_video_distributed')
def process_video_distributed_task(self, video_path, api_base=None, temp_base=None):
    """
    DISTRIBUTED VIDEO PROCESSING - Main entry point

    This task coordinates multiple workers across different machines to process
    one video together. It uses Celery's chord pattern to:
    1. Prepare video (YOLO detection, segmentation) - runs on one worker
    2. Process segments in parallel - distributes across ALL available workers/GPUs
    3. Finalize video (concatenate, merge audio) - runs on one worker

    All workers (across all computers) will collaborate on the same video job.
    """
    try:
        from celery import chain, chord, group

        print(f"[RUNNING] Starting DISTRIBUTED video processing: {video_path}")
        print(f"   All available workers will collaborate on this video")

        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Initializing distributed processing'})

        # Phase 1: Prepare video (runs on one worker)
        print("📋 Phase 1: Preparing video for distribution...")
        prepare_result = prepare_video_task.apply_async(
            args=[video_path],
            kwargs={'api_base': api_base, 'temp_base': temp_base}
        ).get()

        if not prepare_result or 'segments' not in prepare_result:
            raise RuntimeError("Video preparation failed")

        segments = prepare_result['segments']
        total_segments = len(segments)

        print(f"[OK] Video prepared: {total_segments} segments ready for distributed processing")
        print(f"🌐 Distributing segments across all available workers...")

        self.update_state(
            state='PROCESSING',
            meta={'progress': 50, 'status': f'Distributing {total_segments} segments across workers'}
        )

        # Add total_segments to each segment data
        for seg_data in segments:
            seg_data['total_segments'] = total_segments

        # Phase 2: Process segments in parallel (distributes across ALL workers)
        # Using chord: when all segment tasks complete, finalize task runs automatically
        print(f"[INIT] Phase 2: Processing {total_segments} segments in parallel...")

        workflow = chord(
            # Header: All segment processing tasks (run in parallel across all workers)
            group([
                process_segment_task.s(seg_data)
                for seg_data in segments
            ]),
            # Callback: Finalization task (runs when all segments complete)
            finalize_video_task.s(prepare_result)
        )

        # Execute the distributed workflow
        result = workflow.apply_async()

        print(f"[INFO] Workflow submitted:")
        print(f"   - {total_segments} segment tasks queued")
        print(f"   - Tasks will be picked up by ANY available worker")
        print(f"   - Finalization will run automatically when all segments complete")

        # Wait for the entire workflow to complete
        final_result = result.get(timeout=3600)  # 1 hour timeout

        print(f"[OK] DISTRIBUTED processing complete!")
        print(f"   Final video: {final_result.get('path')}")

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Distributed processing complete'})

        return final_result

    except Exception as e:
        print(f"[ERROR] Distributed processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# ORIGINAL MONOLITHIC TASKS (kept for backward compatibility)
# ============================================================================

@celery.task(bind=True, name='watermark.remove_image')
def process_image_task(self, image_path):
    """
    Single-image watermark removal using YOLO mask + ProPainter with FP16.

    ProPainter works on video sequences, so we:
    1. Duplicate the image into a 3-frame sequence
    2. Generate masks for each frame
    3. Run ProPainter with --fp16
    4. Extract the middle frame as result
    """
    try:
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Loading detector'})
        det = get_detector()

        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing - see logs for details")

        # If running on a remote worker (e.g., Salad), the local path from the API host
        # won't exist. In that case, try to download the file from the API via TUNNEL_URL.
        if not os.path.exists(image_path):
            tunnel = os.getenv('TUNNEL_URL')
            try:
                from urllib.parse import urljoin
                import requests
            except Exception:
                tunnel = None

            if tunnel:
                # Handle Windows-style paths coming from the API host
                try:
                    from pathlib import PureWindowsPath
                    base_name = PureWindowsPath(image_path).name
                except Exception:
                    base_name = os.path.basename(image_path.replace('\\', '/'))

                download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                print(f"🌐 Image not found locally. Downloading from: {download_url}")
                try:
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=30)
                    r.raise_for_status()
                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    image_path = remote_cached
                    print(f"[OK] Downloaded to: {image_path}")
                except Exception as dl_err:
                    print(f"[ERROR] Failed to download image from API host: {dl_err}")
                    raise Exception(f"Image file not found and remote download failed: {base_name}")
            else:
                raise Exception(f"Image file not found: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception('Failed to read image')

        height, width = img.shape[:2]
        base_name = Path(image_path).stem
        unique_suffix = self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8]

        self.update_state(state='PROCESSING', meta={'progress': 15, 'status': 'Detecting watermark'})
        detection_start = performance_checkpoint("YOLO Detection")
        detections = det.detect(img, confidence_threshold=0.20, padding=0)  # Lower threshold for faint watermarks
        performance_checkpoint("YOLO Detection", detection_start)

        # If no detections, return original
        if not detections:
            out_name = base_name + '_clean.png'
            out_path = os.path.join(RESULT_DIR, out_name)
            cv2.imwrite(out_path, img)
            return {'path': out_path}

        # Create 3-frame sequence directory for ProPainter
        frame_dir = os.path.join(TEMP_DIR, f"image_frames_{unique_suffix}")
        mask_dir = os.path.join(PROPAINTER_MASK_ROOT, f"{base_name}_{unique_suffix}")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        self.update_state(state='PROCESSING', meta={'progress': 25, 'status': 'Preparing frames'})

        # Create 3 identical frames (ProPainter needs temporal context)
        for i in range(3):
            frame_path = os.path.join(frame_dir, f"{i:04d}.png")
            cv2.imwrite(frame_path, img)

        # Generate mask - precise detection like YOLO GOOD version
        mask = det.create_mask(img, detections)
        if mask is None or mask.size == 0:
            out_name = base_name + '_clean.png'
            out_path = os.path.join(RESULT_DIR, out_name)
            cv2.imwrite(out_path, img)
            shutil.rmtree(frame_dir, ignore_errors=True)
            shutil.rmtree(mask_dir, ignore_errors=True)
            return {'path': out_path}

        # Ensure binary mask
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255

        # Save same mask for all 3 frames
        for i in range(3):
            mask_path = os.path.join(mask_dir, f"{i:04d}.png")
            cv2.imwrite(mask_path, mask)

        self.update_state(state='PROCESSING', meta={'progress': 45, 'status': 'Running faster-propainter pipeline'})

        # Use direct faster-propainter pipeline instead of subprocess for 3x speedup
        pipeline_start = performance_checkpoint("faster-propainter Pipeline")
        try:
            # Use cached ProPainter pipeline (pre-loaded at worker startup)
            faster_propainter_pipeline = get_propainter_pipeline()

            import torch
            use_fp16 = torch.cuda.is_available()
            
            print(f"[RUNNING] Running direct faster-propainter pipeline: FP16={use_fp16} + neighbor_length=20 + ref_stride=10 + subvideo_length=80")
            
            # Direct pipeline call - TRUE FASTER-PROPAINTER SETTINGS
            faster_propainter_pipeline(
                video=frame_dir,                    # Input frame directory
                mask=mask_dir,                      # Mask directory  
                output=PROPAINTER_OUTPUT_ROOT,      # Output directory
                resize_ratio=1.0,                   # Keep original resolution for images
                mask_dilation=4,                    # faster-propainter: standard mask dilation
                ref_stride=15,                      # faster-propainter: faster processing
                neighbor_length=10,                 # faster-propainter: reduced for speed
                subvideo_length=60,                 # faster-propainter: reduced for speed
                raft_iter=20,                       # ProPainter default for quality
                mode="video_inpainting",            # Standard inpainting mode
                save_frames=True,                   # Save individual frames
                fp16=use_fp16                       # Enable FP16 if available
            )
            
            print("[OK] faster-propainter pipeline completed successfully")
            performance_checkpoint("faster-propainter Pipeline", pipeline_start)
            
        except Exception as e:
            print(f"[ERROR] faster-propainter pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"faster-propainter pipeline failed: {e}")
        finally:
            # Clear GPU memory after processing
            clear_gpu_memory()

        # Extract middle frame (frame 0001 out of 0000, 0001, 0002)
        # Frame directory name from ProPainter is the basename of input dir
        propainter_output_name = os.path.basename(frame_dir)
        save_root = os.path.join(PROPAINTER_OUTPUT_ROOT, propainter_output_name)
        middle_frame_path = os.path.join(save_root, 'frames', '0001.png')

        if not os.path.exists(middle_frame_path):
            # Fallback: try frame 0000 if middle doesn't exist
            middle_frame_path = os.path.join(save_root, 'frames', '0000.png')

        if not os.path.exists(middle_frame_path):
            raise RuntimeError(f"ProPainter output frame not found: {middle_frame_path}")

        self.update_state(state='PROCESSING', meta={'progress': 85, 'status': 'Finalizing'})

        # Copy result to final location
        out_name = base_name + '_clean.png'
        out_path = os.path.join(RESULT_DIR, out_name)
        shutil.copy2(middle_frame_path, out_path)

        # Cleanup temporary directories
        try:
            shutil.rmtree(frame_dir, ignore_errors=True)
            shutil.rmtree(mask_dir, ignore_errors=True)
            shutil.rmtree(save_root, ignore_errors=True)
        except Exception as cleanup_exc:
            print(f"[WARNING]  Failed to cleanup temp directories: {cleanup_exc}")

        print(f"[OK] Image processed with ProPainter FP16: {out_path}")

        # If running remotely and the API host is reachable via TUNNEL_URL,
        # upload the result back so the frontend can download from the API server.
        uploaded_path = None
        tunnel = os.getenv('TUNNEL_URL')
        if tunnel and os.getenv('UPLOAD_RESULT_BACK', '1') == '1':
            try:
                import requests
                upload_url = tunnel.rstrip('/') + '/api/upload-result'
                # Quick connectivity test (15 second timeout) - fail fast if server down
                requests.head(tunnel, timeout=15, headers={'ngrok-skip-browser-warning': 'true'})
                print(f"[UPLOAD]  Uploading result to API host: {upload_url}")
                with open(out_path, 'rb') as fp:
                    resp = requests.post(
                        upload_url,
                        headers={'ngrok-skip-browser-warning': 'true'},
                        files={'file': (os.path.basename(out_path), fp, 'image/png')},
                        timeout=60
                    )
                if resp.ok:
                    j = resp.json()
                    if j.get('status') == 'success' and j.get('result_url'):
                        uploaded_path = j['result_url']
                        print(f"[OK] Uploaded result registered at: {uploaded_path}")
                else:
                    print(f"[WARNING]  Upload back failed: HTTP {resp.status_code}")
            except Exception as up_err:
                print(f"[WARNING]  Upload back error: {up_err}")

        return {'path': uploaded_path or out_path}
    except Exception as e:
        print(f"[ERROR] Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.remove_video')
def process_video_task(self, video_path):
    """
    Background task for video watermark removal using YOLO + ProPainter.
    """
    try:
        self.update_state(
            state='STARTED',
            meta={'progress': 0, 'status': 'Loading YOLO detector'}
        )

        detector = get_detector()
        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing - see logs for details")

        # If running on a remote worker (e.g., Salad), the local path from the API host
        # won't exist. In that case, try to download the file from the API via TUNNEL_URL.
        if not os.path.exists(video_path):
            tunnel = os.getenv('TUNNEL_URL')
            try:
                from urllib.parse import urljoin
                import requests
            except Exception:
                tunnel = None

            if tunnel:
                # Handle Windows-style paths coming from the API host
                try:
                    from pathlib import PureWindowsPath
                    base_name = PureWindowsPath(video_path).name
                except Exception:
                    base_name = os.path.basename(video_path.replace('\\', '/'))

                download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                print(f"🌐 Video not found locally. Downloading from: {download_url}")
                try:
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=30)
                    r.raise_for_status()
                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    video_path = remote_cached
                    print(f"[OK] Downloaded to: {video_path}")
                except Exception as dl_err:
                    print(f"[ERROR] Failed to download video from API host: {dl_err}")
                    raise Exception(f"Video file not found and remote download failed: {base_name}")
            else:
                raise Exception(f"Video file not found: {video_path}")

        print(f"Opening video for ProPainter: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")

        self.update_state(
            state='PROCESSING',
            meta={'progress': 5, 'status': 'Scanning video frames'}
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        base_name = Path(video_path).stem
        unique_suffix = self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8]
        mask_dir = os.path.join(PROPAINTER_MASK_ROOT, f"{base_name}_{unique_suffix}")
        os.makedirs(mask_dir, exist_ok=True)

        zero_mask = np.zeros((height, width), dtype=np.uint8)
        last_valid_bbox = None
        frames_processed = 0
        frames_with_watermark = 0
        detect_interval = 3   # Detect every 3rd frame to catch movements
        warmup_frames = 60    # Detect every frame for first N frames or until first hit
        hit_found = False
        det_conf = 0.15       # Balanced confidence like YOLO GOOD version

        # Track bbox per frame for segmentation
        bboxes_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            progress = 5 + int((frames_processed / max(total_frames, 1)) * 40)
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': progress,
                    'status': f'Building masks {frames_processed}/{total_frames}'
                }
            )

            # Only run YOLO detection every Nth frame, but always during warmup until first hit
            if (frames_processed % detect_interval == 0) or (not hit_found and frames_processed < warmup_frames):
                detections = detector.detect(frame, confidence_threshold=det_conf, padding=0)  # Precise detection like YOLO GOOD
                active_detections = detections
                actual_detections = actual_detections + 1 if 'actual_detections' in locals() else 1

                if detections:
                    frames_with_watermark += 1
                    primary = detections[0]
                    if primary.get('confidence', 0) >= det_conf:
                        last_valid_bbox = primary['bbox']
                    hit_found = True
                elif last_valid_bbox:
                    active_detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]
            else:
                # Reuse last detected bounding box for frames in between
                active_detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}] if last_valid_bbox else []

            if active_detections:
                mask = detector.create_mask(frame, active_detections)
                # Track bbox for segmentation
                bboxes_per_frame.append(active_detections[0]['bbox'])
            else:
                mask = zero_mask
                # No detection for this frame
                bboxes_per_frame.append(None)

            mask_path = os.path.join(mask_dir, f"{frames_processed:04d}.png")
            cv2.imwrite(mask_path, mask)
            frames_processed += 1

        cap.release()

        if frames_processed == 0:
            raise RuntimeError("No frames were processed - video may be corrupted")

        detections_run = actual_detections if 'actual_detections' in locals() else (frames_processed + detect_interval - 1) // detect_interval
        print(f"[OK] Masks generated: {frames_processed} frames total")
        print(f"   [REGEN] YOLO detections: {detections_run}/{frames_processed} frames (detect_interval={detect_interval})")
        print(f"   💧 Frames with watermark: {frames_with_watermark}")

        # Detect position segments for smart cropping
        from segment_detector import detect_segments, merge_adjacent_segments, should_use_cropping

        segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
        if segments:
            segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
            print(f"[INFO] Detected {len(segments)} watermark position segments:")
            for i, (start, end, bbox) in enumerate(segments):
                duration = end - start + 1
                print(f"   Segment {i+1}: frames {start}-{end} ({duration} frames) bbox={bbox}")

        use_cropping = should_use_cropping(segments, width, height, min_speedup=5.0) if segments else False

        if use_cropping and segments:
            print("[RUNNING] Smart cropping enabled - processing segments individually")
            import subprocess

            # Extract original frames once for merging later
            original_frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{unique_suffix}_originals")
            os.makedirs(original_frames_dir, exist_ok=True)

            print(f"📷 Extracting original frames for merging...")
            extract_cmd = [
                FFMPEG_EXE, '-i', video_path,
                '-qscale:v', '2',
                '-start_number', '0',  # Start numbering at 0 to match frame indices
                os.path.join(original_frames_dir, '%04d.png')
            ]
            subprocess.run(extract_cmd, capture_output=True, check=True)

            segment_context = {
                'base_name': base_name,
                'unique_suffix': unique_suffix,
                'width': width,
                'height': height,
                'fps': fps,
                'mask_dir': mask_dir,
                'original_frames_dir': original_frames_dir,
            }

            # Phase 1: ProPainter processing (GPU-bound, parallel across multiple GPUs)
            segment_cleaned_dirs = [None] * len(segments)
            segment_temp_dirs = [None] * len(segments)

            # Determine number of GPUs available
            num_gpus = min(SEGMENT_WORKERS, len(segments))

            print(f"[PAINT] Phase 1: ProPainter processing ({len(segments)} segments across {num_gpus} GPUs)")
            print(f"   [RUNNING] Each segment gets its own GPU for true parallel processing")
            self.update_state(
                state='PROCESSING',
                meta={'progress': 55, 'status': f'ProPainter processing ({num_gpus} GPUs)'}
            )

            if num_gpus > 1:
                # Parallel processing with GPU assignment
                completed = 0
                print(f"[INIT] Processing {len(segments)} segments in parallel across {num_gpus} GPUs...")

                # Create process pool with GPU initialization
                with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                    # Submit all segments with GPU assignment
                    futures = {}
                    for seg_idx in range(len(segments)):
                        gpu_id = seg_idx % num_gpus  # Round-robin GPU assignment
                        # Initialize worker with GPU, then process segment
                        future = executor.submit(
                            lambda idx, gpu, seg, ctx: (_init_gpu_worker(gpu), _process_propainter_segment(idx, len(segments), seg, ctx))[1],
                            seg_idx,
                            gpu_id,
                            segments[seg_idx],
                            segment_context
                        )
                        futures[future] = seg_idx

                    try:
                        for future in as_completed(futures):
                            seg_idx = futures[future]
                            _, cleaned_dir, crop_info, temp_dirs = future.result()
                            segment_cleaned_dirs[seg_idx] = cleaned_dir
                            segment_temp_dirs[seg_idx] = temp_dirs
                            completed += 1
                            progress = 55 + int((completed / len(segments)) * 15)
                            self.update_state(
                                state='PROCESSING',
                                meta={'progress': progress, 'status': f'ProPainter {completed}/{len(segments)} complete'}
                            )
                            print(f"[OK] ProPainter segment {seg_idx+1}/{len(segments)} complete ({completed}/{len(segments)} total)")
                    except Exception:
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        raise
            else:
                # Sequential fallback
                for seg_idx, segment in enumerate(segments):
                    _, cleaned_dir, crop_info, temp_dirs = _process_propainter_segment(seg_idx, len(segments), segment, segment_context)
                    segment_cleaned_dirs[seg_idx] = cleaned_dir
                    segment_temp_dirs[seg_idx] = temp_dirs
                    progress = 55 + int(((seg_idx + 1) / len(segments)) * 15)
                    self.update_state(
                        state='PROCESSING',
                        meta={'progress': progress, 'status': f'ProPainter segment {seg_idx+1}/{len(segments)}'}
                    )
                    print(f"[OK] ProPainter segment {seg_idx+1}/{len(segments)} complete")

            if any(d is None for d in segment_cleaned_dirs):
                raise RuntimeError("One or more segments failed ProPainter processing")

            # Clear GPU memory after all ProPainter processing completes
            print("[CLEANUP] Clearing GPU memory after ProPainter phase...")
            clear_gpu_memory()

            # Phase 2: Video encoding (CPU-bound, parallel)
            max_workers = min(4, len(segments))  # Use up to 4 threads for encoding
            print(f"\n[ENCODE]  Phase 2: Video encoding ({len(segments)} segments in parallel, {max_workers} workers)")
            self.update_state(
                state='PROCESSING',
                meta={'progress': 70, 'status': f'Encoding segments'}
            )

            segment_videos = [None] * len(segments)
            encode_context = {
                'base_name': base_name,
                'unique_suffix': unique_suffix,
                'fps': fps
            }

            if max_workers > 1:
                completed = 0
                print(f"🧵 Encoding {len(segments)} segments in parallel...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _encode_segment,
                            seg_idx,
                            len(segments),
                            segment_cleaned_dirs[seg_idx],
                            encode_context,
                            segment_temp_dirs[seg_idx]
                        ): seg_idx
                        for seg_idx in range(len(segments))
                    }
                    try:
                        for future in as_completed(futures):
                            seg_idx = futures[future]
                            _, seg_video_path = future.result()
                            segment_videos[seg_idx] = seg_video_path
                            completed += 1
                            progress = 70 + int((completed / len(segments)) * 10)
                            self.update_state(
                                state='PROCESSING',
                                meta={'progress': progress, 'status': f'Encoding {completed}/{len(segments)} complete'}
                            )
                            print(f"[OK] Encoded segment {seg_idx+1}/{len(segments)} ({completed}/{len(segments)} total)")
                    except Exception:
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        raise
            else:
                for seg_idx in range(len(segments)):
                    _, seg_video_path = _encode_segment(
                        seg_idx,
                        len(segments),
                        segment_cleaned_dirs[seg_idx],
                        encode_context,
                        segment_temp_dirs[seg_idx]
                    )
                    segment_videos[seg_idx] = seg_video_path
                    progress = 70 + int(((seg_idx + 1) / len(segments)) * 10)
                    self.update_state(
                        state='PROCESSING',
                        meta={'progress': progress, 'status': f'Encoding segment {seg_idx+1}/{len(segments)}'}
                    )

            if any(path is None for path in segment_videos):
                raise RuntimeError("One or more segments failed encoding")

            # Concatenate all segments
            print(f"\n[SEGMENT] Concatenating {len(segment_videos)} segments...")
            self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Concatenating segments'})

            concat_list_file = os.path.join(TEMP_DIR, f"{base_name}_{unique_suffix}_concat.txt")
            with open(concat_list_file, 'w') as f:
                for seg_video in segment_videos:
                    f.write(f"file '{seg_video}'\n")

            temp_processed = os.path.join(RESULT_DIR, f"{base_name}_propainter_video.mp4")
            concat_cmd = [
                FFMPEG_EXE, '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list_file,
                '-c', 'copy',
                temp_processed
            ]
            subprocess.run(concat_cmd, capture_output=True, check=True)

            # Cleanup
            shutil.rmtree(original_frames_dir, ignore_errors=True)
            for seg_video in segment_videos:
                os.remove(seg_video)
            os.remove(concat_list_file)

        else:
            # Fall back to full video processing
            if not use_cropping and segments:
                print("[INFO]  Smart cropping not beneficial for this video, using full frame processing")

            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            self.update_state(
                state='PROCESSING',
                meta={'progress': 55, 'status': 'Running faster-propainter pipeline'}
            )

            try:
                # Use cached ProPainter pipeline (pre-loaded at worker startup)
                faster_propainter_pipeline = get_propainter_pipeline()

                import torch
                use_fp16 = torch.cuda.is_available()
                
                # Dynamic optimization based on video resolution
                dynamic_subvideo, _ = get_dynamic_subvideo_length(width, height)
                
                print(f"[RUNNING] Direct faster-propainter pipeline: resolution={width}x{height}, FP16={use_fp16}")
                print(f"   faster-propainter: neighbor_length=10 + ref_stride=15 + raft_iter=20 + subvideo_length=60 + flow_backend={PROPAINTER_FLOW_BACKEND}")

                # Direct pipeline call - OPTIMIZED FOR SPEED
                faster_propainter_pipeline(
                    video=video_path,                   # Input video
                    mask=mask_dir,                      # Mask directory
                    output=PROPAINTER_OUTPUT_ROOT,      # Output directory
                    resize_ratio=1.0,                   # Keep original resolution
                    mask_dilation=4,                    # faster-propainter: standard mask dilation
                    ref_stride=15,                      # faster-propainter: optimized for speed
                    neighbor_length=10,                 # faster-propainter: reduced for speed
                    subvideo_length=60,                 # faster-propainter: reduced for speed
                    raft_iter=20,                       # ProPainter default for quality
                    mode="video_inpainting",            # Standard inpainting
                    save_fps=fps,                       # Preserve original FPS
                    save_frames=False,                  # Only save video, not frames
                    fp16=use_fp16                       # FP16 if available
                )
                
                print("[OK] faster-propainter pipeline completed successfully")
                
            except Exception as e:
                print(f"[ERROR] faster-propainter pipeline failed: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"faster-propainter pipeline failed: {e}")
            finally:
                # Clear GPU memory after full video processing
                clear_gpu_memory()

            save_root = os.path.join(PROPAINTER_OUTPUT_ROOT, base_name)
            produced_video = os.path.join(save_root, 'inpaint_out.mp4')
            if not os.path.exists(produced_video):
                raise RuntimeError(f"ProPainter output not found: {produced_video}")

            temp_processed = os.path.join(RESULT_DIR, f"{base_name}_propainter_video.mp4")
            shutil.copy2(produced_video, temp_processed)

        self.update_state(
            state='PROCESSING',
            meta={'progress': 85, 'status': 'Merging audio'}
        )

        final_output = os.path.join(RESULT_DIR, f"{base_name}_propainter.mp4")

        try:
            check_audio_cmd = [
                FFPROBE_EXE,
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]

            has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
            has_audio = 'audio' in has_audio_check.stdout

            if has_audio:
                cmd = [
                    FFMPEG_EXE,
                    '-y',
                    '-i', temp_processed,
                    '-i', video_path,
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-c:v', 'copy',         # Don't re-encode, just copy video stream!
                    '-c:a', 'copy',         # Copy audio stream directly (instant, no re-encoding!)
                    '-shortest',
                    final_output
                ]
            else:
                cmd = [
                    FFMPEG_EXE,
                    '-y',
                    '-i', temp_processed,
                    '-c:v', 'copy',         # No audio, just copy video stream
                    final_output
                ]

            print(f"Running FFmpeg audio merge: {' '.join(cmd)}")
            audio_merge = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if audio_merge.returncode != 0:
                print("[WARNING]  Audio merge failed, returning video without audio")
                print(audio_merge.stderr)
                final_output = temp_processed
            else:
                if has_audio:
                    verify_cmd = [
                        FFPROBE_EXE,
                        '-v', 'error',
                        '-select_streams', 'a:0',
                        '-show_entries', 'stream=codec_type',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        final_output
                    ]
                    verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)
                    if 'audio' not in verify.stdout:
                        print("[WARNING]  Audio verification failed, keeping silent video")
                        final_output = temp_processed
                if final_output != temp_processed and os.path.exists(temp_processed):
                    os.remove(temp_processed)
        except FileNotFoundError:
            print("[WARNING]  ffmpeg/ffprobe not available, returning silent video")
            final_output = temp_processed
        except subprocess.TimeoutExpired:
            print("[WARNING]  FFmpeg timed out, returning silent video")
            final_output = temp_processed

        try:
            shutil.rmtree(mask_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            print(f"[WARNING]  Failed to delete mask directory {mask_dir}: {cleanup_exc}")

        # Note: Don't call self.update_state(state='SUCCESS') - it overrides the return value!
        # Celery automatically sets state to SUCCESS when the task returns normally

        # If running remotely and the API host is reachable via TUNNEL_URL,
        # upload the result back so the frontend can download from the API server.
        uploaded_path = None
        tunnel = os.getenv('TUNNEL_URL')
        if tunnel and os.getenv('UPLOAD_RESULT_BACK', '1') == '1':
            try:
                import requests
                upload_url = tunnel.rstrip('/') + '/api/upload-result'
                # Quick connectivity test (15 second timeout) - fail fast if server down
                requests.head(tunnel, timeout=15, headers={'ngrok-skip-browser-warning': 'true'})
                print(f"[UPLOAD]  Uploading result to API host: {upload_url}")
                with open(final_output, 'rb') as fp:
                    resp = requests.post(
                        upload_url,
                        headers={'ngrok-skip-browser-warning': 'true'},
                        files={'file': (os.path.basename(final_output), fp, 'video/mp4')},
                        timeout=60
                    )
                if resp.ok:
                    j = resp.json()
                    if j.get('status') == 'success' and j.get('result_url'):
                        uploaded_path = j['result_url']  # e.g. /results/<filename>
                        print(f"[OK] Uploaded result registered at: {uploaded_path}")
                else:
                    print(f"[WARNING]  Upload back failed: HTTP {resp.status_code}")
            except Exception as up_err:
                print(f"[WARNING]  Upload back error: {up_err}")

        return {
            'path': uploaded_path or final_output,
            'metadata': {
                'frames_processed': frames_processed,
                'frames_with_watermark': frames_with_watermark,
                'fps': fps,
                'width': width,
                'height': height,
                'propainter_output': final_output,
            }
        }

    except Exception as e:
        print(f"[ERROR] Error processing video: {e}")
        import traceback
        traceback.print_exc()
        raise



# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/ads.txt')
def ads_txt():
    """Serve ads.txt file for Google AdSense"""
    return send_file(os.path.join(app.static_folder, 'ads.txt'), mimetype='text/plain')


@app.route('/tunnel_url.txt')
def tunnel_url():
    """Serve tunnel URL for frontend auto-detection"""
    # Check environment variable first (for Railway deployment)
    env_url = os.getenv('TUNNEL_URL')
    if env_url:
        return env_url, 200, {'Content-Type': 'text/plain'}

    # Check file (for local development with localtunnel)
    tunnel_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
    if os.path.exists(tunnel_file):
        return send_file(tunnel_file, mimetype='text/plain')

    # Fallback to localhost
    return "http://localhost:9000", 200, {'Content-Type': 'text/plain'}


@app.route('/')
def index():
    """Serve landing page"""
    return send_file('web/index.html')


@app.route('/backgroundremover')
def backgroundremover_page():
    """Background remover UI"""
    return send_file('web/object-removal.html')


@app.route('/object-removal')
def object_removal_page():
    """How-to guide for object removal.

    This used to serve index.html, which made the URL a byte-for-byte duplicate
    of the homepage: it canonicalised to / and Google never indexed it. The
    homepage already targets the transactional query, so a second page chasing
    the same words would only compete with it. This one answers the
    informational query instead and links through to the tool.
    """
    return send_file('web/object-removal-guide.html')


@app.route('/video-tools')
def video_tools_page():
    """Serve video tools page"""
    return send_file('web/video-tools.html')


# ============================================================
# OBJECT REMOVAL API (Background Remover)
# - Videos uploaded to B2, metadata in Redis
# - SAM2 tracking via wsl_sam2_local queue
# - ProPainter inpainting via propainter_local queue
# ============================================================

@app.route('/api/object-removal/get-upload-url', methods=['POST', 'OPTIONS'])
def objrem_get_upload_url():
    """Get presigned B2 upload URL for direct client upload (no Railway ingress)"""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not B2_ENABLED:
        return jsonify({'status': 'error', 'message': 'B2 not configured'}), 500

    try:
        data = request.get_json() or {}
        filename = data.get('filename', 'video.mp4')

        # Validate extension
        ext = os.path.splitext(filename)[1].lower()
        allowed_exts = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}
        if ext not in allowed_exts:
            return jsonify({'status': 'error', 'message': f'File type {ext} not allowed'}), 400

        # Generate unique job ID
        job_id = uuid.uuid4().hex[:12]
        remote_path = f"objrem/{job_id}{ext}"

        # Get B2 upload URL (same pattern as /api/get-upload-url)
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET)

        upload_url, upload_auth_token = b2_api.account_info.take_bucket_upload_url(bucket.id_)

        if upload_url is None:
            api_url = b2_api.account_info.get_api_url()
            auth_token = b2_api.account_info.get_account_auth_token()
            response = b2_api.raw_api.get_upload_url(api_url, auth_token, bucket.id_)
            upload_url = response['uploadUrl']
            upload_auth_token = response['authorizationToken']

        print(f"[OBJREM] Generated upload URL for {remote_path}")

        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'upload_url': upload_url,
            'auth_token': upload_auth_token,
            'remote_path': remote_path,
            'cdn_url': f"{B2_CDN_URL}/{remote_path}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/upload-complete', methods=['POST', 'OPTIONS'])
def objrem_upload_complete():
    """Client notifies B2 upload complete, provides video dimensions"""
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        data = request.get_json() or {}
        job_id = data.get('job_id')
        cdn_url = data.get('cdn_url')
        width = data.get('width', 0)
        height = data.get('height', 0)
        fps = data.get('fps', 30)
        duration = data.get('duration', 0)
        frame_count = data.get('frame_count', 0) or int(duration * fps)

        if not job_id or not cdn_url:
            return jsonify({'status': 'error', 'message': 'Missing job_id or cdn_url'}), 400

        # Store metadata in Redis
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        redis_client.hset(f"objrem:{job_id}", mapping={
            'status': 'uploaded',
            'cdn_url': cdn_url,
            'width': str(width),
            'height': str(height),
            'fps': str(fps),
            'frame_count': str(frame_count),
            'duration': str(duration),
            'points': '[]',
            'frame_index': '0'
        })
        redis_client.expire(f"objrem:{job_id}", 86400 * 7)  # 7 days

        print(f"[OBJREM] Upload complete job {job_id}: {width}x{height} @ {fps}fps, {frame_count} frames")

        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'video_url': cdn_url,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/video/<job_id>')
def objrem_video(job_id):
    """Stream video - redirect to B2 CDN or serve local file"""
    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        job = redis_client.hgetall(f"objrem:{job_id}")

        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        cdn_url = job.get('cdn_url', '')

        # If CDN URL is a full URL (B2), redirect to it
        if cdn_url.startswith('http'):
            return redirect(cdn_url)

        # Otherwise serve local file
        local_path = job.get('local_path', '')
        if local_path and os.path.exists(local_path):
            return send_file(local_path, mimetype='video/mp4')

        return jsonify({'status': 'error', 'message': 'Video not found'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/select', methods=['POST', 'OPTIONS'])
def objrem_select():
    """Store clicked point for SAM2 tracking"""
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON payload'}), 400
        job_id = data.get('job_id')
        points = data.get('points', [])
        frame_index = data.get('frame_index', 0)

        if not job_id:
            return jsonify({'status': 'error', 'message': 'Missing job_id'}), 400

        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        job = redis_client.hgetall(f"objrem:{job_id}")

        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        # Store points for tracking
        redis_client.hset(f"objrem:{job_id}", mapping={
            'points': json.dumps(points),
            'frame_index': str(frame_index)
        })

        print(f"[OBJREM-SELECT] Job {job_id}: {len(points)} points at frame {frame_index}")

        return jsonify({
            'status': 'success',
            'message': 'Point selected - click "Remove Selected" to track',
            'points': points,
            'frame_index': frame_index,
            'width': int(job.get('width', 0)),
            'height': int(job.get('height', 0))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/auto-detect', methods=['POST', 'OPTIONS'])
def objrem_auto_detect():
    """Use YOLO to detect objects in first frame - dispatches to wsl_yolo_local"""
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON payload'}), 400
        job_id = data.get('job_id')

        if not job_id:
            return jsonify({'status': 'error', 'message': 'Missing job_id'}), 400

        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        job = redis_client.hgetall(f"objrem:{job_id}")

        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        # Dispatch YOLO detection task
        from celery import signature
        video_url = job.get('cdn_url', '')

        task_id = f"yolo_{job_id}"
        s1 = signature(
            'yolo.detect_objects',
            args=[video_url, 0],  # video_url, frame_index
            queue='wsl_yolo_local'
        )
        result = s1.apply_async(task_id=task_id)

        redis_client.hset(f"objrem:{job_id}", 'yolo_task_id', task_id)

        print(f"[OBJREM-YOLO] Submitted task {task_id} for job {job_id}")

        return jsonify({
            'status': 'success',
            'message': 'YOLO detection started',
            'task_id': task_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/track', methods=['POST', 'OPTIONS'])
def objrem_track():
    """Start full video tracking with SAM2 via wsl_sam2_local queue"""
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON payload'}), 400
        job_id = data.get('job_id')

        if not job_id:
            return jsonify({'status': 'error', 'message': 'Missing job_id'}), 400

        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        job = redis_client.hgetall(f"objrem:{job_id}")

        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        points_str = job.get('points', '[]')
        points = json.loads(points_str)

        if not points:
            return jsonify({'status': 'error', 'message': 'No points selected'}), 400

        # Update status
        redis_client.hset(f"objrem:{job_id}", mapping={
            'status': 'tracking',
            'progress': '0',
            'message': 'Submitting to SAM2 worker...'
        })

        # Get video URL (B2 CDN or local path)
        video_url = job.get('cdn_url', '')
        if not video_url.startswith('http'):
            video_url = job.get('local_path', '')

        frame_idx = int(job.get('frame_index', 0))

        # Convert points to format expected by SAM2 worker
        point_coords = [(p['x'], p['y']) for p in points]
        labels = [p.get('label', 1) for p in points]
        # Extract object_ids - each click is a separate object
        object_ids = [p.get('object_id', 0) for p in points]

        # Check for user-modified masks (from eraser tool)
        modified_masks = data.get('modified_masks', [])

        # WSL-native temp path for masks
        wsl_masks_dir = f"/tmp/sam2_masks/{job_id}"

        # Submit to wsl_sam2_local queue
        from celery import signature
        task_id = f"objrem_{job_id}"

        s1 = signature(
            'sam2.generate_masks_fullfps',
            args=[video_url, wsl_masks_dir],
            kwargs={
                'prompt_mode': 'point',
                'points': point_coords,
                'labels': labels,
                'object_ids': object_ids,  # Pass object_ids to SAM2
                'frame_idx': frame_idx,
                'api_base': None,
                'modified_masks': modified_masks
            },
            queue='wsl_sam2'
        )
        result = s1.apply_async(task_id=task_id)

        # Store user_id and estimated credits for deduction on completion
        # Get user_id from request body first (frontend sends it), fallback to session
        user_id = data.get('user_id') or session.get('user_id')
        estimated_credits = float(data.get('estimated_credits', 0.5))

        redis_client.hset(f"objrem:{job_id}", mapping={
            'celery_task_id': task_id,
            'wsl_masks_dir': wsl_masks_dir,
            'message': 'Tracking in progress...',
            'user_id': user_id or '',
            'estimated_credits': str(estimated_credits)
        })

        # Store credits in format expected by deduct_credit_on_completion()
        if user_id:
            redis_client.set(f"task:{task_id}:user_id", user_id)
            redis_client.set(f"task:{task_id}:credits", str(max(0.1, round(float(estimated_credits), 1))))  # decimal credits
            redis_client.expire(f"task:{task_id}:user_id", 86400 * 7)  # 7 days
            redis_client.expire(f"task:{task_id}:credits", 86400 * 7)

        print(f"[OBJREM-TRACK] Submitted task {task_id} for job {job_id}")
        print(f"[OBJREM-TRACK] user_id from request: {data.get('user_id')}, from session: {session.get('user_id')}, final: {user_id}")
        print(f"[OBJREM-TRACK] Points: {point_coords}, Object IDs: {object_ids}, Frame: {frame_idx}")

        return jsonify({
            'status': 'success',
            'message': 'Tracking started via Celery',
            'job_id': job_id,
            'task_id': task_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/status/<job_id>', methods=['GET', 'OPTIONS'])
def objrem_status(job_id):
    """Get job status from Redis/Celery"""
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        job = redis_client.hgetall(f"objrem:{job_id}")

        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        status = job.get('status', 'unknown')
        task_id = job.get('celery_task_id')

        # Check Celery task status if we have a task_id
        if task_id and status in ['tracking', 'inpainting']:
            try:
                result = celery.AsyncResult(task_id)
                if result.ready():
                    if result.successful():
                        task_result = result.result
                        if isinstance(task_result, dict):
                            # Update job with result
                            if task_result.get('masks_dir'):
                                redis_client.hset(f"objrem:{job_id}", 'masks_dir', task_result['masks_dir'])
                            # Check for masks_url (from SAM2 tracking)
                            if task_result.get('masks_url'):
                                redis_client.hset(f"objrem:{job_id}", 'masks_url', task_result['masks_url'])
                                status = 'completed'
                            # Check for cdn_url (from simple effects export)
                            if task_result.get('cdn_url'):
                                redis_client.hset(f"objrem:{job_id}", 'result_cdn_url', task_result['cdn_url'])
                                status = 'export_complete'
                            redis_client.hset(f"objrem:{job_id}", 'status', status)
                    else:
                        status = 'error'
                        redis_client.hset(f"objrem:{job_id}", mapping={
                            'status': 'error',
                            'message': str(result.result)
                        })
                else:
                    # Still running - get progress from task meta
                    meta = result.info
                    if isinstance(meta, dict):
                        progress = meta.get('progress', 0)
                        message = meta.get('message', '')
                        if progress:
                            redis_client.hset(f"objrem:{job_id}", 'progress', str(progress))
                        if message:
                            redis_client.hset(f"objrem:{job_id}", 'message', message)
            except Exception as e:
                print(f"[OBJREM-STATUS] Celery check error: {e}")

        # Refresh job data
        job = redis_client.hgetall(f"objrem:{job_id}")

        # Deduct credits when export completes (only once)
        # Use same deduct_credit_on_completion function as backgroundremover
        # Export endpoint already stores task:{task_id}:user_id and task:{task_id}:credits
        new_credits = None
        if job.get('status') == 'export_complete' and not job.get('credits_deducted'):
            task_id = job.get('celery_task_id')
            if task_id:
                new_credits = deduct_credit_on_completion(task_id)
                if new_credits is not None:
                    redis_client.hset(f"objrem:{job_id}", 'credits_deducted', '1')
                    print(f"[OBJREM-CREDITS] Credits deducted via deduct_credit_on_completion for task {task_id}")

        response = {
            'status': job.get('status', 'unknown'),
            'progress': int(job.get('progress', 0)),
            'message': job.get('message', ''),
            'masks_dir': job.get('masks_dir', ''),
            'cdn_url': job.get('result_cdn_url', ''),
            'width': int(job.get('width', 0)),
            'height': int(job.get('height', 0))
        }

        if new_credits is not None:
            response['new_credits'] = new_credits

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/export', methods=['POST', 'OPTIONS'])
def objrem_export():
    """Apply simple effects (blur, greenscreen, color) and export video"""
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON payload'}), 400
        job_id = data.get('job_id')
        operation = data.get('operation', 'keep_object')  # keep_object, remove_object, blur_inside, blur_outside
        background = data.get('background', 'color')  # transparent, color, blur
        bg_color = data.get('bg_color', '#00FF00')  # hex color
        blur_amount = data.get('blur_amount', 25)
        dilation = data.get('dilation', 4)
        output_format = data.get('format', 'mp4')

        if not job_id:
            return jsonify({'status': 'error', 'message': 'Missing job_id'}), 400

        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        job = redis_client.hgetall(f"objrem:{job_id}")

        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        # Check if tracking completed successfully
        job_status = job.get('status', '')
        if job_status == 'error':
            error_msg = job.get('message', 'Tracking failed')
            return jsonify({'status': 'error', 'message': f'Tracking failed: {error_msg}'}), 400

        if job_status not in ('completed', 'inpainting', 'processing'):
            return jsonify({'status': 'error', 'message': f'Tracking not complete. Status: {job_status}'}), 400

        masks_url = job.get('masks_url')
        masks_dir_local = job.get('masks_dir')

        # Accept either masks_url (B2) OR masks_dir (local) - both are valid
        if not masks_url and not masks_dir_local:
            return jsonify({'status': 'error', 'message': 'No masks available - run tracking first'}), 400

        # Update status
        redis_client.hset(f"objrem:{job_id}", mapping={
            'status': 'inpainting',
            'progress': '0',
            'message': 'Applying effects...'
        })

        # Get video URL
        video_url = job.get('cdn_url', '')
        if not video_url.startswith('http'):
            video_url = job.get('video_url', '')

        # Submit to WSL SAM2 worker (simple effects, no ProPainter!)
        from celery import signature
        task_id = f"simplefx_{job_id}"

        video_path_local = job.get('video_path')
        print(f"[OBJREM-EXPORT] masks_dir from Redis: {masks_dir_local}")
        print(f"[OBJREM-EXPORT] video_path from Redis: {video_path_local}")

        s2 = signature(
            'sam2.apply_simple_effects',
            args=[video_url, masks_url],
            kwargs={
                'operation': operation,
                'background': background,
                'bg_color': bg_color,
                'blur_amount': blur_amount,
                'dilation': dilation,
                'output_format': output_format,
                'masks_dir_local': masks_dir_local,
                'video_path_local': video_path_local
            },
            queue='wsl_sam2'
        )
        result = s2.apply_async(task_id=task_id)

        redis_client.hset(f"objrem:{job_id}", 'celery_task_id', task_id)

        # Store credits in format expected by deduct_credit_on_completion()
        # Get user_id and credits from the job data (stored during tracking)
        user_id = job.get('user_id') or session.get('user_id')
        estimated_credits = float(job.get('estimated_credits', 0.5))
        if user_id:
            redis_client.set(f"task:{task_id}:user_id", user_id)
            redis_client.set(f"task:{task_id}:credits", str(max(0.1, round(float(estimated_credits), 1))))  # decimal credits
            redis_client.expire(f"task:{task_id}:user_id", 86400 * 7)  # 7 days
            redis_client.expire(f"task:{task_id}:credits", 86400 * 7)

        print(f"[OBJREM-EXPORT] Submitted simple effects task {task_id}")
        print(f"[OBJREM-EXPORT] user_id from job: '{job.get('user_id')}', from session: {session.get('user_id')}, final: {user_id}")
        print(f"[OBJREM-EXPORT] Operation: {operation}, Background: {background}, Color: {bg_color}")

        return jsonify({
            'status': 'success',
            'message': 'Processing started',
            'job_id': job_id,
            'task_id': task_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/object-removal/preview', methods=['POST', 'OPTIONS'])
def objrem_preview():
    """Preview not available in production (would need to download video/masks from B2)"""
    if request.method == 'OPTIONS':
        return ('', 204)
    return jsonify({'status': 'error', 'message': 'Preview not available in production mode. Use Export to generate final video.'}), 501


@app.route('/api/object-removal/download/<job_id>')
def objrem_download(job_id):
    """Download result - redirect to B2 CDN or serve local file"""
    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        job = redis_client.hgetall(f"objrem:{job_id}")

        if not job:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        result_url = job.get('result_cdn_url', '')

        if result_url.startswith('http'):
            return redirect(result_url)

        # Try to find local result file
        result_path = job.get('result_path', '')
        if result_path and os.path.exists(result_path):
            return send_file(result_path, as_attachment=True, download_name=f"result_{job_id}.mp4")

        return jsonify({'status': 'error', 'message': 'Result not found'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/login')
def login_page():
    """Serve login page"""
    return send_file('web/login.html')


@app.route('/reset-password')
def reset_password_page():
    """Serve password reset page"""
    return send_file('web/reset-password.html')





@app.route('/api/remove-watermark', methods=['POST'])
def remove_watermark():
    """
    Submit image for watermark removal

    Returns:
        {
            'task_id': str,
            'status': 'queued'
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read file
        image_data = file.read()

        # Calculate hash for caching/deduplication
        file_hash = hashlib.md5(image_data).hexdigest()

        # Determine file type
        is_video = file.filename.lower().endswith(('.mp4', '.mov', '.avi'))

        if is_video:
            # Save video temporarily (D drive only!)
            temp_filename = f"{file_hash}{os.path.splitext(file.filename)[1]}"
            temp_path = os.path.join(UPLOAD_DIR, temp_filename)

            with open(temp_path, 'wb') as f:
                f.write(image_data)

            # Queue video processing
            task = process_video_task.apply_async(args=[temp_path])

            return jsonify({
                'task_id': task.id,
                'status': 'queued',
                'file_type': 'video'
            })

        # Process image with YOLO + ProPainter
        # Save image temporarily
        temp_filename = f"{file_hash}.png"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_path, 'wb') as f:
            f.write(image_data)

        # Queue image processing with ProPainter
        task = process_image_task.apply_async(args=[temp_path])

        return jsonify({
            'task_id': task.id,
            'status': 'queued',
            'file_type': 'image'
        })

    except Exception as e:
        print(f"[ERROR] Error queuing task: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<task_id>', methods=['GET', 'OPTIONS'])
def get_status(task_id):
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Check task status

    Returns:
        {
            'state': 'PENDING' | 'STARTED' | 'PROCESSING' | 'SUCCESS' | 'FAILURE',
            'progress': str,
            'result': { 'result_url': str } (if SUCCESS)
        }
    """
    try:
        # Special handling for distributed tasks (manual Redis tracking)
        if task_id.startswith('distributed_'):
            video_id = task_id.replace('distributed_', '')
            tracking_key = f"segments:{video_id}"

            # Check if tracking exists
            total_bytes = celery.backend.get(f"{tracking_key}:total")
            if not total_bytes:
                return jsonify({
                    'state': 'PENDING',
                    'progress': 'Task is waiting in queue...',
                    'info': {'progress': 0, 'status': 'Waiting in queue...'}
                })

            # Get completion status
            completed_bytes = celery.backend.get(tracking_key) or b'0'
            total = int(total_bytes)
            completed = int(completed_bytes)

            print(f"[INFO] Distributed task status - {task_id}: {completed}/{total} segments complete")

            if completed < total:
                # Still processing segments
                progress = int((completed / total) * 90)  # 0-90%
                return jsonify({
                    'state': 'PROCESSING',
                    'progress': f'Processing segments: {completed}/{total}',
                    'info': {'progress': progress, 'status': f'Segment {completed}/{total} complete'}
                })
            else:
                # All segments done - check if finalization/encoding complete
                try:
                    redis_client = celery.backend.client
                    encoding_status = redis_client.get(f"video:{video_id}:status")

                    if encoding_status and encoding_status.decode() == "complete":
                        # Encoding is DONE! Get the final path
                        final_path_raw = redis_client.get(f"video:{video_id}:final_path")
                        if final_path_raw:
                            final_path = final_path_raw.decode()
                            # Check if path is CDN URL, web path, or local path
                            if final_path.startswith('http'):
                                result_url = final_path  # CDN URL - return as-is
                            elif final_path.startswith('/results/'):
                                result_url = final_path
                            else:
                                filename = os.path.basename(final_path)
                                result_url = f'/results/{filename}'

                            print(f"[STATUS] ✅ Encoding complete for {video_id}! Returning: {result_url}")
                            # Deduct credit on successful completion
                            new_credits = deduct_credit_on_completion(task_id)
                            response_data = {
                                'state': 'SUCCESS',
                                'result': {'result_url': result_url},
                                'metadata': {'total_segments': total}
                            }
                            if new_credits is not None:
                                response_data['new_credits'] = new_credits
                            return jsonify(response_data)

                    # Encoding still in progress
                    print(f"[STATUS] Segments complete ({completed}/{total}), encoding in progress for {video_id}")
                    return jsonify({
                        'state': 'PROCESSING',
                        'progress': 'Finalizing video (encoding in progress)...',
                        'info': {'progress': 95, 'status': 'Video encoding in background (almost done!)'}
                    })

                except Exception as e:
                    print(f"[ERROR] Failed to check finalization status: {e}")
                    # Fallback - keep showing encoding in progress
                    return jsonify({
                        'state': 'PROCESSING',
                        'progress': 'Finalizing video...',
                        'info': {'progress': 95, 'status': 'Background encoding'}
                    })

        # Regular Celery task handling
        from celery.result import AsyncResult

        task = AsyncResult(task_id, app=celery)

        # Always log for debugging stuck "Waiting in queue" issue
        print(f"[INFO] Status check - Task: {task_id}, State: {task.state}, Info: {task.info}")

        response = {
            'state': task.state
        }

        if task.state == 'PENDING':
            response['progress'] = 'Task is waiting in queue...'
            response['info'] = {'progress': 0, 'status': 'Waiting in queue...'}

            # Get queue position and ETA
            try:
                from celery.task.control import inspect
                i = inspect(app=celery)
                scheduled = i.scheduled() or {}
                reserved = i.reserved() or {}
                active = i.active() or {}

                # Count total pending tasks and find position
                all_pending = []
                for worker_tasks in scheduled.values():
                    all_pending.extend(worker_tasks)
                for worker_tasks in reserved.values():
                    all_pending.extend(worker_tasks)

                # Find this task's position (1-indexed)
                queue_position = 1
                for idx, t in enumerate(all_pending):
                    if t.get('id') == task_id:
                        queue_position = idx + 1
                        break
                else:
                    # Task not found in pending - it's next in line
                    queue_position = 1

                total_in_queue = len(all_pending)
                active_count = sum(len(tasks) for tasks in active.values())

                # Get average processing time for ETA
                avg_time = 120  # Default 2 minutes
                try:
                    redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                    times = redis_client.lrange('processing_times:recent', 0, 19)
                    if times:
                        times_float = [float(t) for t in times]
                        avg_time = sum(times_float) / len(times_float)
                except:
                    pass

                # Calculate ETA: (position / active_workers) * avg_time
                workers = max(1, active_count) if active_count > 0 else 1
                estimated_wait = int((queue_position / workers) * avg_time)

                response['queue'] = {
                    'position': queue_position,
                    'total': max(total_in_queue, 1),
                    'active_workers': active_count,
                    'estimated_wait_seconds': estimated_wait
                }
            except Exception as e:
                print(f"[WARN] Could not get queue position: {e}")
        elif task.state == 'STARTED':
            info = task.info or {}
            response['progress'] = info.get('status', 'Starting...')
            response['info'] = {'progress': info.get('progress', 5), 'status': info.get('status', 'Starting...')}
        elif task.state == 'PROCESSING':
            info = task.info or {}
            response['progress'] = info.get('status', 'Processing...')
            response['info'] = {'progress': info.get('progress', 50), 'status': info.get('status', 'Processing...')}
        elif task.state == 'SUCCESS':
            # task.result is a dict with 'path' and 'metadata' OR 'chord_id' for distributed tasks
            result_data = task.result
            print(f"[POLL] DEBUG - task.result type: {type(result_data)}, value: {result_data}")

            if isinstance(result_data, dict):
                # Check if this is a finalize task with background encoding
                if result_data.get('status') == 'background_encoding':
                    video_id = result_data.get('video_id')
                    print(f"[POLL] Background encoding in progress for video {video_id}, checking Redis...")

                    # Check if encoding is complete in Redis
                    try:
                        redis_client = celery.backend.client
                        encoding_status = redis_client.get(f"video:{video_id}:status")

                        if encoding_status and encoding_status.decode() == "complete":
                            # Encoding is done! Get the final path
                            final_path_raw = redis_client.get(f"video:{video_id}:final_path")
                            if final_path_raw:
                                final_path = final_path_raw.decode()
                                # Check if path is CDN URL, web path, or local path
                                if final_path.startswith('http'):
                                    result_url = final_path  # CDN URL - return as-is
                                elif final_path.startswith('/results/'):
                                    result_url = final_path
                                else:
                                    # Local path - extract filename
                                    filename = os.path.basename(final_path)
                                    result_url = f'/results/{filename}'
                                print(f"[POLL] Background encoding COMPLETE for {video_id}! Returning result_url: {result_url}")
                                # Deduct credit on successful completion
                                new_credits = deduct_credit_on_completion(task_id)
                                response['result'] = {
                                    'result_url': result_url
                                }
                                if 'metadata' in result_data:
                                    response['metadata'] = result_data['metadata']
                                if new_credits is not None:
                                    response['new_credits'] = new_credits
                                return jsonify(response)  # Return SUCCESS with result_url
                        else:
                            # Encoding still in progress - keep frontend polling
                            print(f"[POLL] Background encoding not complete yet for {video_id}, status: {encoding_status}")
                            response['state'] = 'PROCESSING'
                            response['progress'] = result_data.get('message', 'Encoding in progress...')
                            response['info'] = {
                                'progress': 95,
                                'status': 'Video encoding in background (almost done!)'
                            }
                            return jsonify(response)  # Keep polling
                    except Exception as e:
                        print(f"[ERROR] Failed to check Redis for background encoding status: {e}")
                        # Fall through to normal error handling
                        response['state'] = 'PROCESSING'
                        response['progress'] = 'Encoding in progress...'
                        response['info'] = {'progress': 95, 'status': 'Background encoding'}
                        return jsonify(response)

                # Check if this is a prepare task that returned chord_id
                if 'chord_id' in result_data:
                    chord_id = result_data['chord_id']
                    print(f"[RETRY] Prepare task complete, switching to chord tracking: {chord_id}")
                    # Tell frontend to switch to tracking the chord instead
                    # Frontend expects HTTP 409 status code to detect task switching
                    response['state'] = 'PROCESSING'
                    response['progress'] = result_data.get('message', 'Processing segments in parallel')
                    response['info'] = {
                        'progress': 50,
                        'status': 'Segments processing in parallel'
                    }
                    response['stale'] = True  # Frontend recognizes this pattern
                    response['current_task_id'] = chord_id  # Frontend will switch to this task_id
                    return jsonify(response), 409  # HTTP 409 Conflict - task superseded

                # Check if this task forwarded to another task (YOLO forwarding pattern)
                if 'task_id' in result_data and result_data.get('status') == 'processing':
                    new_task_id = result_data['task_id']
                    print(f"[POLL] Task forwarded to: {new_task_id}")
                    response['state'] = 'PROCESSING'
                    response['progress'] = result_data.get('message', 'Processing...')
                    response['info'] = {'progress': 30, 'status': result_data.get('message', 'Processing...')}
                    response['stale'] = True
                    response['current_task_id'] = new_task_id
                    return jsonify(response), 409

                result_path = result_data.get('path')
                if not result_path:
                    print(f"[ERROR] Task {task_id} SUCCESS but no path in result: {result_data}")
                    return jsonify({'error': 'Invalid result format - missing path'}), 500
            else:
                result_path = result_data

            # Final safety check for None result_path
            if not result_path:
                print(f"[ERROR] Task {task_id} has None result_path")
                return jsonify({'error': 'Invalid result - path is None'}), 500

            filename = os.path.basename(result_path)
            response['result'] = {
                'result_url': f'/results/{filename}'
            }
            if isinstance(result_data, dict) and 'metadata' in result_data:
                response['metadata'] = result_data['metadata']
            # Deduct credit on successful completion
            new_credits = deduct_credit_on_completion(task_id)
            if new_credits is not None:
                response['new_credits'] = new_credits
        elif task.state == 'FAILURE':
            response['error'] = str(task.info)
            print(f"[ERROR] Task failed: {task.info}")

        return jsonify(response)
    except Exception as e:
        print(f"[ERROR] Status endpoint error for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500


@app.route('/api/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """
    Download processed file

    Returns:
        Processed image/video file
    """
    from celery.result import AsyncResult

    task = AsyncResult(task_id, app=celery)

    if task.state != 'SUCCESS':
        return jsonify({
            'error': 'Task not complete',
            'status': task.state
        }), 400

    try:
        result = task.result

        if 'data' in result:
            # Image result
            result_bytes = result['data']
            return send_file(
                io.BytesIO(result_bytes),
                mimetype='image/png',
                as_attachment=True,
                download_name='watermark_removed.png'
            )
        elif 'path' in result:
            # Video result
            return send_file(
                result['path'],
                mimetype='video/mp4',
                as_attachment=True,
                download_name='watermark_removed.mp4'
            )
        else:
            return jsonify({'error': 'Invalid result format'}), 500

    except Exception as e:
        print(f"[ERROR] Error retrieving result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-from-url', methods=['POST', 'OPTIONS'])
@require_auth
def download_from_url():
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Download video from URL - tries direct download first,
    falls back to Playwright if available (for protected sites)

    Request: { "url": "https://..." }
    Response: { "status": "success", "task_id": "...", "video_url": "/uploads/..." }
    """
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

        # Normalize URL
        url = url.strip()

        # Validate URL format
        if url.startswith('/') or not url.startswith('http'):
            return jsonify({'status': 'error', 'message': 'URL must start with http:// or https://'}), 400

        print(f"📋 Downloading from URL: {url}")

        # Validate URL to prevent SSRF attacks
        if not validate_url(url):
            return jsonify({'status': 'error', 'message': 'Invalid or unsafe URL'}), 400

        # Generate unique filename
        task_id = str(uuid.uuid4())

        # Determine file extension from URL
        import re
        ext_match = re.search(r'\.(mp4|mov|avi|webm|mkv)(\?|$)', url.lower())
        file_ext = ext_match.group(1) if ext_match else 'mp4'
        output_path = os.path.join(UPLOAD_DIR, f'{task_id}.{file_ext}')

        # ============================================
        # Method 1: Direct download with requests
        # ============================================
        import requests as req_lib
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'video/*,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': url,
        }

        try:
            print(f"[DOWNLOAD] Trying direct download...")
            response = req_lib.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)
            response.raise_for_status()

            # Check if response is actually a video
            content_type = response.headers.get('content-type', '').lower()
            is_video = (
                'video' in content_type or
                'octet-stream' in content_type or
                url.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv'))
            )

            if is_video:
                # Download the video
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[OK] Video downloaded: {output_path} ({file_size:.1f}MB)")

                # Upload to B2 immediately
                remote_path = f"uploads/{task_id}.{file_ext}"
                cdn_url = upload_to_b2(output_path, remote_path)
                if not cdn_url:
                    return jsonify({'error': 'B2 upload failed'}), 500

                # Store in Redis
                redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                redis_client.setex(f"upload:{task_id}:cdn_url", 86400, cdn_url)
                redis_client.setex(f"upload:{task_id}:remote_path", 86400, remote_path)

                # Delete local temp file
                os.remove(output_path)
                print(f"[B2] Uploaded to CDN: {cdn_url}")

                return jsonify({
                    'status': 'success',
                    'task_id': task_id,
                    'video_url': cdn_url
                })
            else:
                print(f"[WARNING] Response is not a video (content-type: {content_type})")

        except req_lib.exceptions.RequestException as e:
            print(f"[WARNING] Direct download failed: {e}")

        # ============================================
        # Method 2: Playwright browser (if available)
        # ============================================
        try:
            from playwright.sync_api import sync_playwright
            import time
            import html

            print("[RUNNING] Trying Playwright browser extraction...")

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled']
                )

                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )

                page = context.new_page()
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

                print(f"🌐 Navigating to: {url}")
                page.goto(url, wait_until='domcontentloaded', timeout=30000)
                time.sleep(3)

                # Find video URLs in page
                video_urls = []
                content = page.content()
                content_video_urls = re.findall(r'https?://[^\s"\'<>]+\.(mp4|mov|webm)[^\s"\'<>]*', content)
                video_urls.extend([u[0] if isinstance(u, tuple) else u for u in content_video_urls])

                # Try video element
                if not video_urls:
                    try:
                        video_src = page.locator('video').first.get_attribute('src')
                        if video_src:
                            video_urls.append(video_src)
                    except:
                        pass

                browser.close()

                if video_urls:
                    video_src = html.unescape(video_urls[0])
                    print(f"[OK] Found video URL: {video_src}")

                    # Download the extracted video
                    response = req_lib.get(video_src, headers=headers, stream=True, timeout=300)
                    response.raise_for_status()

                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=65536):
                            f.write(chunk)

                    # Upload to B2 immediately
                    remote_path = f"uploads/{task_id}.{file_ext}"
                    cdn_url = upload_to_b2(output_path, remote_path)
                    if not cdn_url:
                        return jsonify({'error': 'B2 upload failed'}), 500

                    # Store in Redis
                    redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                    redis_client.setex(f"upload:{task_id}:cdn_url", 86400, cdn_url)
                    redis_client.setex(f"upload:{task_id}:remote_path", 86400, remote_path)

                    # Delete local temp file
                    os.remove(output_path)
                    print(f"[B2] Uploaded to CDN: {cdn_url}")

                    return jsonify({
                        'status': 'success',
                        'task_id': task_id,
                        'video_url': cdn_url
                    })

        except ImportError:
            print("[INFO] Playwright not installed - browser extraction unavailable")
        except Exception as e:
            print(f"[WARNING] Playwright extraction failed: {e}")

        # ============================================
        # All methods failed
        # ============================================
        return jsonify({
            'status': 'error',
            'message': 'Could not download video. Please use a direct video URL (ending in .mp4, .mov, .webm) or upload the file directly.'
        }), 400

    except Exception as e:
        print(f"[ERROR] Download error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/download-external', methods=['POST', 'OPTIONS'])
def download_external():
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Download video from external URL using Playwright bypass + cookies
    Bypasses Cloudflare protection using saved cookies

    Request: { "url": "https://example.com/video..." }
    Response: { "status": "success", "task_id": "...", "video_url": "/uploads/..." }
    """
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

        # Normalize URL
        url = url.strip()

        # Require full URL
        if url.startswith('/') or not url.startswith('http'):
            return jsonify({'status': 'error', 'message': 'URL must start with http:// or https://'}), 400

        print(f"📋 Downloading from external URL: {url}")

        # Validate URL to prevent SSRF attacks
        if not validate_url(url):
            return jsonify({'status': 'error', 'message': 'Invalid or unsafe URL'}), 400

        # Generate unique filename
        task_id = str(uuid.uuid4())
        output_path = os.path.join(UPLOAD_DIR, f'{task_id}.mp4')

        # Import Playwright
        from playwright.sync_api import sync_playwright
        import time
        import json

        # Path to cookies file
        cookies_file = os.path.join(SCRIPT_DIR, 'downz', 'cookies.json')
        print(f"[POLL] Looking for cookies at: {cookies_file}")
        print(f"[POLL] SCRIPT_DIR is: {SCRIPT_DIR}")

        with sync_playwright() as p:
            print("[RUNNING] Launching browser for external download...")
            browser = p.chromium.launch(
                headless=True,  # Run headless in production
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )

            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York'
            )

            # Load cookies if they exist
            if os.path.exists(cookies_file):
                print(f"📂 Loading cookies from {cookies_file}...")
                with open(cookies_file, 'r') as f:
                    cookies = json.load(f)
                    context.add_cookies(cookies)
                print("[OK] Cookies loaded!")
            else:
                print(f"[WARNING]  No cookies found at: {cookies_file}")
                browser.close()
                return jsonify({
                    'status': 'error',
                    'message': 'Authentication required. Please contact administrator to set up cookies.',
                    'hint': 'Some external videos require login cookies.'
                }), 401

            page = context.new_page()

            # Hide webdriver
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            print(f"🌐 Navigating to: {url}")
            page.goto(url, wait_until='domcontentloaded', timeout=30000)

            time.sleep(2)

            print("[POLL] Extracting video URL from page content...")

            video_src = None

            # Try to find video URLs in page content
            content = page.content()
            import re
            video_urls = re.findall(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content)

            if video_urls:
                import html
                video_src = html.unescape(video_urls[0])  # Decode &amp; to &
                print(f"[OK] Found video URL in page content: {video_src}")
            else:
                # Fallback: try video element
                print("⏳ No .mp4 URL found, trying video element...")
                video_element = page.query_selector('video')
                if video_element:
                    video_src = video_element.get_attribute('src')
                    if not video_src:
                        source = page.query_selector('video source')
                        if source:
                            video_src = source.get_attribute('src')

                if not video_src:
                    # Take screenshot for debugging
                    screenshot_path = os.path.join(TEMP_DIR, f'debug_{task_id}.png')
                    page.screenshot(path=screenshot_path)
                    print(f"📸 Screenshot saved to {screenshot_path}")

                    browser.close()
                    return jsonify({
                        'status': 'error',
                        'message': f'No video URL found. Screenshot saved to {screenshot_path}'
                    }), 404

            if video_src:
                print(f"[OK] Found video source: {video_src}")

                # Make absolute URL if relative
                if video_src.startswith('//'):
                    video_src = 'https:' + video_src
                elif video_src.startswith('/'):
                    from urllib.parse import urljoin
                    video_src = urljoin(url, video_src)

                print(f"[DOWNLOAD]  Downloading video...")

                # Download using Playwright's request context with retry logic
                max_retries = 3
                retry_delay = 3

                for retry in range(max_retries):
                    try:
                        if retry > 0:
                            print(f"[RETRY] Retry attempt {retry}/{max_retries-1}...")
                            time.sleep(retry_delay)
                        else:
                            # Small delay before first attempt to avoid rate limiting
                            time.sleep(1)

                        response = page.request.get(video_src)

                        if response.ok:
                            with open(output_path, 'wb') as f:
                                f.write(response.body())

                            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                            print(f"[OK] Downloaded successfully! Size: {file_size:.2f} MB")

                            # Upload to B2 immediately
                            remote_path = f"uploads/{task_id}.mp4"
                            cdn_url = upload_to_b2(output_path, remote_path)
                            if not cdn_url:
                                browser.close()
                                return jsonify({'error': 'B2 upload failed'}), 500

                            # Store in Redis
                            redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                            redis_client.setex(f"upload:{task_id}:cdn_url", 86400, cdn_url)
                            redis_client.setex(f"upload:{task_id}:remote_path", 86400, remote_path)

                            # Delete local temp file
                            os.remove(output_path)
                            print(f"[B2] Uploaded to CDN: {cdn_url}")

                            browser.close()

                            return jsonify({
                                'status': 'success',
                                'task_id': task_id,
                                'video_url': cdn_url
                            })
                        else:
                            print(f"[WARNING]  Download failed: HTTP {response.status}")
                            if retry == max_retries - 1:
                                browser.close()
                                return jsonify({
                                    'status': 'error',
                                    'message': f'Download failed after {max_retries} attempts: HTTP {response.status}'
                                }), 500
                    except Exception as download_error:
                        print(f"[WARNING]  Download error: {download_error}")
                        if retry == max_retries - 1:
                            browser.close()
                            return jsonify({
                                'status': 'error',
                                'message': f'Download failed after {max_retries} attempts: {str(download_error)}'
                            }), 500
            else:
                browser.close()
                return jsonify({
                    'status': 'error',
                    'message': 'Could not find video source URL'
                }), 404

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def check_rate_limit(ip):
    """Check if IP has exceeded rate limit (10 uploads per hour)"""
    current_time = time.time()

    if ip in UPLOAD_RATE_LIMIT:
        count, first_upload_time = UPLOAD_RATE_LIMIT[ip]

        # Reset if hour has passed
        if current_time - first_upload_time > 3600:
            UPLOAD_RATE_LIMIT[ip] = (1, current_time)
            return True

        # Check limit
        if count >= 10:
            return False

        UPLOAD_RATE_LIMIT[ip] = (count + 1, first_upload_time)
        return True
    else:
        UPLOAD_RATE_LIMIT[ip] = (1, current_time)
        return True


def validate_video_limits(video_path):
    """
    Check video duration and FPS against processing limits.
    Returns (is_valid, error_message)
    """
    try:
        import subprocess
        import json as json_module

        probe_cmd = [
            FFPROBE_EXE, '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration',
            '-of', 'json',
            video_path
        ]

        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        probe_data = json_module.loads(probe_result.stdout)

        if not probe_data.get('streams'):
            return True, None  # Not a video or can't read, let it pass

        stream = probe_data['streams'][0]

        # Extract FPS
        fps_str = stream.get('r_frame_rate', '30/1')
        fps_num, fps_den = map(int, fps_str.split('/'))
        fps = fps_num / fps_den if fps_den != 0 else 30

        # Extract duration
        duration = float(stream.get('duration', 0))

        # Check FPS limit
        if fps > MAX_VIDEO_FPS:
            return False, f"Video exceeds maximum FPS of {MAX_VIDEO_FPS} (yours: {int(fps)}fps)"

        # Check duration limit
        if duration > MAX_VIDEO_DURATION_SECONDS:
            return False, f"Video exceeds maximum duration of {MAX_VIDEO_DURATION_SECONDS} seconds (yours: {int(duration)}s)"

        return True, None

    except Exception as e:
        # If we can't validate, let it through (fail open for edge cases)
        print(f"⚠️ Video validation error: {e}")
        return True, None


def is_user_admin(user_id):
    """Check if user has admin privileges."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT is_admin FROM users WHERE id = %s', (user_id,))
                result = cur.fetchone()
                return result and result[0] is True
    except Exception:
        return False


def check_user_video_rate_limit(user_id, is_admin=False):
    """Check if user has exceeded rate limit (5 videos per minute). Admins bypass."""
    if is_admin:
        return True

    current_time = time.time()
    window = 60  # 1 minute window
    max_requests = 5

    if user_id not in USER_VIDEO_RATE_LIMIT:
        USER_VIDEO_RATE_LIMIT[user_id] = []

    # Remove timestamps older than window
    USER_VIDEO_RATE_LIMIT[user_id] = [
        ts for ts in USER_VIDEO_RATE_LIMIT[user_id]
        if current_time - ts < window
    ]

    if len(USER_VIDEO_RATE_LIMIT[user_id]) >= max_requests:
        return False

    USER_VIDEO_RATE_LIMIT[user_id].append(current_time)
    return True


# ============================================================================
# DIRECT B2 UPLOAD (Zero Railway ingress)
# ============================================================================

@app.route('/api/get-upload-url', methods=['POST', 'OPTIONS'])
@require_auth
def get_upload_url():
    """Get presigned B2 upload URL for direct client upload (zero Railway ingress)"""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not B2_ENABLED:
        return jsonify({'error': 'Direct upload not available', 'fallback': True}), 200

    try:
        data = request.get_json() or {}
        filename = data.get('filename', 'video.mp4')
        content_type = data.get('content_type', 'video/mp4')

        # Validate extension
        ext = os.path.splitext(filename)[1].lower()
        allowed_exts = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.png', '.jpg', '.jpeg', '.gif'}
        if ext not in allowed_exts:
            return jsonify({'error': f'File type {ext} not allowed'}), 400

        # Generate unique task ID
        task_id = str(uuid.uuid4())
        remote_path = f"uploads/{task_id}{ext}"

        # Get B2 upload URL
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET)

        # Get upload URL from pool or request fresh one (CORRECT b2sdk v2 API)
        upload_url, upload_auth_token = b2_api.account_info.take_bucket_upload_url(bucket.id_)

        if upload_url is None:
            # Pool is empty, get fresh upload URL with all 3 required params
            api_url = b2_api.account_info.get_api_url()
            auth_token = b2_api.account_info.get_account_auth_token()
            response = b2_api.raw_api.get_upload_url(api_url, auth_token, bucket.id_)
            upload_url = response['uploadUrl']
            upload_auth_token = response['authorizationToken']

        print(f"[B2-DIRECT] Generated upload URL for {remote_path}")

        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'upload_url': upload_url,
            'auth_token': upload_auth_token,
            'remote_path': remote_path,
            'cdn_url': f"{B2_CDN_URL}/{remote_path}"
        })

    except Exception as e:
        print(f"[B2-DIRECT] Error getting upload URL: {e}")
        return jsonify({'error': str(e), 'fallback': True}), 200


@app.route('/api/upload-complete', methods=['POST', 'OPTIONS'])
@require_auth
def upload_complete():
    """Client notifies that direct B2 upload is complete"""
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        data = request.get_json() or {}
        task_id = data.get('task_id')
        remote_path = data.get('remote_path')
        cdn_url = data.get('cdn_url')
        filename = data.get('filename')

        if not task_id or not cdn_url:
            return jsonify({'error': 'Missing task_id or cdn_url'}), 400

        # Store CDN URL in Redis
        try:
            redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
            redis_client.setex(f"upload:{task_id}:cdn_url", 86400, cdn_url)
            redis_client.setex(f"upload:{task_id}:remote_path", 86400, remote_path)
            print(f"[B2-DIRECT] Upload complete: {task_id} -> {cdn_url}")
        except Exception as e:
            print(f"[B2-DIRECT] Redis store failed: {e}")

        # Get file extension
        ext = os.path.splitext(filename or remote_path)[1].lower() if filename or remote_path else '.mp4'

        # Trigger sprite sheet generation in background for instant scrubbing
        try:
            from concurrent.futures import ThreadPoolExecutor
            sprite_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SpriteGen")
            sprite_executor.submit(generate_sprite_sheet, cdn_url, task_id)
            print(f"[B2-DIRECT] Sprite generation triggered for {task_id}")
        except Exception as sprite_err:
            print(f"[B2-DIRECT] Sprite generation trigger failed: {sprite_err}")

        # Trigger HEVC transcode pipeline in background (for iPhone videos)
        # This runs async and stores preview URL in Redis when ready
        try:
            hevc_transcode_executor.submit(transcode_hevc_pipeline, cdn_url, task_id)
            print(f"[HEVC] Transcode pipeline triggered for {task_id}")
        except Exception as hevc_err:
            print(f"[HEVC] Transcode trigger failed: {hevc_err}")

        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'video_url': cdn_url,
            'sprite_url': f'/api/sprite/{task_id}/sprite.jpg',
            'vtt_url': f'/api/sprite/{task_id}/thumbs.vtt',
            'message': 'Upload complete, ready for processing'
        })

    except Exception as e:
        print(f"[B2-DIRECT] Error in upload-complete: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/<task_id>', methods=['GET', 'OPTIONS'])
def get_preview(task_id):
    """
    Get preview URL for video (transcoded H.264 for HEVC videos, original for others)
    Frontend polls this after upload to get browser-compatible video URL.
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)

        # Check transcode status
        status = redis_client.get(f"preview_status:{task_id}")

        if status == "ready":
            # Transcode complete - return H.264 preview URL
            url = redis_client.get(f"preview_url:{task_id}")
            return jsonify({
                'status': 'ready',
                'transcoded': True,
                'url': url
            })
        elif status == "original":
            # Not HEVC or transcode failed - return original URL
            url = redis_client.get(f"preview_url:{task_id}")
            if not url:
                # Fallback: get from upload storage
                url = redis_client.get(f"upload:{task_id}:cdn_url")
            return jsonify({
                'status': 'ready',
                'transcoded': False,
                'url': url
            })
        elif status in ["checking", "transcoding"]:
            # Still processing
            return jsonify({
                'status': 'processing',
                'stage': status
            })
        else:
            # No status yet - transcode hasn't started or task doesn't exist
            # Return original URL if available
            url = redis_client.get(f"upload:{task_id}:cdn_url")
            if url:
                return jsonify({
                    'status': 'ready',
                    'transcoded': False,
                    'url': url
                })
            return jsonify({
                'status': 'not_found'
            }), 404

    except Exception as e:
        print(f"[HEVC] Error getting preview: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
@require_auth
def upload_file():
    """DEPRECATED: Use /api/get-upload-url for direct B2 uploads"""
    if request.method == 'OPTIONS':
        return ('', 204)

    return jsonify({
        'status': 'error',
        'message': 'Direct upload disabled. Use /api/get-upload-url for client-side B2 upload.',
        'migration_endpoint': '/api/get-upload-url'
    }), 410




@app.route('/api/extract-frames/<task_id>', methods=['GET', 'OPTIONS'])
def extract_frames(task_id):
    """
    Extract frame thumbnails from uploaded video for timeline UI

    Returns: { "status": "success", "frames": [{frame_number, timestamp, thumbnail_url}], "total_frames", "fps", "duration" }
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        # Find the uploaded video
        video_path = None
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
            if os.path.exists(test_path):
                video_path = test_path
                break

        if not video_path:
            return jsonify({'status': 'error', 'message': 'Video not found'}), 404

        # Check if frames are already cached
        cache_key = f'frames_{task_id}'
        with FRAME_CACHE_LOCK:
            if cache_key in FRAME_CACHE:
                return jsonify(FRAME_CACHE[cache_key])

        # Get video metadata using ffprobe
        probe_cmd = [
            FFPROBE_EXE, '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
            '-of', 'json',
            video_path
        ]

        import subprocess
        import json as json_module
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        probe_data = json_module.loads(probe_result.stdout)

        # Extract metadata
        stream = probe_data['streams'][0]
        fps_str = stream.get('r_frame_rate', '30/1')
        fps_num, fps_den = map(int, fps_str.split('/'))
        fps = fps_num / fps_den if fps_den != 0 else 30

        duration = float(stream.get('duration', 0))
        total_frames = int(stream.get('nb_frames', int(duration * fps)))

        # Extract thumbnails at 1-second intervals (or max 60 thumbnails for long videos)
        max_thumbnails = 60
        interval_seconds = max(1, int(duration / max_thumbnails))

        # Create thumbnails directory for this task
        thumbnails_dir = os.path.join(CACHE_DIR, 'thumbnails', task_id)
        os.makedirs(thumbnails_dir, exist_ok=True)

        frames_data = []

        # Extract thumbnails using ffmpeg
        for i in range(0, int(duration), interval_seconds):
            thumbnail_path = os.path.join(thumbnails_dir, f'frame_{i:04d}.jpg')

            # Skip if already exists
            if not os.path.exists(thumbnail_path):
                ffmpeg_cmd = [
                    FFMPEG_EXE, '-y',
                    '-ss', str(i),  # Seek to timestamp
                    '-i', video_path,
                    '-vframes', '1',  # Extract 1 frame
                    '-vf', 'scale=120:-1',  # Scale to 120px width (maintain aspect ratio)
                    '-q:v', '5',  # Quality (lower = better, 2-5 is good)
                    thumbnail_path
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True, timeout=5)

            if os.path.exists(thumbnail_path):
                frame_number = int(i * fps)
                frames_data.append({
                    'frame_number': frame_number,
                    'timestamp': i,
                    'thumbnail_url': f'/api/thumbnail/{task_id}/frame_{i:04d}.jpg'
                })

        result = {
            'status': 'success',
            'frames': frames_data,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        }

        # Cache the result
        with FRAME_CACHE_LOCK:
            FRAME_CACHE[cache_key] = result

        return jsonify(result)

    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'Frame extraction timeout'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/thumbnail/<task_id>/<filename>', methods=['GET'])
def serve_thumbnail(task_id, filename):
    """
    Serve thumbnail images for timeline
    """
    try:
        thumbnails_dir = os.path.join(CACHE_DIR, 'thumbnails', task_id)
        thumbnail_path = os.path.join(thumbnails_dir, filename)

        if not os.path.exists(thumbnail_path):
            return jsonify({'status': 'error', 'message': 'Thumbnail not found'}), 404

        from flask import send_file
        return send_file(thumbnail_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# SPRITE SHEET GENERATION FOR INSTANT SCRUBBING
# ============================================================================

def format_vtt_time(seconds):
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def generate_sprite_sheet(video_url, task_id, thumb_width=160, thumb_height=90, interval=1.0, cols=10):
    """
    Generate thumbnail sprite sheet + VTT file for instant scrubbing.

    Args:
        video_url: CDN URL or local path to video
        task_id: Unique task identifier
        thumb_width: Width of each thumbnail (default 160)
        thumb_height: Height of each thumbnail (default 90)
        interval: Seconds between thumbnails (default 1.0)
        cols: Number of columns in sprite grid (default 10)

    Returns:
        dict with sprite_path, vtt_path, duration, or None on error
    """
    # The Railway web container ships without ffmpeg on purpose: the GPU workers
    # do the video work. Without this guard the call below runs subprocess.run
    # with FFPROBE_EXE as None and raises a TypeError on every single upload,
    # which is what filled the logs with tracebacks. Scrubbing thumbnails are a
    # nicety, so skip them quietly instead. Same shape as check_video_codec.
    if not FFPROBE_EXE or not FFMPEG_EXE:
        return None

    try:
        sprite_dir = os.path.join(CACHE_DIR, 'sprites', task_id)
        os.makedirs(sprite_dir, exist_ok=True)

        # Get video duration using ffprobe
        probe_cmd = [
            FFPROBE_EXE, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_url
        ]

        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if probe_result.returncode != 0:
            print(f"[SPRITE] ffprobe failed for {task_id}: {probe_result.stderr}")
            return None

        import json as json_module
        probe_data = json_module.loads(probe_result.stdout)
        duration = float(probe_data['format']['duration'])

        # Calculate grid dimensions
        num_thumbs = int(duration / interval) + 1
        rows = (num_thumbs + cols - 1) // cols

        # FFmpeg: Extract frames and tile into sprite sheet
        sprite_path = os.path.join(sprite_dir, 'sprite.jpg')

        # Build filter: fps -> scale -> tile
        vf_filter = f"fps=1/{interval},scale={thumb_width}:{thumb_height},tile={cols}x{rows}"

        ffmpeg_cmd = [
            FFMPEG_EXE, '-y',
            '-i', video_url,
            '-vf', vf_filter,
            '-frames:v', '1',
            '-q:v', '5',  # JPEG quality (2-5 is good)
            sprite_path
        ]

        print(f"[SPRITE] Generating sprite for {task_id}: {num_thumbs} thumbs ({cols}x{rows} grid)")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"[SPRITE] FFmpeg failed for {task_id}: {result.stderr}")
            return None

        # Generate VTT file
        vtt_path = os.path.join(sprite_dir, 'thumbs.vtt')
        with open(vtt_path, 'w') as f:
            f.write('WEBVTT\n\n')
            for i in range(num_thumbs):
                start = i * interval
                end = min((i + 1) * interval, duration)
                x = (i % cols) * thumb_width
                y = (i // cols) * thumb_height
                f.write(f'{format_vtt_time(start)} --> {format_vtt_time(end)}\n')
                f.write(f'sprite.jpg#xywh={x},{y},{thumb_width},{thumb_height}\n\n')

        # Also save metadata for frontend
        meta_path = os.path.join(sprite_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            import json as json_module
            json_module.dump({
                'duration': duration,
                'num_thumbs': num_thumbs,
                'thumb_width': thumb_width,
                'thumb_height': thumb_height,
                'cols': cols,
                'rows': rows,
                'interval': interval
            }, f)

        print(f"[SPRITE] Generated sprite for {task_id}: {sprite_path}")
        return {
            'sprite_path': sprite_path,
            'vtt_path': vtt_path,
            'duration': duration,
            'num_thumbs': num_thumbs
        }

    except Exception as e:
        print(f"[SPRITE] Error generating sprite for {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/api/sprite/<task_id>/<filename>')
def serve_sprite(task_id, filename):
    """Serve sprite sheet or VTT file"""
    try:
        # Sanitize inputs
        safe_task_id = secure_filename(task_id)
        safe_filename = secure_filename(filename)

        sprite_dir = os.path.join(CACHE_DIR, 'sprites', safe_task_id)
        file_path = os.path.join(sprite_dir, safe_filename)

        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        # Determine MIME type
        if safe_filename.endswith('.jpg') or safe_filename.endswith('.jpeg'):
            mimetype = 'image/jpeg'
        elif safe_filename.endswith('.webp'):
            mimetype = 'image/webp'
        elif safe_filename.endswith('.vtt'):
            mimetype = 'text/vtt'
        elif safe_filename.endswith('.json'):
            mimetype = 'application/json'
        else:
            mimetype = 'application/octet-stream'

        return send_from_directory(sprite_dir, safe_filename, mimetype=mimetype)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sprite-status/<task_id>')
def sprite_status(task_id):
    """Check if sprite is ready for a given task"""
    try:
        safe_task_id = secure_filename(task_id)
        sprite_dir = os.path.join(CACHE_DIR, 'sprites', safe_task_id)

        sprite_path = os.path.join(sprite_dir, 'sprite.jpg')
        vtt_path = os.path.join(sprite_dir, 'thumbs.vtt')
        meta_path = os.path.join(sprite_dir, 'meta.json')

        ready = os.path.exists(sprite_path) and os.path.exists(vtt_path)

        response = {'ready': ready}

        # Include metadata if available
        if ready and os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                import json as json_module
                response['meta'] = json_module.load(f)

        return jsonify(response)

    except Exception as e:
        return jsonify({'ready': False, 'error': str(e)}), 500


@app.route('/api/generate-sprite/<task_id>', methods=['POST'])
def trigger_sprite_generation(task_id):
    """
    Manually trigger sprite generation for a task.
    Expects JSON body with 'video_url' (CDN URL).
    """
    try:
        data = request.get_json() or {}
        video_url = data.get('video_url')

        if not video_url:
            # Try to get from Redis
            try:
                redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                video_url = redis_client.get(f"upload:{task_id}:cdn_url")
            except:
                pass

        if not video_url:
            return jsonify({'error': 'No video_url provided and not found in cache'}), 400

        # Check if already exists
        safe_task_id = secure_filename(task_id)
        sprite_path = os.path.join(CACHE_DIR, 'sprites', safe_task_id, 'sprite.jpg')
        if os.path.exists(sprite_path):
            return jsonify({'status': 'exists', 'message': 'Sprite already generated'})

        # Generate in background thread
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(generate_sprite_sheet, video_url, task_id)

        return jsonify({
            'status': 'started',
            'message': 'Sprite generation started in background'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST', 'OPTIONS'])
@require_auth
@require_credits(min_credits=1)
def process_video():
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Process video to remove watermarks

    Request: { "task_id": "uuid-from-download" }
    Response: { "status": "success", "task_id": "celery-task-id" }
    """
    try:
        # Check user video rate limit (5 per minute, admins bypass)
        user_id = session.get('user_id')
        if not check_user_video_rate_limit(user_id, is_user_admin(user_id)):
            return jsonify({
                'status': 'error',
                'message': 'Rate limit exceeded. Maximum 5 videos per minute.',
                'retry_after': 60
            }), 429

        data = request.get_json()
        task_id = data.get('task_id')

        if not task_id:
            return jsonify({'status': 'error', 'message': 'No task_id provided'}), 400

        # Find uploaded media (video first, then image) with any extension
        video_path = None
        # Try common video extensions
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
            if os.path.exists(test_path):
                video_path = test_path
                break
        # If not found, try common image extensions
        if not video_path:
            for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
                if os.path.exists(test_path):
                    video_path = test_path
                    break

        if not video_path:
            return jsonify({'status': 'error', 'message': 'Media not found'}), 404

        # Validate video limits (duration and FPS) for video files only
        ext = os.path.splitext(video_path)[1].lower()
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if ext not in image_exts:
            is_valid, error_msg = validate_video_limits(video_path)
            if not is_valid:
                return jsonify({'status': 'error', 'message': error_msg}), 400

        # Queue processing task via Celery and return the real task id
        print(f"📤 Queuing processing task for: {video_path}")

        try:
            # Decide pipeline based on extension
            ext = os.path.splitext(video_path)[1].lower()
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            if ext in image_exts:
                result = celery.send_task('watermark.remove_image', args=[video_path])
            else:
                # Use WSL sender/receiver architecture for YOLO mode
                # Chain: WSL YOLO detection → Windows ProPainter
                from celery import signature, chain
                import uuid

                def _current_public_base():
                    env_url = os.getenv('TUNNEL_URL')
                    if env_url:
                        return env_url.strip()
                    try:
                        tunnel_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
                        if os.path.exists(tunnel_file):
                            with open(tunnel_file, 'r') as f:
                                return f.read().strip()
                    except Exception:
                        pass
                    return 'http://localhost:9000'

                base = _current_public_base()
                video_id = task_id or str(uuid.uuid4())[:8]

                # Create masks directory
                masks_dir = f"/tmp/{video_id}_yolo_masks"

                # Chain: WSL YOLO detection → Windows ProPainter inpainting
                # yolo.generate_masks runs in WSL (queue=wsl_yolo)
                # watermark._continue_after_masks runs in Windows (queue=propainter)
                print(f"[YOLO] Using WSL sender/receiver chain")
                print(f"  - Video: {video_path}")
                print(f"  - Masks dir: {masks_dir}")
                print(f"  - Queue flow: wsl_yolo → propainter")

                s1 = signature('yolo.generate_masks', args=[video_path, masks_dir], kwargs={'api_base': base}, queue='wsl_yolo')
                s2 = signature('watermark._continue_after_masks', args=[video_path, video_id, None, None, None, 0, base], queue='propainter')

                result = chain(s1, s2).apply_async()
            print(f"[OK] Task queued with ID: {result.id}")

            # Store user_id and estimated credits for deduction on completion
            user_id = session.get('user_id')
            estimated_credits = data.get('estimated_credits', 1)
            if user_id:
                try:
                    redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                    redis_client.setex(f"task:{result.id}:user_id", 86400 * 7, str(user_id))
                    redis_client.setex(f"task:{result.id}:credits", 86400 * 7, str(estimated_credits))
                    print(f"[CREDITS] Stored user {user_id}, credits {estimated_credits} for task {result.id}")
                except Exception as e:
                    print(f"[CREDITS] Failed to store: {e}")

            return jsonify({'status': 'success', 'task_id': result.id})

        except Exception as e:
            print(f"[ERROR] Failed to queue task: {e}")
            import traceback; traceback.print_exc()
            return jsonify({'status': 'error', 'message': f'Failed to connect to Redis: {str(e)}'}), 500

    except Exception as e:
        print(f"[ERROR] Process endpoint error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Redirect to Cloudflare CDN - NO local serving"""
    filename = sanitize_filename(filename)
    task_id = os.path.splitext(filename)[0]

    # Check Redis for CDN URL
    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        cdn_url = redis_client.get(f"upload:{task_id}:cdn_url")
        if cdn_url:
            print(f"[CDN] Redirecting upload {filename} to: {cdn_url}")
            return redirect(cdn_url)
    except Exception as e:
        print(f"[CDN] Redis lookup failed: {e}")

    # No fallback to local files - strict B2-only
    return jsonify({
        'error': 'File not found in CDN',
        'message': 'This file was not uploaded to B2 storage'
    }), 404


@app.route('/cool.mp4')
def serve_cool_video():
    """Redirect to B2 CDN for showcase video (zero Railway egress)"""
    return redirect(f"{B2_CDN_URL}/static/static/cool.mp4")


@app.route('/s2.mp4')
def serve_s2_video():
    """Redirect to B2 CDN for s2 before video (zero Railway egress)"""
    return redirect(f"{B2_CDN_URL}/static/static/s2.mp4")


@app.route('/s2removed.mp4')
def serve_s2removed_video():
    """Redirect to B2 CDN for s2removed after video (zero Railway egress)"""
    return redirect(f"{B2_CDN_URL}/static/static/s2removed.mp4")


@app.route('/training/<filename>')
def serve_training_video(filename):
    """Redirect to B2 CDN for training videos (zero Railway egress)"""
    # Sanitize filename to prevent path traversal
    filename = sanitize_filename(filename)

    # Only allow .mp4 files
    if not filename.endswith('.mp4'):
        return jsonify({'error': 'Invalid file type'}), 400

    print(f"[TRAINING-VIDEO] Redirecting {filename} to B2 CDN")
    return redirect(f"{B2_CDN_URL}/static/training/{filename}")


@app.route('/admin/list-videos', methods=['GET'])
def admin_list_videos():
    """Admin endpoint to list videos in Railway volume"""
    admin_secret = os.getenv('ADMIN_SECRET', 'dev-secret-123')

    if request.args.get('secret') != admin_secret:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        static_videos = []
        training_videos = []

        if os.path.exists(STATIC_VIDEOS_DIR):
            static_videos = os.listdir(STATIC_VIDEOS_DIR)

        if os.path.exists(TRAINING_VIDEOS_DIR):
            training_videos = os.listdir(TRAINING_VIDEOS_DIR)

        return jsonify({
            'static_videos_dir': STATIC_VIDEOS_DIR,
            'static_videos': static_videos,
            'training_videos_dir': TRAINING_VIDEOS_DIR,
            'training_videos': training_videos,
            'is_railway': IS_RAILWAY,
            'data_dir': DATA_DIR if IS_RAILWAY else 'N/A'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/migrate-static-to-b2', methods=['POST'])
def admin_migrate_static_to_b2():
    """One-time migration: Upload all static videos from Railway volume to B2 CDN"""
    admin_secret = os.getenv('ADMIN_SECRET', 'dev-secret-123')

    if request.headers.get('X-Admin-Secret') != admin_secret:
        return jsonify({'error': 'Unauthorized'}), 401

    if not B2_ENABLED:
        return jsonify({'error': 'B2 not enabled'}), 400

    results = {'migrated': [], 'failed': [], 'skipped': []}

    # Migrate static videos
    if os.path.exists(STATIC_VIDEOS_DIR):
        for filename in os.listdir(STATIC_VIDEOS_DIR):
            if filename.endswith('.mp4'):
                local_path = os.path.join(STATIC_VIDEOS_DIR, filename)
                remote_path = f"static/static/{filename}"
                try:
                    cdn_url = upload_to_b2(local_path, remote_path)
                    results['migrated'].append({'file': filename, 'cdn_url': cdn_url})
                    print(f"[MIGRATE] ✅ {filename} -> {cdn_url}")
                except Exception as e:
                    results['failed'].append({'file': filename, 'error': str(e)})
                    print(f"[MIGRATE] ❌ {filename}: {e}")

    # Migrate training videos
    if os.path.exists(TRAINING_VIDEOS_DIR):
        for filename in os.listdir(TRAINING_VIDEOS_DIR):
            if filename.endswith('.mp4'):
                local_path = os.path.join(TRAINING_VIDEOS_DIR, filename)
                remote_path = f"static/training/{filename}"
                try:
                    cdn_url = upload_to_b2(local_path, remote_path)
                    results['migrated'].append({'file': filename, 'cdn_url': cdn_url})
                    print(f"[MIGRATE] ✅ {filename} -> {cdn_url}")
                except Exception as e:
                    results['failed'].append({'file': filename, 'error': str(e)})
                    print(f"[MIGRATE] ❌ {filename}: {e}")

    # Migrate demo videos
    demo_dir = os.path.join(DATA_DIR, 'demo_videos') if IS_RAILWAY else 'demo_videos'
    if os.path.exists(demo_dir):
        for filename in os.listdir(demo_dir):
            if filename.endswith('.mp4'):
                local_path = os.path.join(demo_dir, filename)
                remote_path = f"demo_videos/{filename}"
                try:
                    cdn_url = upload_to_b2(local_path, remote_path)
                    results['migrated'].append({'file': filename, 'cdn_url': cdn_url})
                    print(f"[MIGRATE] ✅ {filename} -> {cdn_url}")
                except Exception as e:
                    results['failed'].append({'file': filename, 'error': str(e)})
                    print(f"[MIGRATE] ❌ {filename}: {e}")

    return jsonify(results)


@app.route('/admin/migrate-web-assets-to-b2', methods=['POST'])
def admin_migrate_web_assets_to_b2():
    """Migrate web static assets (CSS, JS, images) to B2 CDN - CRITICAL for zero egress"""
    admin_secret = os.getenv('ADMIN_SECRET', 'dev-secret-123')

    if request.headers.get('X-Admin-Secret') != admin_secret:
        return jsonify({'error': 'Unauthorized'}), 401

    if not B2_ENABLED:
        return jsonify({'error': 'B2 not enabled'}), 400

    results = {'migrated': [], 'failed': [], 'skipped': []}
    web_folder = app.static_folder

    # Extensions to upload
    static_extensions = ('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot')

    # Walk through web folder and upload all static assets
    for root, dirs, files in os.walk(web_folder):
        for filename in files:
            if filename.endswith(static_extensions):
                local_path = os.path.join(root, filename)
                # Get relative path from web folder
                rel_path = os.path.relpath(local_path, web_folder)
                remote_path = f"static/{rel_path}".replace('\\', '/')

                try:
                    cdn_url = upload_to_b2(local_path, remote_path)
                    file_size = os.path.getsize(local_path)
                    results['migrated'].append({
                        'file': rel_path,
                        'cdn_url': cdn_url,
                        'size_kb': round(file_size / 1024, 1)
                    })
                    print(f"[MIGRATE-WEB] ✅ {rel_path} ({file_size/1024:.1f}KB) -> {cdn_url}")
                except Exception as e:
                    results['failed'].append({'file': rel_path, 'error': str(e)})
                    print(f"[MIGRATE-WEB] ❌ {rel_path}: {e}")

    total_size = sum(item.get('size_kb', 0) for item in results['migrated'])
    results['total_size_kb'] = total_size
    results['message'] = f"Migrated {len(results['migrated'])} files ({total_size:.1f}KB total). These will now be served from CDN!"

    return jsonify(results)


@app.route('/admin/upload-video', methods=['POST'])
def admin_upload_video():
    """Admin endpoint to upload videos to Railway volume"""
    # Simple secret key auth - replace 'your-admin-secret' with a real secret
    admin_secret = os.getenv('ADMIN_SECRET', 'dev-secret-123')

    if request.headers.get('X-Admin-Secret') != admin_secret:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_type = request.form.get('type', 'static')  # 'static' or 'training'

    if video.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Sanitize filename
    filename = sanitize_filename(video.filename)

    # Only allow .mp4 files
    if not filename.endswith('.mp4'):
        return jsonify({'error': 'Only .mp4 files allowed'}), 400

    # Determine destination directory
    if video_type == 'training':
        dest_dir = TRAINING_VIDEOS_DIR
    else:
        dest_dir = STATIC_VIDEOS_DIR

    # Create directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Save file locally first
    file_path = os.path.join(dest_dir, filename)
    video.save(file_path)

    # Upload to B2 CDN to avoid Railway egress costs
    cdn_url = None
    if B2_ENABLED:
        try:
            remote_path = f"static/{video_type}/{filename}"
            cdn_url = upload_to_b2(file_path, remote_path)
            print(f"[ADMIN-UPLOAD] ✅ Uploaded to B2: {cdn_url}")
            # Clean up local file after B2 upload
            os.remove(file_path)
            print(f"[ADMIN-UPLOAD] Cleaned up local file: {file_path}")
        except Exception as b2_err:
            print(f"[ADMIN-UPLOAD] ⚠️ B2 upload failed: {b2_err}, keeping local file")

    return jsonify({
        'success': True,
        'filename': filename,
        'type': video_type,
        'path': file_path,
        'url': cdn_url or (f'/training/{filename}' if video_type == 'training' else f'/{filename}'),
        'cdn_url': cdn_url
    })


@app.route('/results/<filename>')
def serve_result(filename):
    """Redirect to Cloudflare CDN - NO local serving"""
    filename = sanitize_filename(filename)
    base_name = os.path.splitext(filename)[0]
    task_id = base_name.replace('_processed', '').replace('_cleaned', '')

    # Check Redis for CDN URL
    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        cdn_url = redis_client.get(f"video:{task_id}:final_path")
        if cdn_url and cdn_url.startswith('http'):
            print(f"[CDN] Redirecting result {filename} to: {cdn_url}")
            return redirect(cdn_url)
    except Exception as e:
        print(f"[CDN] Redis lookup failed: {e}")

    # No fallback to local files - strict B2-only
    return jsonify({
        'error': 'Result not ready or not found',
        'message': 'Processing may still be in progress'
    }), 404


@app.route('/demo_videos/<filename>')
def serve_demo_video(filename):
    """Redirect to B2 CDN for demo videos (zero Railway egress)"""
    # Sanitize filename
    filename = sanitize_filename(filename)
    print(f"[DEMO-VIDEO] Redirecting {filename} to B2 CDN")
    return redirect(f"{B2_CDN_URL}/demo_videos/{filename}")


@app.route('/api/upload-result', methods=['POST', 'OPTIONS'])
def upload_result():
    if request.method == 'OPTIONS':
        return ('', 204)
    """Accept result file from a remote worker and store it under results/.

    Request: multipart/form-data with field 'file' and optional 'filename'.
    Response: { "status": "success", "result_url": "/results/<filename>" }
    """
    try:
        print(f"[UPLOAD-RESULT] Received upload request from {request.remote_addr}")

        if 'file' not in request.files:
            print(f"[UPLOAD-RESULT ERROR] No file in request")
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        up = request.files['file']
        if up.filename == '':
            print(f"[UPLOAD-RESULT ERROR] Empty filename")
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        req_filename = request.form.get('filename') or up.filename
        safe_name = sanitize_filename(req_filename)

        print(f"[UPLOAD-RESULT] Saving file: {safe_name} to {RESULT_DIR}")
        os.makedirs(RESULT_DIR, exist_ok=True)
        dest = os.path.join(RESULT_DIR, safe_name)

        up.save(dest)
        file_size_mb = os.path.getsize(dest) / (1024 * 1024)

        print(f"[UPLOAD-RESULT] ✅ Saved {safe_name} ({file_size_mb:.2f} MB) to {dest}")
        print(f"[UPLOAD-RESULT] File exists check: {os.path.exists(dest)}")
        print(f"[UPLOAD-RESULT] RESULT_DIR contents: {os.listdir(RESULT_DIR) if os.path.exists(RESULT_DIR) else 'DIR NOT FOUND'}")

        # Upload to B2 CDN to avoid Railway egress costs
        cdn_url = None
        if B2_ENABLED:
            try:
                remote_path = f"results/{safe_name}"
                cdn_url = upload_to_b2(dest, remote_path)
                print(f"[UPLOAD-RESULT] ✅ Uploaded to B2: {cdn_url}")
                # Store CDN URL in Redis for serving
                task_id = safe_name.split('_')[0] if '_' in safe_name else safe_name.rsplit('.', 1)[0]
                redis_client.setex(f"video:{task_id}:final_path", 86400, cdn_url)
                # Clean up local file after B2 upload
                os.remove(dest)
                print(f"[UPLOAD-RESULT] Cleaned up local file: {dest}")
            except Exception as b2_err:
                print(f"[UPLOAD-RESULT] ⚠️ B2 upload failed: {b2_err}, keeping local file")

        return jsonify({'status': 'success', 'result_url': cdn_url or f'/results/{safe_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/notify-result', methods=['POST', 'OPTIONS'])
def notify_result():
    """Worker notifies server of completed CDN URL (no file upload needed)"""
    if request.method == 'OPTIONS':
        return ('', 204)
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON payload'}), 400
        video_id = data.get('video_id')
        cdn_url = data.get('cdn_url')

        if not video_id or not cdn_url:
            return jsonify({'status': 'error', 'message': 'Missing video_id or cdn_url'}), 400

        print(f"[NOTIFY-RESULT] Received CDN URL for video {video_id}: {cdn_url}")

        redis_client = celery.backend.client
        redis_client.set(f"video:{video_id}:final_path", cdn_url)
        redis_client.set(f"video:{video_id}:status", "complete")

        print(f"[NOTIFY-RESULT] ✅ Stored CDN URL in Redis for video {video_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"[NOTIFY-RESULT] Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/upload-segment', methods=['POST', 'OPTIONS'])
def upload_segment():
    """
    Upload a processed segment video from a distributed worker.
    Used by process_segment_task to upload completed segment videos.

    Request: multipart/form-data with 'file', 'video_id', 'seg_idx'
    Response: { "status": "success", "segment_url": "/results/<filename>" }
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        video_id = request.form.get('video_id', '')
        seg_idx = request.form.get('seg_idx', '')

        safe_name = sanitize_filename(file.filename)
        os.makedirs(RESULT_DIR, exist_ok=True)
        dest = os.path.join(RESULT_DIR, safe_name)
        file.save(dest)

        print(f"[OK] Received segment upload: video_id={video_id}, seg_idx={seg_idx}, file={safe_name}")

        # Upload to B2 CDN to avoid Railway egress costs
        cdn_url = None
        if B2_ENABLED:
            try:
                remote_path = f"segments/{video_id}/{safe_name}"
                cdn_url = upload_to_b2(dest, remote_path)
                print(f"[UPLOAD-SEGMENT] ✅ Uploaded to B2: {cdn_url}")
                # Clean up local file after B2 upload
                os.remove(dest)
                print(f"[UPLOAD-SEGMENT] Cleaned up local file: {dest}")
            except Exception as b2_err:
                print(f"[UPLOAD-SEGMENT] ⚠️ B2 upload failed: {b2_err}, keeping local file")

        return jsonify({
            'status': 'success',
            'segment_url': cdn_url or f'/results/{safe_name}',
            'video_id': video_id,
            'seg_idx': seg_idx
        })
    except Exception as e:
        print(f"[ERROR] Segment upload error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/temp/<path:filepath>')
def serve_temp_file(filepath):
    """Redirect to B2 temp files - workers download from CDN"""
    # Temp files now stored in B2 under temp/ prefix
    cdn_url = f"{B2_CDN_URL}/temp/{filepath}"
    print(f"[TEMP] Redirecting to B2: {cdn_url}")
    return redirect(cdn_url)


@app.route('/privacy')
def privacy_policy():
    """Serve Privacy Policy page"""
    return send_file(os.path.join(app.static_folder, 'privacy.html'))


@app.route('/terms')
def terms_of_service():
    """Serve Terms of Service page"""
    return send_file(os.path.join(app.static_folder, 'terms.html'))


@app.route('/premium')
def premium_page():
    """Serve Premium/Pricing page"""
    return send_file(os.path.join(app.static_folder, 'premium.html'))


# The API, API support and enterprise pages advertised a programmatic
# offering that does not exist - no key issuance, no endpoints, no docs.
# They were near-duplicates of each other and of /contact, so they now fold
# into the contact page rather than standing alone as thin landing pages.
@app.route('/api')
@app.route('/api-support')
@app.route('/enterprise')
def business_enquiries_redirect():
    return redirect('/contact', code=301)


@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve HTML and other static files from web folder"""
    # Redirect static assets to CDN (zero Railway egress)
    static_extensions = ('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot')
    if B2_ENABLED and filename.endswith(static_extensions):
        # Redirect to CDN - files must be uploaded to B2 under static/ prefix
        cdn_url = f"{B2_CDN_URL}/static/{filename}"
        print(f"[CDN] Redirecting static asset: {filename} -> {cdn_url}")
        return redirect(cdn_url)

    # SEO: 301 redirect .html URLs to clean URLs (avoid duplicate content)
    if filename.endswith('.html'):
        clean_url = '/' + filename[:-5]  # Remove .html extension
        return redirect(clean_url, code=301)

    # Try adding .html extension for clean URLs (e.g., /contact -> contact.html)
    html_file = filename + '.html'
    html_path = os.path.join(app.static_folder, html_file)
    if os.path.exists(html_path):
        return send_file(html_path)

    # Source and working files sit in the same folder as the pages, so the
    # fallback below was serving them to anyone who asked - and to crawlers.
    # Only the few text files that have to be public are allowed through.
    public_text_files = {'robots.txt', 'ads.txt', 'sitemap.xml'}
    base = os.path.basename(filename).lower()
    key_stem = base[:-4] if base.endswith('.txt') else ''
    is_indexnow_key = len(key_stem) == 32 and all(c in '0123456789abcdef' for c in key_stem)
    if (base.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.log', '.sh', '.bat'))
            and base not in public_text_files and not is_indexnow_key):
        return ('Not Found', 404)

    # Otherwise serve as static file (fallback)
    return send_from_directory(app.static_folder, filename)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get server statistics including queue info and average processing time

    Returns:
        {
            'queue_length': int,
            'active_tasks': int,
            'avg_processing_time': float (seconds),
            'timestamp': str
        }
    """
    try:
        # Get Celery stats. celery.task.control was dropped in Celery 4, so the
        # old import raised on every call and this endpoint always fell through
        # to the zeroed fallback below, which is why the queue always read 0.
        i = celery.control.inspect()
        active = i.active()
        scheduled = i.scheduled()
        reserved = i.reserved()  # Tasks that are reserved but not yet started

        active_count = sum(len(tasks) for tasks in (active or {}).values())
        scheduled_count = sum(len(tasks) for tasks in (scheduled or {}).values())
        reserved_count = sum(len(tasks) for tasks in (reserved or {}).values())

        # Total queue = scheduled + reserved (waiting tasks)
        total_queue = scheduled_count + reserved_count

        # Get average processing time from Redis
        avg_processing_time = 120  # Default 2 minutes
        try:
            redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
            times = redis_client.lrange('processing_times:recent', 0, 19)  # Last 20 jobs
            if times:
                times_float = [float(t) for t in times]
                avg_processing_time = sum(times_float) / len(times_float)
        except Exception as e:
            print(f"[WARN] Could not get avg processing time: {e}")

        return jsonify({
            'queue_length': total_queue,
            'active_tasks': active_count,
            'avg_processing_time': round(avg_processing_time, 1),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"[ERROR] Stats endpoint error: {e}")
        return jsonify({
            'queue_length': 0,
            'active_tasks': 0,
            'avg_processing_time': 120,
            'timestamp': datetime.utcnow().isoformat()
        })



@app.route('/api/sam2/status/<task_id>', methods=['GET', 'OPTIONS'])
def sam2_status(task_id):
    """Check SAM2 job status - polls Celery task result."""
    if request.method == 'OPTIONS':
        return ('', 204)
    try:
        result = celery.AsyncResult(task_id)
        if result.state == 'PENDING':
            return jsonify({'status': 'processing', 'progress': 0, 'message': 'Waiting in queue...'})
        elif result.state == 'PROCESSING':
            info = result.info or {}
            return jsonify({'status': 'processing', 'progress': info.get('progress', 50), 'message': info.get('status', 'Processing...')})
        elif result.state == 'SUCCESS':
            data = result.result or {}
            # Deduct credit on successful completion
            new_credits = deduct_credit_on_completion(task_id)
            response_data = {'status': 'completed', 'result_url': data.get('output_path') or data.get('result_url')}
            if new_credits is not None:
                response_data['new_credits'] = new_credits
            return jsonify(response_data)
        elif result.state == 'FAILURE':
            return jsonify({'status': 'failed', 'error': str(result.info)})
        else:
            return jsonify({'status': 'processing', 'progress': 25, 'message': f'{result.state}'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/sam2/result/<request_id>', methods=['GET'])
def sam2_get_result(request_id):
    """Poll for the result of a SAM2 selection task."""
    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        result_key = f'sam2:result:{request_id}'
        result = redis_client.get(result_key)

        if result:
            # Result is ready, return it and delete the key
            redis_client.delete(result_key)
            return jsonify(json.loads(result))
        else:
            # Not ready yet
            return jsonify({'status': 'pending'}), 202
    except Exception as e:
        print(f"[ERROR] SAM2 result check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sam2/select-object', methods=['POST'])
def sam2_select_object():
    """Interactive SAM2 object selection - uses local worker via Redis"""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON payload'}), 400
        frame_base64 = data.get('frame_data')
        points = data.get('points', [])
        video_width = data.get('video_width')
        video_height = data.get('video_height')

        if not frame_base64 or not points:
            return jsonify({'status': 'error', 'message': 'Missing frame data or points'}), 400

        request_id = f"req_{uuid.uuid4().hex[:12]}"

        REDIS_URL = os.environ.get('REDIS_URL')
        if not REDIS_URL:
            return jsonify({'status': 'error', 'message': 'Redis not configured'}), 500

        print(f"[SAM2] Using Redis: {REDIS_URL[:50]}...")
        redis_client = redis.from_url(REDIS_URL, decode_responses=False)

        # Subscribe to response channel BEFORE pushing request
        response_channel = f'sam2:selection:response:{request_id}'
        pubsub = redis_client.pubsub()
        pubsub.subscribe(response_channel)

        # Push request to list (worker uses BRPOP)
        request_data = {
            'request_id': request_id,
            'frame_data': frame_base64,
            'points': points,
            'video_width': video_width,
            'video_height': video_height
        }
        print(f"[SAM2] Pushing request {request_id} to list: sam2:selection:request")
        print(f"[SAM2] Request data: points={len(points)}, video={video_width}x{video_height}")
        redis_client.lpush('sam2:selection:request', json.dumps(request_data))

        # Wait for response. A worker that has just come up needs longer than a
        # warm one, and returning early here is what makes a click look ignored.
        timeout = 20.0
        start_time = time.time()

        print(f"[SAM2] Waiting for response on: {response_channel}")
        while time.time() - start_time < timeout:
            message = pubsub.get_message(timeout=0.1)
            if message and message['type'] == 'message':
                response_data = json.loads(message['data'])
                pubsub.unsubscribe()
                pubsub.close()
                if response_data.get('status') == 'success':
                    return jsonify({
                        'status': 'success',
                        'mask': response_data['mask'],
                        'score': response_data.get('score')
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': response_data.get('error', 'Unknown error')
                    }), 500

        pubsub.unsubscribe()
        pubsub.close()
        return jsonify({
            'status': 'busy',
            'message': 'The GPU is still starting up. Wait a few seconds and tap again.'
        }), 504

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sam2/process-video', methods=['POST'])
@require_auth
@require_credits(min_credits=1)
def sam2_process_video():
    """
    Trigger SAM2 video processing with user-selected points.
    Points are passed directly to the SAM2 worker to generate masks.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON payload'}), 400
        task_id = data.get('task_id')
        points = data.get('points', [])
        bbox = data.get('bbox')  # [x1, y1, x2, y2] for bbox mode
        prompt_mode = data.get('prompt_mode', 'point')  # 'point' or 'bbox'
        video_width = data.get('video_width')
        video_height = data.get('video_height')
        frame_index = data.get('frame_index', 0)

        if not task_id:
            return jsonify({'status': 'error', 'message': 'Missing task_id'}), 400

        # Validate based on prompt mode
        if prompt_mode == 'bbox':
            if not bbox or len(bbox) != 4:
                return jsonify({'status': 'error', 'message': 'Invalid bbox (need [x1, y1, x2, y2])'}), 400
        else:
            if not points or len(points) == 0:
                return jsonify({'status': 'error', 'message': 'No points selected'}), 400

        # Get video CDN URL from Redis (videos are uploaded directly to B2)
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        cdn_url_key = f"upload:{task_id}:cdn_url"
        cdn_url = redis_client.get(cdn_url_key)

        video_path = None
        if cdn_url:
            # Video is in B2 - worker will download it
            video_path = cdn_url.decode('utf-8') if isinstance(cdn_url, bytes) else cdn_url
        else:
            # Fallback: check for local file
            for ext in ['.mp4', '.mov', '.avi', '.webm']:
                test_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
                if os.path.exists(test_path):
                    video_path = test_path
                    break

        if not video_path:
            return jsonify({'status': 'error', 'message': f'Video not found for task {task_id}. CDN URL or local file missing.'}), 404

        # Create a unique job ID for this SAM2 processing request
        job_id = f"sam2_{task_id}_{uuid.uuid4().hex[:8]}"

        # Get public base URL for video download (same pattern as YOLO from d5443f3d)
        def _current_public_base():
            env_url = os.getenv('TUNNEL_URL')
            if env_url:
                return env_url.strip()
            try:
                tunnel_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
                if os.path.exists(tunnel_file):
                    with open(tunnel_file, 'r') as f:
                        return f.read().strip()
            except Exception:
                pass
            return 'http://localhost:9000'

        api_base = _current_public_base()

        # Create masks directory
        masks_dir = f"/tmp/{task_id}_sam2_masks"

        # Chain: WSL SAM2 mask generation → Windows ProPainter inpainting
        from celery import signature, chain

        # Build kwargs based on prompt mode
        if prompt_mode == 'bbox':
            # Bbox mode: use bounding box for SAM2 tracking
            wsl_bbox = [int(x) for x in bbox]  # [x1, y1, x2, y2]
            s1_kwargs = {'prompt_mode': 'bbox', 'bbox': wsl_bbox, 'frame_idx': frame_index, 'api_base': api_base}
            print(f"[SAM2] Mode: bbox, bbox={wsl_bbox}")
        else:
            # Point mode: use click points for SAM2 tracking
            # Frontend sends: [{x, y, label, ...}, ...]
            # WSL worker expects: points=[(x,y), ...], labels=[1, 1, ...]
            wsl_points = [(int(p.get('x', 0)), int(p.get('y', 0))) for p in points]
            wsl_labels = [int(p.get('label', 1)) for p in points]
            s1_kwargs = {'prompt_mode': 'point', 'points': wsl_points, 'labels': wsl_labels, 'frame_idx': frame_index, 'api_base': api_base}
            print(f"[SAM2] Mode: point, points={len(wsl_points)}")

        s1 = signature('sam2.generate_masks_fullfps',
                       args=[video_path, masks_dir],
                       kwargs=s1_kwargs,
                       queue='wsl_sam2')
        s2 = signature('watermark._continue_after_masks',
                       args=[video_path, task_id, points or [], video_width, video_height, frame_index, api_base],
                       queue='propainter')

        result = chain(s1, s2).apply_async(task_id=job_id)

        # Store user_id and estimated credits for deduction on completion
        user_id = session.get('user_id')
        estimated_credits = data.get('estimated_credits', 1)
        if user_id:
            try:
                redis_client.setex(f"task:{job_id}:user_id", 86400 * 7, str(user_id))
                redis_client.setex(f"task:{job_id}:credits", 86400 * 7, str(estimated_credits))
                print(f"[CREDITS] Stored user {user_id}, credits {estimated_credits} for SAM2 task {job_id}")
            except Exception as e:
                print(f"[CREDITS] Failed to store: {e}")

        print(f"[SAM2] Started WSL chain job {job_id} for video {task_id}")
        print(f"[SAM2] Prompt mode: {prompt_mode}, Video: {video_width}x{video_height}")
        print(f"[SAM2] Queue flow: wsl_sam2 -> propainter")
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'task_id': job_id,  # For compatibility
            'message': 'SAM2 processing started'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/process-static-mask', methods=['POST', 'OPTIONS'])
@require_auth
@require_credits(min_credits=1)
def process_static_mask():
    """
    Process video with a user-drawn static mask.
    Same flow as local static_mask_gui.py:
    1. Decode mask from base64
    2. Get video frame count
    3. Replicate mask for all frames
    4. Zip and upload masks to B2
    5. Call watermark._continue_after_masks (same as SAM2)
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        import base64
        import zipfile
        import shutil
        import uuid
        import time

        data = request.get_json()
        task_id = data.get('task_id')
        mask_base64 = data.get('mask_base64')
        video_width = data.get('video_width')
        video_height = data.get('video_height')
        estimated_credits = data.get('estimated_credits', 1)
        frame_count_from_client = data.get('frame_count')  # From frontend video metadata

        if not task_id:
            return jsonify({'status': 'error', 'message': 'No task_id provided'}), 400
        if not mask_base64:
            return jsonify({'status': 'error', 'message': 'No mask provided'}), 400

        # Get video URL from Redis (already uploaded by user)
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        video_url = redis_client.get(f"upload:{task_id}:cdn_url")
        if not video_url:
            return jsonify({'status': 'error', 'message': 'Video URL not found - please re-upload'}), 404

        # Get frame count - prefer client-provided, then cv2, then Redis, then fallback
        total_frames = frame_count_from_client or 100  # Use client estimate if provided
        if not frame_count_from_client:
            try:
                import cv2
                video_path = None
                for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                    test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
                    if os.path.exists(test_path):
                        video_path = test_path
                        break
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
                        cap.release()
            except Exception as cv_err:
                print(f"[STATIC MASK] cv2 not available: {cv_err}")
                # Try to get frame count from Redis metadata if stored
                try:
                    stored_frames = redis_client.get(f"upload:{task_id}:frames")
                    if stored_frames:
                        total_frames = int(stored_frames)
                except:
                    pass
        print(f"[STATIC MASK] Using {total_frames} frames (from_client={frame_count_from_client is not None})")

        # Decode mask from base64
        mask_data = mask_base64.split(',')[1] if ',' in mask_base64 else mask_base64
        mask_bytes = base64.b64decode(mask_data)

        # Create masks directory and replicate mask for all frames
        masks_dir = os.path.join(TEMP_DIR, f'{task_id}_static_masks')
        os.makedirs(masks_dir, exist_ok=True)

        # Write the same mask for each frame (0000.png, 0001.png, etc.)
        for i in range(total_frames):
            mask_path = os.path.join(masks_dir, f'{i:04d}.png')
            with open(mask_path, 'wb') as f:
                f.write(mask_bytes)
        print(f"[STATIC MASK] Created {total_frames} mask files")

        # Zip masks
        zip_path = os.path.join(TEMP_DIR, f'{task_id}_masks.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for mask_file in sorted(os.listdir(masks_dir)):
                zf.write(os.path.join(masks_dir, mask_file), mask_file)
        print(f"[STATIC MASK] Zipped masks to {zip_path}")

        # Upload masks to B2
        timestamp = int(time.time())
        remote_path = f"masks/{timestamp}_{task_id}_masks.zip"
        masks_url = upload_to_b2(zip_path, remote_path)

        if not masks_url:
            return jsonify({'status': 'error', 'message': 'Failed to upload masks to B2'}), 500
        print(f"[STATIC MASK] Uploaded masks to B2: {masks_url}")

        # Cleanup local files
        shutil.rmtree(masks_dir, ignore_errors=True)
        os.remove(zip_path)

        # Queue task using same flow as SAM2
        job_id = f"static_{task_id}_{uuid.uuid4().hex[:8]}"

        # sam2_result format expected by _continue_after_masks
        sam2_result = {
            'masks_url': masks_url,
            'mode': 'static'
        }

        # Send task to worker (same task as SAM2 uses)
        celery.send_task(
            'watermark._continue_after_masks',
            args=[sam2_result, video_url, task_id, [], video_width, video_height, 0, None],
            task_id=job_id,
            queue='propainter'
        )

        # Store user_id and estimated credits for deduction on completion
        user_id = session.get('user_id')
        if user_id:
            try:
                redis_client.setex(f"task:{job_id}:user_id", 86400 * 7, str(user_id))
                redis_client.setex(f"task:{job_id}:credits", 86400 * 7, str(estimated_credits))
                print(f"[CREDITS] Stored user {user_id}, credits {estimated_credits} for static mask task {job_id}")
            except Exception as e:
                print(f"[CREDITS] Failed to store: {e}")

        print(f"[STATIC MASK] Started job {job_id} for video {task_id}")
        print(f"[STATIC MASK] Video URL: {video_url}")
        print(f"[STATIC MASK] Masks URL: {masks_url}")

        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'task_id': job_id,
            'message': 'Static mask processing started'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# Run Server
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("WatermarkAI Production Server")
    print("=" * 60)
    print("Starting Flask server on http://0.0.0.0:9000")
    print("")
    print("To start Celery worker (in separate terminal):")
    print("  celery -A server_production.celery worker --loglevel=info --concurrency=2")
    print("")
    print("To start Redis (required):")
    print("  redis-server")
    print("")
    print("=" * 60)

    # Run Flask app
    app.run(
        host='0.0.0.0',  # Listen on all interfaces for tunnel
        port=9000,
        debug=False,  # Set to False for production
        threaded=True
    )

# ===================================
# BILLING ENDPOINTS (Stripe)
# ===================================

# Stripe Price IDs for credit packages (configurable via env vars for test/live switching)
STRIPE_PRICE_IDS = {
    # Test pack (1 credit for $0.01)
    'credits_1': os.getenv('STRIPE_PRICE_ID_CREDITS_1', 'price_1Se59cDbvxhrePJFEXWcF9wZ'),   # Test Pack - $0.01
    # One-time credit packs (5/15/60 credits) - fallback to live prices
    'credits_5': os.getenv('STRIPE_PRICE_ID_CREDITS_5', 'price_1ScgBpDbvxhrePJFnMa5NdJh'),   # Starter Pack - $2.99
    'credits_15': os.getenv('STRIPE_PRICE_ID_CREDITS_15', 'price_1ScgCKDbvxhrePJFGytHR4i5'),  # Basic Pack - $6.99
    'credits_60': os.getenv('STRIPE_PRICE_ID_CREDITS_60', 'price_1ScgCgDbvxhrePJF6JXX6Wh4'),  # Pro Pack - $24.99
    # Monthly subscription tiers
    'starter': os.getenv('STRIPE_PRICE_ID_STARTER', ''),
    'pro': os.getenv('STRIPE_PRICE_ID_PRO', ''),
}

# Credit amounts for each package
CREDIT_AMOUNTS = {
    # Test pack
    'credits_1': 1,
    # One-time credit packs
    'credits_5': 5,
    'credits_15': 15,
    'credits_60': 60,
    # Monthly subscription credits
    'starter': 10,
    'pro': 35,
}

@app.route('/api/billing/create-checkout-session', methods=['POST', 'OPTIONS'])
def create_checkout_session():
    """Create a Stripe checkout session for subscriptions or one-time credit purchases"""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not STRIPE_ENABLED:
        return jsonify({'error': 'Billing is not enabled on this server'}), 503

    try:
        data = request.get_json() or {}
        plan = data.get('plan')
        package = data.get('package')
        user_id = data.get('user_id')
        mode = data.get('mode', 'subscription')  # 'subscription' or 'payment'

        # Determine which price ID to use
        price_key = package if package else plan
        price_id = STRIPE_PRICE_IDS.get(price_key)

        if not price_id:
            return jsonify({'error': f'Invalid plan/package: {price_key}'}), 400

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Fetch user email from database
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT email FROM users WHERE id = %s', (user_id,))
                result = cur.fetchone()
                if not result:
                    return jsonify({'error': 'User not found'}), 404
                user_email = result[0]

        # Get base URL for success/cancel redirects
        base_url = request.host_url.rstrip('/')

        # Create checkout session
        checkout_params = {
            'payment_method_types': ['card'],
            'line_items': [{
                'price': price_id,
                'quantity': 1,
            }],
            'mode': mode,
            'success_url': f'{base_url}/success.html?session_id={{CHECKOUT_SESSION_ID}}',
            'cancel_url': f'{base_url}/premium.html',
            'customer_email': user_email,  # Pre-fill with user's actual email
            'client_reference_id': user_id,
            'metadata': {
                'user_id': user_id,
                'plan': plan if plan else '',
                'package': package if package else '',
            }
        }

        # Only add customer_creation for one-time payments (not subscriptions)
        if mode == 'payment':
            checkout_params['customer_creation'] = 'always'

        session = stripe.checkout.Session.create(**checkout_params)

        return jsonify({'url': session.url})

    except Exception as e:
        print(f"[BILLING-ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/billing/webhook', methods=['POST'])
@app.route('/api/stripe/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for payment events"""
    print(f"[STRIPE-WEBHOOK] *** WEBHOOK HIT *** STRIPE_ENABLED={STRIPE_ENABLED}")
    if not STRIPE_ENABLED:
        return jsonify({'error': 'Billing is not enabled'}), 503

    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')

    try:
        # Verify webhook signature if secret is configured
        if webhook_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            event = json.loads(payload)
            print("[WARNING] Stripe webhook signature verification disabled (no STRIPE_WEBHOOK_SECRET)")

        event_type = event['type']
        data_object = event['data']['object']

        print(f"[STRIPE-WEBHOOK] Received: {event_type}")
        print(f"[STRIPE-WEBHOOK] AUTH_ENABLED={AUTH_ENABLED}, db_pool={'OK' if db_pool else 'NONE'}")

        # Handle successful checkout (one-time or subscription)
        if event_type == 'checkout.session.completed':
            user_id = data_object.get('client_reference_id')
            metadata = data_object.get('metadata', {})
            package = metadata.get('package')
            plan = metadata.get('plan')

            # Debug: Log all relevant data
            print(f"[STRIPE-WEBHOOK] client_reference_id (user_id): {user_id}")
            print(f"[STRIPE-WEBHOOK] metadata: {metadata}")
            print(f"[STRIPE-WEBHOOK] package={package}, plan={plan}")

            # Determine credit amount
            key = package if package else plan
            credits_to_add = CREDIT_AMOUNTS.get(key, 0)
            print(f"[STRIPE-WEBHOOK] key={key}, credits_to_add={credits_to_add}")

            if credits_to_add > 0 and user_id:
                # Award credits to user
                print(f"[BILLING] User {user_id} purchased {key}: +{credits_to_add} credits")
                if AUTH_ENABLED:
                    try:
                        with get_db() as conn:
                            cur = conn.cursor()
                            cur.execute(
                                'UPDATE users SET credits = credits + %s WHERE id = %s RETURNING credits',
                                (credits_to_add, user_id)
                            )
                            result = cur.fetchone()
                            if result:
                                new_balance = result[0]
                                print(f"[BILLING] ✅ Credits added successfully. User {user_id} new balance: {new_balance}")

                                # Record purchase in history
                                try:
                                    cur.execute('''
                                        INSERT INTO purchases (user_id, stripe_session_id, stripe_payment_intent, package, credits_awarded, amount_cents, currency, status)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    ''', (
                                        user_id,
                                        data_object.get('id'),
                                        data_object.get('payment_intent'),
                                        key,
                                        credits_to_add,
                                        data_object.get('amount_total'),
                                        data_object.get('currency', 'usd'),
                                        'completed'
                                    ))
                                    print(f"[BILLING] ✅ Purchase recorded in history")
                                except Exception as pe:
                                    print(f"[BILLING] ⚠️ Failed to record purchase history: {pe}")
                            else:
                                print(f"[BILLING] ❌ User {user_id} not found in database")
                    except Exception as e:
                        print(f"[BILLING] ❌ Failed to add credits for user {user_id}: {e}")
                        import traceback; traceback.print_exc()
                else:
                    print(f"[BILLING] ⚠️ AUTH_ENABLED=False - cannot add credits (no database)")
            else:
                print(f"[BILLING] ❌ SKIPPED credit addition: credits_to_add={credits_to_add}, user_id={user_id}")
                if not user_id:
                    print(f"[BILLING]    Missing user_id! Check client_reference_id in checkout session")
                if credits_to_add == 0:
                    print(f"[BILLING]    credits_to_add=0! Check package key '{key}' exists in CREDIT_AMOUNTS")

        # Handle subscription renewals
        elif event_type == 'invoice.payment_succeeded':
            customer_id = data_object.get('customer')
            subscription_id = data_object.get('subscription')

            if subscription_id:
                # TODO: Award renewal credits based on subscription plan
                print(f"[BILLING] Subscription {subscription_id} renewed for customer {customer_id}")

        # Handle subscription updates/cancellations
        elif event_type in ['customer.subscription.updated', 'customer.subscription.deleted']:
            subscription = data_object
            customer_id = subscription.get('customer')
            status = subscription.get('status')

            print(f"[BILLING] Subscription for customer {customer_id} is now: {status}")

        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"[WEBHOOK-ERROR] {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/billing/create-portal-session', methods=['POST', 'OPTIONS'])
def create_portal_session():
    """Create a Stripe billing portal session for subscription management"""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not STRIPE_ENABLED:
        return jsonify({'error': 'Billing is not enabled'}), 503

    try:
        data = request.get_json() or {}
        customer_id = data.get('customer_id')
        session_id = data.get('session_id')

        # Fallback: retrieve customer_id from checkout session
        if not customer_id and session_id:
            try:
                checkout_session = stripe.checkout.Session.retrieve(session_id)
                customer_id = checkout_session.customer

                # If no customer attached to session, find by email
                if not customer_id:
                    checkout_email = checkout_session.customer_email or (
                        checkout_session.customer_details.email if checkout_session.customer_details else None
                    )
                    if checkout_email:
                        customers = stripe.Customer.list(email=checkout_email, limit=1)
                        if customers.data:
                            customer_id = customers.data[0].id
            except stripe.error.StripeError as exc:
                return jsonify({'error': str(exc)}), 400

        # Fallback: find customer by authenticated user's email
        if not customer_id:
            user_id = session.get('user_id')
            if user_id and AUTH_ENABLED:
                try:
                    with get_db() as conn:
                        cur = conn.cursor()
                        cur.execute('SELECT email FROM users WHERE id = %s', (user_id,))
                        row = cur.fetchone()
                        if row:
                            user_email = row[0]
                            customers = stripe.Customer.list(email=user_email, limit=1)
                            if customers.data:
                                customer_id = customers.data[0].id
                except Exception as e:
                    print(f"[BILLING-PORTAL] Could not find customer by email: {e}")

        if not customer_id:
            return jsonify({'error': 'No Stripe customer found. Please make a purchase first.'}), 400

        # Get base URL for return redirect
        base_url = request.host_url.rstrip('/')

        # Create portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f'{base_url}/premium.html',
        )

        return jsonify({'url': portal_session.url})

    except Exception as e:
        print(f"[BILLING-PORTAL-ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/billing/purchase-history', methods=['GET', 'OPTIONS'])
def get_purchase_history():
    """Get user's purchase history from local database"""
    if request.method == 'OPTIONS':
        return ('', 204)

    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        with get_db() as conn:
            cur = conn.cursor()
            # Get purchases from local database (user_id stored as string)
            cur.execute('''
                SELECT id, stripe_session_id, package, credits_awarded, amount_cents, currency, status, created_at
                FROM purchases
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 50
            ''', (str(user_id),))
            rows = cur.fetchall()

            purchases = []
            for row in rows:
                purchases.append({
                    'id': row[0],
                    'stripe_session_id': row[1],
                    'package': row[2] or 'Unknown',
                    'credits_awarded': row[3] or 0,
                    'amount_cents': row[4] or 0,
                    'currency': row[5] or 'usd',
                    'status': row[6] or 'completed',
                    'created_at': row[7].isoformat() if row[7] else None
                })

            return jsonify({'purchases': purchases})

    except Exception as e:
        print(f"[PURCHASE-HISTORY-ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Explicit CORS preflight catch-all for /api/* (helps when proxies strip headers)
# ============================================================================
# Native sign-in (iOS)
# ============================================================================

NATIVE_AUTH_SCHEME = os.getenv('NATIVE_AUTH_SCHEME', 'markremoverai')
APPLE_SIGNIN_BUNDLE_IDS = [
    b.strip() for b in os.getenv(
        'APPLE_SIGNIN_BUNDLE_IDS', 'com.markremoverai.app'
    ).split(',') if b.strip()
]


def _mint_native_auth_code(user_id):
    """One-time code the app trades for a session. Lives in Redis for 5 minutes
    and is deleted on first use, so intercepting the redirect buys nothing after
    the app has already redeemed it."""
    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        code = secrets.token_urlsafe(32)
        redis_client.setex(f'nativeauth:{code}', 300, str(user_id))
        return code
    except Exception as exc:
        print(f"[NATIVE-AUTH] Could not mint code: {exc}")
        return None


def _display_name(raw_name, email):
    """A person's name, or a clean fallback - never an email hash.

    Apple sends the name only on the very first authorization, and with Hide My
    Email the address is a random string like k2j9x8mn4p@privaterelay.appleid.com.
    Falling back to the part before the @ therefore printed that random string as
    the user's name, which is what appeared on the profile screen.
    """
    name = " ".join((raw_name or "").split())
    if name:
        # First and last only. Apple can hand back a long chain of given,
        # middle and family names, and the profile screen has one line.
        parts = name.split()
        return parts[0] if len(parts) == 1 else f"{parts[0]} {parts[-1]}"

    local = (email or "").split("@")[0]
    domain = (email or "").split("@")[-1].lower()
    # A relay local-part carries no information about the person.
    if not local or "privaterelay.appleid.com" in domain or not any(c.isalpha() for c in local):
        return "There"
    # someone.name / someone_name / someone.name123 -> Someone Name
    cleaned = local.replace(".", " ").replace("_", " ").replace("-", " ")
    words = [w for w in cleaned.split() if any(c.isalpha() for c in w)]
    if not words:
        return "There"
    words = [w.capitalize() for w in words][:2]
    return " ".join(words)


def _session_for_user(cur, user_id):
    """Loads a user and installs them into the Flask session."""
    # Covers the Google and Apple sign-in paths, which return the user object
    # directly instead of going through /api/auth/status first.
    grant_daily_free_credits(cur, user_id)
    cur.execute(
        'SELECT email, name, credits, email_verified, created_at FROM users WHERE id = %s',
        (user_id,)
    )
    row = cur.fetchone()
    if not row:
        return None

    email, name, credits, email_verified, created_at = row
    session['user_id'] = user_id
    session['email'] = email
    session['name'] = name
    session.permanent = True

    return {
        'id': user_id,
        'email': email,
        'name': name,
        'credits': float(credits or 0),
        'email_verified': bool(email_verified),
        'created_at': created_at.isoformat() if created_at else None
    }


@app.route('/api/auth/exchange', methods=['POST', 'OPTIONS'])
def auth_exchange():
    """Trades the one-time code from the native Google flow for a session."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED or not db_pool:
        return jsonify({'error': 'Authentication not enabled'}), 503

    code = (request.get_json(silent=True) or {}).get('code', '')
    if not code:
        return jsonify({'error': 'Missing code'}), 400

    try:
        redis_client = redis.from_url(os.environ.get('REDIS_URL'), decode_responses=True)
        key = f'nativeauth:{code}'
        user_id = redis_client.get(key)
        # Burn it immediately - a code is good for exactly one exchange.
        redis_client.delete(key)
    except Exception as exc:
        print(f"[NATIVE-AUTH] Redis lookup failed: {exc}")
        return jsonify({'error': 'Sign-in temporarily unavailable'}), 503

    if not user_id:
        return jsonify({'error': 'That sign-in link expired. Try again.'}), 401

    with get_db() as conn:
        cur = conn.cursor()
        user = _session_for_user(cur, int(user_id))

    if not user:
        return jsonify({'error': 'Account not found'}), 404

    print(f"[NATIVE-AUTH] Exchanged code for user {user_id}")
    return jsonify({'status': 'success', 'user': user})


def _verify_apple_identity_token(identity_token):
    """Validates a Sign in with Apple JWT against Apple's published keys."""
    import jwt
    from jwt import PyJWKClient

    jwk_client = PyJWKClient('https://appleid.apple.com/auth/keys')
    signing_key = jwk_client.get_signing_key_from_jwt(identity_token)

    return jwt.decode(
        identity_token,
        signing_key.key,
        algorithms=['RS256'],
        audience=APPLE_SIGNIN_BUNDLE_IDS,
        issuer='https://appleid.apple.com'
    )


@app.route('/api/auth/apple', methods=['POST', 'OPTIONS'])
def auth_apple():
    """Sign in with Apple. The app sends the identity token it got natively."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED or not db_pool:
        return jsonify({'error': 'Authentication not enabled'}), 503

    data = request.get_json(silent=True) or {}
    identity_token = data.get('identity_token')
    if not identity_token:
        return jsonify({'error': 'Missing identity_token'}), 400

    try:
        claims = _verify_apple_identity_token(identity_token)
    except Exception as exc:
        print(f"[APPLE-SIGNIN] Token rejected: {exc}")
        return jsonify({'error': 'Could not verify that Apple sign-in.'}), 401

    apple_id = claims.get('sub')
    # Apple only sends the address on the very first authorization, and it may
    # be a private relay one. The app forwards what it was given.
    email = (claims.get('email') or data.get('email') or '').strip().lower()
    name = (data.get('name') or '').strip()

    if not apple_id:
        return jsonify({'error': 'Apple token was missing a subject'}), 400

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('ALTER TABLE users ADD COLUMN IF NOT EXISTS apple_id TEXT')

            cur.execute('SELECT id FROM users WHERE apple_id = %s', (apple_id,))
            row = cur.fetchone()

            if row:
                user_id = row[0]
            else:
                user_id = None
                if email:
                    cur.execute('SELECT id FROM users WHERE email = %s', (email,))
                    existing = cur.fetchone()
                    if existing:
                        # Same person arriving by a new door - link, don't fork.
                        user_id = existing[0]
                        cur.execute('UPDATE users SET apple_id = %s WHERE id = %s', (apple_id, user_id))
                        print(f"[APPLE-SIGNIN] Linked Apple to existing user {email}")

                if user_id is None:
                    if not email:
                        return jsonify({'error': 'Apple did not share an email address.'}), 400
                    # Apple has already verified the address, so skip our own
                    # verification mail and match the Google signup grant.
                    cur.execute(
                        '''INSERT INTO users (apple_id, email, name, credits, email_verified)
                           VALUES (%s, %s, %s, %s, TRUE) RETURNING id''',
                        (apple_id, email, _display_name(name, email), 2)
                    )
                    user_id = cur.fetchone()[0]
                    print(f"[APPLE-SIGNIN] New user {email} (2 free credits)")

            user = _session_for_user(cur, user_id)

        return jsonify({'status': 'success', 'user': user})

    except Exception as exc:
        print(f"[APPLE-SIGNIN] Failed: {exc}")
        return jsonify({'error': 'Sign-in failed'}), 500


# ============================================================================
# Apple In-App Purchase
# ============================================================================
# iOS cannot sell credits through Stripe - Apple requires StoreKit for digital
# goods. The app buys a consumable, then posts the signed transaction here so
# the credits land on the same `users.credits` column Stripe writes to.

APPLE_BUNDLE_ID = os.getenv('APPLE_BUNDLE_ID', 'com.markremoverai.app')

# Product id -> credits. Priced to match the web packs.
APPLE_PRODUCT_CREDITS = {
    'com.markremoverai.app.credits5': 5,
    'com.markremoverai.app.credits15': 15,
    'com.markremoverai.app.credits60': 60,
}

# Apple's root CAs, PEM or DER, used to anchor the x5c chain on a signed
# transaction. Without them we cannot prove a payload came from Apple, so the
# endpoint refuses to grant anything.
APPLE_ROOT_CERT_DIR = os.getenv('APPLE_ROOT_CERT_DIR', os.path.join(SCRIPT_DIR, 'certs', 'apple'))

_apple_verifier = None
_apple_verifier_error = None


def _load_apple_verifier():
    """Build the App Store signed-data verifier once, or record why we can't."""
    global _apple_verifier, _apple_verifier_error
    if _apple_verifier is not None or _apple_verifier_error is not None:
        return _apple_verifier

    try:
        from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier
        from appstoreserverlibrary.models.Environment import Environment
    except ImportError:
        _apple_verifier_error = 'app-store-server-library is not installed'
        print(f"[APPLE-IAP] {_apple_verifier_error}")
        return None

    roots = []
    try:
        for name in sorted(os.listdir(APPLE_ROOT_CERT_DIR)):
            if name.lower().endswith(('.cer', '.der', '.pem')):
                with open(os.path.join(APPLE_ROOT_CERT_DIR, name), 'rb') as fh:
                    roots.append(fh.read())
    except FileNotFoundError:
        pass

    if not roots:
        _apple_verifier_error = f'no Apple root certificates in {APPLE_ROOT_CERT_DIR}'
        print(f"[APPLE-IAP] {_apple_verifier_error}")
        return None

    # Sandbox by default: the App Store Connect app record - and therefore the
    # numeric app id below - does not exist until the app is submitted, and
    # Production without it raises rather than verifying anything.
    env_name = os.getenv('APPLE_IAP_ENVIRONMENT', 'sandbox').lower()
    environment = Environment.PRODUCTION if env_name == 'production' else Environment.SANDBOX

    # Required by the library for Production, ignored for Sandbox. It is the
    # numeric "Apple ID" on the App Store Connect app record, not the bundle id.
    raw_app_id = os.getenv('APPLE_APP_APPLE_ID', '').strip()
    app_apple_id = int(raw_app_id) if raw_app_id.isdigit() else None

    if environment == Environment.PRODUCTION and app_apple_id is None:
        _apple_verifier_error = (
            'APPLE_IAP_ENVIRONMENT=production requires APPLE_APP_APPLE_ID '
            '(the numeric app id from App Store Connect)'
        )
        print(f"[APPLE-IAP] {_apple_verifier_error}")
        return None

    try:
        _apple_verifier = SignedDataVerifier(
            roots, True, environment, APPLE_BUNDLE_ID, app_apple_id
        )
        print(f"[APPLE-IAP] Verifier ready ({environment.name}, bundle {APPLE_BUNDLE_ID})")
    except Exception as exc:
        _apple_verifier_error = f'verifier init failed: {exc}'
        print(f"[APPLE-IAP] {_apple_verifier_error}")

    return _apple_verifier


def _ensure_apple_purchase_table(cur):
    """One row per Apple transaction id. The unique key is what makes redeeming
    idempotent - a replayed receipt collides instead of paying out twice."""
    cur.execute('''
        CREATE TABLE IF NOT EXISTS apple_purchases (
            id SERIAL PRIMARY KEY,
            transaction_id TEXT UNIQUE NOT NULL,
            original_transaction_id TEXT,
            user_id INTEGER NOT NULL,
            product_id TEXT NOT NULL,
            credits_awarded INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')


@app.route('/api/billing/apple/redeem', methods=['POST', 'OPTIONS'])
@require_auth
def apple_redeem():
    """Verify a StoreKit 2 signed transaction and credit the signed-in user."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED or not db_pool:
        return jsonify({'status': 'error', 'message': 'Accounts are unavailable right now.'}), 503

    data = request.get_json(silent=True) or {}
    jws = data.get('signed_transaction')
    if not jws:
        return jsonify({'status': 'error', 'message': 'Missing signed_transaction'}), 400

    verifier = _load_apple_verifier()
    if verifier is None:
        # Fail closed. An unverified payload is just a string from the internet.
        print(f"[APPLE-IAP] Refusing to redeem: {_apple_verifier_error}")
        return jsonify({
            'status': 'error',
            'message': 'Purchase verification is unavailable. Your purchase is safe and will be credited once this is fixed.'
        }), 503

    try:
        payload = verifier.verify_and_decode_signed_transaction(jws)
    except Exception as exc:
        print(f"[APPLE-IAP] Verification failed: {exc}")
        return jsonify({'status': 'error', 'message': 'Could not verify that purchase.'}), 400

    product_id = getattr(payload, 'productId', None)
    transaction_id = getattr(payload, 'transactionId', None)
    original_transaction_id = getattr(payload, 'originalTransactionId', None)
    bundle_id = getattr(payload, 'bundleId', None)

    if bundle_id and bundle_id != APPLE_BUNDLE_ID:
        return jsonify({'status': 'error', 'message': 'That purchase belongs to a different app.'}), 400

    credits_to_add = APPLE_PRODUCT_CREDITS.get(product_id, 0)
    if not credits_to_add or not transaction_id:
        return jsonify({'status': 'error', 'message': f'Unknown product {product_id}'}), 400

    user_id = session.get('user_id')

    try:
        with get_db() as conn:
            cur = conn.cursor()
            _ensure_apple_purchase_table(cur)

            # Claim the transaction first. If it is already claimed this is a
            # replay or a retry, so report the current balance and stop.
            cur.execute(
                '''INSERT INTO apple_purchases
                       (transaction_id, original_transaction_id, user_id, product_id, credits_awarded)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (transaction_id) DO NOTHING
                   RETURNING id''',
                (str(transaction_id), str(original_transaction_id or ''), user_id, product_id, credits_to_add)
            )
            claimed = cur.fetchone()

            if not claimed:
                cur.execute('SELECT credits FROM users WHERE id = %s', (user_id,))
                row = cur.fetchone()
                print(f"[APPLE-IAP] Transaction {transaction_id} already redeemed")
                return jsonify({
                    'status': 'success',
                    'already_redeemed': True,
                    'credits': float(row[0]) if row else 0
                })

            cur.execute(
                'UPDATE users SET credits = credits + %s WHERE id = %s RETURNING credits',
                (credits_to_add, user_id)
            )
            row = cur.fetchone()
            new_balance = float(row[0]) if row else 0

            print(f"[APPLE-IAP] User {user_id} redeemed {product_id}: +{credits_to_add} -> {new_balance}")
            return jsonify({
                'status': 'success',
                'credits_added': credits_to_add,
                'credits': new_balance
            })

    except Exception as exc:
        print(f"[APPLE-IAP] Redeem failed: {exc}")
        return jsonify({'status': 'error', 'message': 'Could not apply that purchase.'}), 500


@app.route('/api/<path:subpath>', methods=['OPTIONS'])
def cors_preflight(subpath):
    return ('', 204)
