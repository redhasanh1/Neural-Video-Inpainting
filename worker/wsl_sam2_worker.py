"""
WSL2 Celery worker for SAM2 mask generation (FULL FPS).

Runs entirely inside WSL2, connected to the same Redis broker as Windows.
Generates masks into a shared path (e.g., /mnt/d/watermarkz/temp/<video_id>_sam2_masks),
so the Windows Celery task can immediately consume them without extra copies.
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
import glob
import time
import zipfile
import shutil
import gc
from celery import Celery
from gpu_coordinator import GPUCoordinator

# Resolve base dir assuming this file resides in repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Redis URL (shared with Windows worker)
REDIS_URL = os.getenv('REDIS_URL', 'redis://:watermarkz_secure_2024@localhost:6379/0')
redis_path = os.path.join(BASE_DIR, 'redis_url.txt')
if os.path.exists(redis_path):
    with open(redis_path, 'r') as f:
        REDIS_URL = f.read().strip()

celery = Celery('wsl_sam2_worker', broker=REDIS_URL, backend=REDIS_URL)
celery.conf.task_track_started = True
celery.conf.broker_connection_retry_on_startup = True

# This worker reaches Redis through Railway's public proxy, which drops idle
# connections. Celery's redis transport waits on a blocking BRPOP, and with no
# socket timeout that call never returns once the connection is half-open: the
# process stays alive and "ready", stops consuming wsl_sam2, and because it
# never exits the tmux watchdog never restarts it and no crash notification is
# sent. That is how four renders sat queued while the container looked healthy.
# socket_timeout bounds the wait, health_check_interval re-pings an idle
# connection, keepalive stops the proxy reaping it. Verified against celery 5.6:
# socket options in result_backend_transport_options never reach the client, so
# the backend needs the redis_* keys below.
celery.conf.update(
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

# Add local path for imports
sys.path.insert(0, BASE_DIR)

# GPU Coordinator for exclusive GPU access (prevents SAM2/ProPainter conflicts)
_gpu_coord = GPUCoordinator(worker_type='sam2')


@celery.task(name='sam2.generate_masks_fullfps', bind=True)
def generate_masks_fullfps(self, video_path, masks_dir, prompt_mode='point', points=None, labels=None, object_ids=None, bbox=None, frame_idx=0, api_base=None, reid_mode=None, modified_masks=None):
    """
    Generate SAM2 masks at full FPS inside WSL2.

    Args:
        video_path (str): Windows or WSL path to video (or remote path like /data/uploads/...)
        masks_dir (str): Windows or WSL path for output masks
        prompt_mode (str): 'point' or 'bbox'
        points (list): List of (x, y) tuples if prompt_mode is 'point' (supports multiple clicks!)
        labels (list): List of labels (1=foreground, 0=background) for each point
        object_ids (list): List of object IDs for each point (each click = separate object)
        bbox (list): [x1, y1, x2, y2] if prompt_mode is 'bbox'
        frame_idx (int): starting frame index
        api_base (str): Base URL for downloading video if path is remote
        modified_masks (list): Pre-erased masks from user [{objectId, mask (base64)}] - if provided, use instead of generating from points
    Returns:
        dict: {status, masks_dir, masks_saved, total_frames}
    """
    # Extract job_id early for error handling
    job_id = None
    task_id = self.request.id
    if task_id and task_id.startswith('objrem_'):
        job_id = task_id.replace('objrem_', '')

    # Acquire GPU lock (waits if ProPainter is using GPU)
    with _gpu_coord.gpu_exclusive(task_id):
        try:
            return _generate_masks_fullfps_impl(self, video_path, masks_dir, prompt_mode, points, labels, object_ids, bbox, frame_idx, api_base, reid_mode, job_id, modified_masks)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[SAM2-ERROR] Task failed: {e}")

            # Update Redis with error status so frontend knows
            if job_id:
                try:
                    import redis
                    redis_client = redis_from_url(os.environ.get('REDIS_URL'), decode_responses=True)
                    redis_client.hset(f"objrem:{job_id}", mapping={
                        'status': 'error',
                        'message': str(e)[:500]
                    })
                    print(f"[SAM2-ERROR] Updated Redis with error status for job {job_id}")
                except Exception as redis_err:
                    print(f"[SAM2-ERROR] Failed to update Redis: {redis_err}")

            # Force VRAM cleanup to prevent memory buildup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("[SAM2-ERROR] Cleaned up CUDA memory")
            except Exception:
                pass

            # Re-raise so Celery marks the task as FAILED
            raise


def _generate_masks_fullfps_impl(self, video_path, masks_dir, prompt_mode, points, labels, object_ids, bbox, frame_idx, api_base, reid_mode, job_id, modified_masks=None):
    """Actual implementation of generate_masks_fullfps (wrapped for error handling)."""
    import os
    import cv2
    import numpy as np
    from pathlib import Path
    import requests

    # Import helper module from repo (runs under WSL)
    import sam2_track_wsl2 as sam2w
    import subprocess, shlex, shutil

    def to_wsl_path(p: str) -> str:
        if not isinstance(p, str):
            return p
        # Already looks like WSL
        if p.startswith('/mnt/') or p.startswith('/'):
            return p
        # Convert D:\... → /mnt/d/...
        if len(p) >= 2 and p[1] == ':' and p[0].isalpha():
            drive = p[0].lower()
            rest = p[2:].replace('\\', '/')
            rest = rest.lstrip('/')
            return f"/mnt/{drive}/{rest}"
        return p.replace('\\', '/')

    video_path_wsl = to_wsl_path(video_path)
    masks_dir_wsl = to_wsl_path(masks_dir)
    temp_frames_dir = to_wsl_path('/tmp/sam2_frames')

    # If video doesn't exist locally OR is a Railway path (/data/...), download it from api_base
    is_railway_path = video_path.startswith('/data/') or video_path_wsl.startswith('/data/')
    if (is_railway_path or not os.path.exists(video_path_wsl)) and api_base:
        filename = os.path.basename(video_path)
        download_url = f"{api_base}/uploads/{filename}"
        local_video = f"/tmp/{filename}"
        print(f"[WSL] Downloading video from {download_url}...")
        self.update_state(state='PROCESSING', meta={'status': 'Downloading video', 'progress': 2})
        r = requests.get(download_url, timeout=300)
        r.raise_for_status()
        with open(local_video, 'wb') as f:
            f.write(r.content)
        video_path_wsl = local_video
        print(f"[WSL] Downloaded {len(r.content) / 1024 / 1024:.1f} MB to {local_video}")

    os.makedirs(masks_dir_wsl, exist_ok=True)

    # Optional: track at 10fps for speed, then Windows will expand masks back to full-FPS
    use_10fps_env = os.getenv('SAM2_WSL_TRACK_10FPS', '0').lower() in ('1','true','yes','on')
    # If Windows already created a 10fps video (e.g., *_tracking_10fps.mp4), do NOT try to downsample again
    already_10 = ('_tracking_10fps' in os.path.basename(video_path_wsl)) or video_path_wsl.lower().endswith('_10fps.mp4')
    ffmpeg_ok = shutil.which('ffmpeg') is not None

    # --- ROTATION NORMALIZE (shared by BOTH decode paths) ---
    # Phone videos (.mov/.mp4) carry a container display-rotation flag. Browsers APPLY it, so
    # the user clicks in the UPRIGHT orientation and the frontend sends upright video_width/height.
    # But every decoder downstream reads the RAW sideways frames / sideways dims: extract_frames
    # (cv2) AND the DALI streamer (cv2 dim-read + ffmpeg convert that then squishes it). The click
    # coords miss the object -> all-black masks -> "No mask content generated - SAM2 tracking failed".
    # Fix once, here, before either path runs: bake the rotation into the pixels with ffmpeg
    # (autorotate is ON by default; -metadata rotate=0 strips the legacy tag) at FULL resolution so
    # the downstream downscale + point-scaling stay correct. Then extract_frames AND the DALI
    # _convert_to_h265 both receive an upright file whose cv2 dims match the frontend's coords.
    if ffmpeg_ok:
        try:
            _probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream_side_data=rotation:stream_tags=rotate',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path_wsl],
                capture_output=True, text=True, timeout=30).stdout
            _rot = 0
            for _tok in _probe.split():
                try:
                    _rot = int(float(_tok)) % 360
                    break
                except Exception:
                    continue
            if _rot != 0:
                _upright = f"/tmp/upright_{os.path.basename(video_path_wsl)}.mp4"
                _ok = False
                for _enc in (['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '20'],
                             ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20']):
                    try:
                        subprocess.run(
                            ['ffmpeg', '-y', '-i', video_path_wsl] + _enc +
                            ['-an', '-metadata:s:v:0', 'rotate=0', _upright],
                            check=True, capture_output=True, timeout=600)
                        _ok = True
                        break
                    except Exception:
                        continue
                if _ok:
                    print(f"[WSL-SAM2][ROTATE] Baked {_rot}deg display rotation upright "
                          f"(cv2/DALI ignore the flag): {os.path.basename(video_path_wsl)} -> {os.path.basename(_upright)}")
                    video_path_wsl = _upright
                else:
                    print(f"[WSL-SAM2][ROTATE] WARNING: {_rot}deg rotation flag present but ffmpeg "
                          f"autorotate failed; masks may be misaligned")
        except Exception as _e:
            print(f"[WSL-SAM2][ROTATE] rotation check skipped: {_e}")

    track_src = video_path_wsl
    if use_10fps_env and not already_10 and ffmpeg_ok:
        video_10_path = f"/tmp/sam2_{os.path.basename(video_path_wsl)}_10fps.mp4"
        cmd = f"ffmpeg -y -i {shlex.quote(video_path_wsl)} -vf fps=10 -c:v libx264 -preset ultrafast -crf 18 -an {shlex.quote(video_10_path)}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            track_src = video_10_path
        except Exception as e:
            # Fallback: if ffmpeg fails, track at full FPS instead of crashing
            print(f"[WSL] Warning: ffmpeg downsample failed ({e}); tracking at full FPS")

    # Extract frames
    self.update_state(state='PROCESSING', meta={'status': 'Extracting frames', 'progress': 5})
    total_frames, fps, orig_w, orig_h, new_w, new_h = sam2w.extract_frames(track_src, temp_frames_dir)

    # Scale coordinates if frames were resized
    scale = new_h / orig_h if orig_h != new_h else 1.0
    if scale != 1.0:
        print(f"[WSL-SAM2] Scaling coordinates by {scale:.3f} (original {orig_w}x{orig_h} -> {new_w}x{new_h})")

    # Handle user-modified masks (from eraser tool)
    # If provided, save them as initial mask for frame 0 and use mask-based propagation
    initial_mask = None
    if modified_masks and len(modified_masks) > 0:
        import base64
        print(f"[WSL-SAM2] Received {len(modified_masks)} modified mask(s) from user")

        # Combine all object masks into one (OR them together)
        combined_mask = None
        for mask_info in modified_masks:
            try:
                mask_b64 = mask_info.get('mask', '')
                if not mask_b64:
                    continue

                # Decode base64 PNG to numpy array
                mask_bytes = base64.b64decode(mask_b64)
                mask_arr = np.frombuffer(mask_bytes, np.uint8)
                mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

                if mask_img is not None:
                    # Scale mask to match extracted frames
                    if scale != 1.0:
                        mask_img = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                    # Combine with OR
                    if combined_mask is None:
                        combined_mask = mask_img
                    else:
                        combined_mask = np.maximum(combined_mask, mask_img)

                    print(f"[WSL-SAM2] Decoded mask for objectId {mask_info.get('objectId')}: {mask_img.shape}, nonzero={np.count_nonzero(mask_img)}")
            except Exception as e:
                print(f"[WSL-SAM2] Failed to decode mask: {e}")

        if combined_mask is not None:
            initial_mask = combined_mask
            # Save as frame 0 mask so it's visible in output
            mask_path = os.path.join(masks_dir_wsl, f"{frame_idx:05d}.png")
            cv2.imwrite(mask_path, combined_mask)
            print(f"[WSL-SAM2] Saved combined initial mask to {mask_path}, shape={combined_mask.shape}")

    # Run tracking (PyTorch-only in-process; TensorRT hybrid can be added later)
    self.update_state(state='PROCESSING', meta={'status': 'Tracking SAM2', 'progress': 20})
    if initial_mask is not None:
        # Use pre-provided mask instead of generating from points
        print(f"[WSL-SAM2] Using user-modified mask for tracking (skipping point-based generation)")
        masks_saved = sam2w.track_video_pytorch_only(
            temp_frames_dir, total_frames, masks_dir_wsl,
            points=None, labels=None, object_ids=None, bbox=None,
            frame_idx_start=int(frame_idx),
            video_path=video_path_wsl,
            initial_mask=initial_mask  # Pass the pre-erased mask
        )
    elif prompt_mode == 'point' and points is not None and len(points) > 0:
        # Convert points to list of (x, y) tuples and scale
        pts = [(int(p[0] * scale), int(p[1] * scale)) for p in points]
        # Labels default to all foreground (1) if not provided
        lbls = labels if labels is not None else [1] * len(pts)
        # Object IDs: each click = separate object (default all to 0 if not provided)
        obj_ids = object_ids if object_ids is not None else [0] * len(pts)
        print(f"[WSL-SAM2] Using {len(pts)} scaled point(s): {pts}, object_ids: {obj_ids}")
        masks_saved = sam2w.track_video_pytorch_only(
            temp_frames_dir, total_frames, masks_dir_wsl,
            points=pts, labels=lbls, object_ids=obj_ids, bbox=None, frame_idx_start=int(frame_idx),
            video_path=video_path_wsl  # For DALI GPU-direct streaming
        )
    elif prompt_mode == 'bbox' and bbox is not None:
        bb = [int(x * scale) for x in bbox[:4]]
        print(f"[WSL-SAM2] Using scaled bbox: {bb}")
        masks_saved = sam2w.track_video_pytorch_only(
            temp_frames_dir, total_frames, masks_dir_wsl,
            points=None, labels=None, bbox=bb, frame_idx_start=int(frame_idx),
            video_path=video_path_wsl  # For DALI GPU-direct streaming
        )
    else:
        raise ValueError('Invalid prompt: provide points or bbox')

    # Force VRAM cleanup after tracking
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # Check if we should skip B2 upload (local mode)
    SKIP_B2_UPLOAD = os.getenv('SKIP_B2_UPLOAD', '0').lower() in ('1', 'true', 'yes', 'on')

    if SKIP_B2_UPLOAD:
        # LOCAL MODE: Keep masks on disk, return local path
        print(f"[LOCAL] Skipping B2 upload - masks stay at: {masks_dir_wsl}")
        self.update_state(state='PROCESSING', meta={'status': 'Masks saved locally', 'progress': 90})
        masks_url = None
        masks_dir_result = masks_dir_wsl  # Return local path
    else:
        # Upload masks to B2 CDN (but keep local copy for simple_effects)
        self.update_state(state='PROCESSING', meta={'status': 'Uploading masks to B2', 'progress': 90})
        masks_url = upload_masks_to_b2(masks_dir_wsl, video_path)
        masks_dir_result = masks_dir_wsl  # Keep local path for simple_effects to use

    # Cleanup temporary files (but NOT video - simple_effects needs it)
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        print(f"[CLEANUP] Removed frames directory: {temp_frames_dir}")

    # Update Redis directly (fixes race condition - export may be called before /status is polled)
    if job_id:
        try:
            import redis
            redis_client = redis_from_url(os.environ.get('REDIS_URL'), decode_responses=True)
            redis_client.hset(f"objrem:{job_id}", mapping={
                'masks_dir': masks_dir_result,
                'masks_url': masks_url or '',
                'video_path': video_path_wsl,  # Store local video path
                'status': 'completed'
            })
            print(f"[REDIS] Updated objrem:{job_id} with masks_dir={masks_dir_result}, video_path={video_path_wsl}")
        except Exception as e:
            print(f"[REDIS] Failed to update job: {e}")

    return {
        'status': 'success',
        'masks_url': masks_url,
        'masks_dir': masks_dir_result,  # Local path when SKIP_B2_UPLOAD=1
        'masks_saved': int(masks_saved),
        'total_frames': int(total_frames),
        'fps': float(fps),  # tracking fps (10 or full), used for expansion on Windows
    }


def upload_masks_to_b2(masks_dir, video_path):
    """
    Zip and upload masks to B2 CDN, return CDN URL.
    Cleans up local masks after upload.
    """
    from b2sdk.v2 import B2Api, InMemoryAccountInfo

    B2_KEY_ID = os.getenv('B2_KEY_ID')
    B2_APP_KEY = os.getenv('B2_APP_KEY')
    B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
    B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

    if not B2_KEY_ID or not B2_APP_KEY:
        print("[B2] Warning: B2 credentials not set, returning local path")
        return None

    # Extract video_id from path for naming
    video_basename = os.path.basename(video_path)
    video_id = os.path.splitext(video_basename)[0]
    # Remove any suffix like _tracking_10fps
    if '_tracking_' in video_id:
        video_id = video_id.split('_tracking_')[0]

    # Zip masks
    zip_path = f"/tmp/{video_id}_masks.zip"
    mask_files = sorted(glob.glob(f"{masks_dir}/*.png"))
    print(f"[B2] Zipping {len(mask_files)} masks...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mask_file in mask_files:
            zf.write(mask_file, os.path.basename(mask_file))

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[B2] Zip created: {zip_size_mb:.1f} MB")

    # Upload to B2
    print(f"[B2] Uploading to {B2_BUCKET}...")
    upload_start = time.time()

    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
    bucket = b2_api.get_bucket_by_name(B2_BUCKET)

    timestamp = int(time.time())
    remote_path = f"masks/{timestamp}_{video_id}_masks.zip"
    bucket.upload_local_file(local_file=zip_path, file_name=remote_path)

    masks_url = f"{B2_CDN_URL}/{remote_path}"
    upload_time = time.time() - upload_start
    print(f"[B2] Upload complete in {upload_time:.1f}s: {masks_url}")

    # DON'T delete local masks - simple_effects will use them and clean up
    os.remove(zip_path)
    print(f"[B2] Keeping local masks at: {masks_dir}")

    return masks_url


def to_wsl_path(p: str) -> str:
    """Convert Windows path to WSL path."""
    if not isinstance(p, str):
        return p
    if p.startswith('/mnt/') or p.startswith('/'):
        return p
    if len(p) >= 2 and p[1] == ':' and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].replace('\\', '/')
        rest = rest.lstrip('/')
        return f"/mnt/{drive}/{rest}"
    return p.replace('\\', '/')


@celery.task(name='yolo.generate_masks', bind=True)
def generate_masks_yolo(self, video_path, masks_dir, confidence_threshold=0.3, padding=30, api_base=None):
    """
    Generate masks using YOLO detection (automatic, no user clicks).

    Uses TensorRT engine if available (20-30x faster), falls back to PyTorch.

    Args:
        video_path: Path to video file
        masks_dir: Output directory for masks
        confidence_threshold: YOLO confidence threshold
        padding: Padding around detections
        api_base: Base URL for downloading video if path is remote

    Returns:
        dict with masks_url, total_frames, etc.
    """
    task_id = self.request.id

    # Acquire GPU lock (waits if ProPainter is using GPU)
    with _gpu_coord.gpu_exclusive(task_id):
        import cv2
        import numpy as np
        import requests

        video_path_wsl = to_wsl_path(video_path)
        masks_dir_wsl = to_wsl_path(masks_dir)
        os.makedirs(masks_dir_wsl, exist_ok=True)

        # If video is a Railway path (/data/...), download it from api_base
        is_railway_path = video_path.startswith('/data/') or video_path_wsl.startswith('/data/')
        if (is_railway_path or not os.path.exists(video_path_wsl)) and api_base:
            filename = os.path.basename(video_path)
            download_url = f"{api_base}/uploads/{filename}"
            local_video = f"/tmp/{filename}"
            print(f"[YOLO] Downloading video from {download_url}...")
            self.update_state(state='PROCESSING', meta={'status': 'Downloading video', 'progress': 2})
            r = requests.get(download_url, timeout=300)
            r.raise_for_status()
            with open(local_video, 'wb') as f:
                f.write(r.content)
            video_path_wsl = local_video
            print(f"[YOLO] Downloaded {len(r.content) / 1024 / 1024:.1f} MB to {local_video}")

        self.update_state(state='PROCESSING', meta={'status': 'Loading YOLO model', 'progress': 5})

        # Check video dimensions first (TensorRT engine needs 640x640, fails on portrait/landscape)
        cap_check = cv2.VideoCapture(video_path_wsl)
        vid_w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_check.release()
        # Use PyTorch model (TRT batch=64 engine doesn't work with single-frame inference)
        from ultralytics import YOLO

        pt_model_path = '/mnt/d/watermarkz/runs/detect/sora_watermark_v2/weights/best.pt'

        if not os.path.exists(pt_model_path):
            raise RuntimeError(f"YOLO model not found: {pt_model_path}")
        model = YOLO(pt_model_path)
        print(f"[YOLO] Loaded PyTorch model: {pt_model_path}")

        # Open video
        self.update_state(state='PROCESSING', meta={'status': 'Opening video', 'progress': 10})
        cap = cv2.VideoCapture(video_path_wsl)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path_wsl}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[YOLO] Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")

        # Process frames
        masks_saved = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress every 10 frames
            if frame_idx % 10 == 0:
                progress = 10 + int(80 * frame_idx / total_frames)
                self.update_state(state='PROCESSING', meta={
                    'status': f'Detecting frame {frame_idx}/{total_frames}',
                    'progress': progress
                })

            # Run YOLO detection
            results = model(frame, conf=confidence_threshold, device='cuda', verbose=False)

            # Create mask from detections
            mask = np.zeros((height, width), dtype=np.uint8)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Add padding
                    x1 = max(0, int(x1) - padding)
                    y1 = max(0, int(y1) - padding)
                    x2 = min(width, int(x2) + padding)
                    y2 = min(height, int(y2) + padding)

                    # Fill mask
                    mask[y1:y2, x1:x2] = 255

            # Save mask
            mask_path = os.path.join(masks_dir_wsl, f"{frame_idx:05d}.png")
            cv2.imwrite(mask_path, mask)
            masks_saved += 1
            frame_idx += 1

        cap.release()
        print(f"[YOLO] Generated {masks_saved} masks")

        # Cleanup VRAM
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Upload to B2
        self.update_state(state='PROCESSING', meta={'status': 'Uploading masks to B2', 'progress': 90})
        masks_url = upload_masks_to_b2(masks_dir_wsl, video_path)

        # Cleanup temporary files
        frames_dir = "/tmp/sam2_frames"
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"[CLEANUP] Removed frames directory: {frames_dir}")

        # Clean downloaded video if it was in /tmp/
        if video_path_wsl.startswith('/tmp/') and os.path.exists(video_path_wsl):
            try:
                os.remove(video_path_wsl)
                print(f"[CLEANUP] Removed temporary video: {video_path_wsl}")
            except Exception as e:
                print(f"[CLEANUP] Could not remove video: {e}")

        return {
            'status': 'success',
            'masks_url': masks_url,
            'masks_dir': None,
            'masks_saved': int(masks_saved),
            'total_frames': int(total_frames),
            'fps': float(fps),
            'mode': 'yolo',  # Tell receiver this was YOLO mode (use 4 workers)
        }


@celery.task(name='sam2.apply_simple_effects', bind=True)
def apply_simple_effects(self, video_url, masks_url, operation='keep_object', background='color', bg_color='#00FF00', blur_amount=25, dilation=4, output_format='mp4', masks_dir_local=None, video_path_local=None):
    """
    REVOLUTIONARY GPU-ACCELERATED video effects pipeline.
    ALL processing on GPU - decode, blend, blur, encode. BLAZING FAST!
    """
    task_id = self.request.id

    # Acquire GPU lock (waits if ProPainter is using GPU)
    with _gpu_coord.gpu_exclusive(task_id):
        return _apply_simple_effects_impl(self, video_url, masks_url, operation, background, bg_color, blur_amount, dilation, output_format, masks_dir_local, video_path_local)


def _apply_simple_effects_impl(self, video_url, masks_url, operation='keep_object', background='color', bg_color='#00FF00', blur_amount=25, dilation=4, output_format='mp4', masks_dir_local=None, video_path_local=None):
    """Actual implementation of apply_simple_effects (wrapped with GPU lock)."""
    import cv2
    import numpy as np
    import requests
    import subprocess
    import torch
    import torch.nn.functional as F

    print(f"[GPU-FX] 🚀 REVOLUTIONARY GPU Pipeline Starting!")
    print(f"[GPU-FX] Operation: {operation}, bg={background}, color={bg_color}")

    # Derive local masks path from task_id
    task_id = self.request.id
    if task_id and task_id.startswith('simplefx_'):
        job_id = task_id.replace('simplefx_', '')
        derived_masks_dir = f"/tmp/sam2_masks/{job_id}"
        if os.path.exists(derived_masks_dir):
            masks_dir_local = derived_masks_dir
            print(f"[GPU-FX] Using local masks: {masks_dir_local}")

    # Skip video download for masks-only mode (not needed)
    video_path = None
    if output_format != 'masks':
        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Loading video...'})

        # Check for local video first (avoids re-downloading)
        if video_path_local and os.path.exists(video_path_local):
            video_path = video_path_local
            print(f"[GPU-FX] Using LOCAL video: {video_path}")
        else:
            # Download video
            video_path = "/tmp/simple_fx_video.mp4"
            print(f"[GPU-FX] Downloading video...")
            r = requests.get(video_url, timeout=300)
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                f.write(r.content)
            print(f"[GPU-FX] Downloaded {len(r.content) / 1024 / 1024:.1f} MB")

        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    # Load masks
    self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Loading masks to GPU...'})

    if masks_dir_local and os.path.exists(masks_dir_local):
        masks_dir = masks_dir_local
        mask_files = sorted(glob.glob(f"{masks_dir}/*.png"))
        print(f"[GPU-FX] LOCAL masks: {len(mask_files)} files")
    else:
        masks_dir = "/tmp/simple_fx_masks"
        if os.path.exists(masks_dir):
            shutil.rmtree(masks_dir)
        os.makedirs(masks_dir)
        print(f"[GPU-FX] Downloading masks from B2...")
        r = requests.get(masks_url, timeout=300)
        r.raise_for_status()
        zip_path = "/tmp/simple_fx_masks.zip"
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(masks_dir)
        os.remove(zip_path)
        mask_files = sorted(glob.glob(f"{masks_dir}/*.png"))
        print(f"[GPU-FX] Extracted {len(mask_files)} masks")

    # MASKS ONLY mode - just zip up masks, no video processing
    if output_format == 'masks':
        output_path = f"/tmp/masks_only_{int(time.time())}.zip"
        print(f"[GPU-FX] Masks-only mode - zipping {len(mask_files)} masks...")
        self.update_state(state='PROCESSING', meta={'progress': 50, 'status': 'Zipping masks...'})
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for mask_file in mask_files:
                zf.write(mask_file, os.path.basename(mask_file))
        print(f"[GPU-FX] Masks ZIP created: {output_path}")

        # Upload to B2
        self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Uploading masks...'})
        output_url = upload_video_to_b2(output_path)
        os.remove(output_path)
        print(f"[GPU-FX] Masks uploaded: {output_url}")
        return {'status': 'success', 'cdn_url': output_url}

    # Parse color to RGB
    bg_color_hex = bg_color.lstrip('#')
    color_rgb = [int(bg_color_hex[i:i+2], 16) for i in (0, 2, 4)]

    # Determine output format and setup encoder
    print(f"[GPU-FX] Output format: {output_format}")

    # PNG sequence mode - no ffmpeg encoder, save frames directly
    png_output_dir = None
    ffmpeg_encode = None

    if output_format == 'png':
        # PNG sequence mode - save RGBA frames with transparency
        png_output_dir = "/tmp/simple_fx_png_output"
        if os.path.exists(png_output_dir):
            shutil.rmtree(png_output_dir)
        os.makedirs(png_output_dir)
        output_path = f"/tmp/simple_fx_output_{int(time.time())}.zip"
        print(f"[GPU-FX] PNG sequence mode - saving frames to {png_output_dir}")
    elif output_format == 'webm':
        # WebM with VP9 + alpha (ONLY codec that supports alpha in browsers)
        # Note: VP9 is CPU-only, but AV1 NVENC strips alpha channel
        output_path = "/tmp/simple_fx_output.webm"
        print(f"[GPU-FX] WebM mode with VP9 alpha encoding")
        ffmpeg_encode = subprocess.Popen([
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgba',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',   # VP9 with alpha
            '-crf', '23',
            '-b:v', '0',
            '-auto-alt-ref', '0',     # Required for alpha channel
            '-cpu-used', '4',         # Faster encoding (0-5, higher=faster)
            '-threads', '8',          # Multi-threaded encoding
            '-row-mt', '1',           # Row-based multithreading
            output_path
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        # MP4 mode (default) - NVENC h264
        output_path = "/tmp/simple_fx_output.mp4"
        print(f"[GPU-FX] MP4 mode - starting NVENC encoder pipeline...")
        ffmpeg_encode = subprocess.Popen([
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-i', video_path,  # For audio
            '-map', '0:v',
            '-map', '1:a?',
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',
            '-cq', '18',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # === GPU STREAMING: Process ONE frame at a time (like CPU, but fast!) ===
    # Pre-allocate reusable GPU buffers (~25MB total, not 12GB!)
    device = torch.device('cuda')
    bg_tensor = torch.tensor(color_rgb, dtype=torch.float32, device=device).view(1, 1, 3)

    # Pre-compute blur kernel if needed
    blur_kernel = None
    if background == 'blur' or operation in ['blur_inside', 'blur_outside']:
        blur_k = blur_amount * 2 + 1
        blur_kernel = torch.ones(3, 1, blur_k, blur_k, device=device) / (blur_k * blur_k)

    # === PRELOAD ALL MASKS TO GPU (eliminates disk I/O in loop) ===
    print(f"[GPU-FX] Preloading {len(mask_files)} masks to GPU...")
    all_masks_gpu = []
    for mf in mask_files:
        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((height, width), dtype=np.uint8)
        if m.shape[:2] != (height, width):
            m = cv2.resize(m, (width, height))
        all_masks_gpu.append(torch.from_numpy(m).to(device).float() / 255.0)
    print(f"[GPU-FX] ✅ Masks preloaded to GPU")

    print(f"[GPU-FX] 🚀 STREAMING: Processing {total_frames} frames with NVDEC GPU decoding...")

    # === NVDEC GPU VIDEO DECODING (replaces cv2.VideoCapture CPU decoding) ===
    # Try NVDEC first, fallback to CPU if GPU decoding fails
    ffmpeg_decode = subprocess.Popen([
        'ffmpeg',
        '-hwaccel', 'cuda',           # Use GPU for decoding
        '-i', video_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        'pipe:1'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = width * height * 3
    frame_idx = 0

    while True:
        # Read frame from NVDEC decoder
        raw_frame = ffmpeg_decode.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break

        # Already in RGB format from FFmpeg
        frame_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)

        # Get preloaded mask from GPU (no disk I/O!)
        if frame_idx < len(all_masks_gpu):
            mask_gpu = all_masks_gpu[frame_idx]
        else:
            mask_gpu = torch.zeros(height, width, device=device)

        # === GPU: Transfer single frame (~6MB) ===
        frame_gpu = torch.from_numpy(frame_rgb).to(device).float()  # [H, W, 3]

        # Apply dilation on GPU
        if dilation > 0:
            mask_4d = mask_gpu.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            for _ in range(dilation):
                mask_4d = F.max_pool2d(mask_4d, 3, stride=1, padding=1)
            mask_gpu = mask_4d.squeeze()  # [H, W]

        # Create 3-channel mask
        mask_3ch = mask_gpu.unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]
        inv_mask = 1.0 - mask_3ch

        # Apply operation
        if operation == 'keep_object':
            if background == 'blur':
                frame_4d = frame_gpu.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                blurred = F.conv2d(F.pad(frame_4d, [blur_amount]*4, mode='reflect'),
                                   blur_kernel, padding=0, groups=3)
                blurred = blurred.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
                result = frame_gpu * mask_3ch + blurred * inv_mask
            elif background == 'transparent':
                # Don't composite RGB - alpha channel handles transparency
                result = frame_gpu
            else:  # color
                result = frame_gpu * mask_3ch + bg_tensor.expand(height, width, 3) * inv_mask

        elif operation == 'remove_object':
            if background == 'blur':
                frame_4d = frame_gpu.permute(2, 0, 1).unsqueeze(0)
                blurred = F.conv2d(F.pad(frame_4d, [blur_amount]*4, mode='reflect'),
                                   blur_kernel, padding=0, groups=3)
                blurred = blurred.squeeze(0).permute(1, 2, 0)
                result = frame_gpu * inv_mask + blurred * mask_3ch
            elif background == 'transparent':
                # Don't composite RGB - alpha channel handles transparency
                result = frame_gpu
            else:  # color
                result = frame_gpu * inv_mask + bg_tensor.expand(height, width, 3) * mask_3ch

        elif operation == 'blur_inside':
            frame_4d = frame_gpu.permute(2, 0, 1).unsqueeze(0)
            blurred = F.conv2d(F.pad(frame_4d, [blur_amount]*4, mode='reflect'),
                               blur_kernel, padding=0, groups=3)
            blurred = blurred.squeeze(0).permute(1, 2, 0)
            result = blurred * mask_3ch + frame_gpu * inv_mask

        elif operation == 'blur_outside':
            frame_4d = frame_gpu.permute(2, 0, 1).unsqueeze(0)
            blurred = F.conv2d(F.pad(frame_4d, [blur_amount]*4, mode='reflect'),
                               blur_kernel, padding=0, groups=3)
            blurred = blurred.squeeze(0).permute(1, 2, 0)
            result = frame_gpu * mask_3ch + blurred * inv_mask

        elif operation == 'fill_inside':
            result = frame_gpu * inv_mask + bg_tensor.expand(height, width, 3) * mask_3ch

        elif operation == 'fill_outside':
            if background == 'transparent':
                # Don't composite RGB - alpha channel handles transparency
                result = frame_gpu
            else:
                result = frame_gpu * mask_3ch + bg_tensor.expand(height, width, 3) * inv_mask

        else:
            result = frame_gpu

        # === Output frame based on format ===
        if output_format == 'png':
            # PNG sequence: Save RGBA frame with transparency
            # For keep_object: object is opaque, background is transparent
            # For remove_object: object area is transparent
            result_cpu = result.clamp(0, 255).byte().cpu().numpy()

            # Create alpha channel from mask
            if operation == 'keep_object':
                # Object visible, background transparent
                alpha = (mask_gpu * 255).clamp(0, 255).byte().cpu().numpy()
            else:
                # Background visible, object transparent
                alpha = ((1.0 - mask_gpu) * 255).clamp(0, 255).byte().cpu().numpy()

            # Combine RGB + Alpha into RGBA
            rgba_frame = np.dstack([result_cpu, alpha])

            # Save as PNG (OpenCV expects BGRA)
            bgra_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGRA)
            png_path = os.path.join(png_output_dir, f"{frame_idx:05d}.png")
            cv2.imwrite(png_path, bgra_frame)

        elif output_format == 'webm':
            # WebM: Send RGBA to VP9 encoder with alpha
            result_cpu = result.clamp(0, 255).byte().cpu().numpy()

            if operation == 'keep_object':
                alpha = (mask_gpu * 255).clamp(0, 255).byte().cpu().numpy()
            else:
                alpha = ((1.0 - mask_gpu) * 255).clamp(0, 255).byte().cpu().numpy()

            rgba_frame = np.dstack([result_cpu, alpha])
            ffmpeg_encode.stdin.write(rgba_frame.tobytes())

        else:
            # MP4: Pipe RGB to NVENC encoder
            result_cpu = result.clamp(0, 255).byte().cpu().numpy()
            ffmpeg_encode.stdin.write(result_cpu.tobytes())

        frame_idx += 1

        # Progress update every 100 frames
        if frame_idx % 100 == 0:
            progress = 15 + int(75 * frame_idx / total_frames)
            self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'Frame {frame_idx}/{total_frames}'})
            print(f"[GPU-FX] Frame {frame_idx}/{total_frames}")

    # Cleanup NVDEC decoder
    ffmpeg_decode.stdout.close()
    ffmpeg_decode.wait()
    print(f"[GPU-FX] Processed {frame_idx} frames with GPU streaming (NVDEC + mask preload)")

    # Free preloaded masks from GPU memory
    del all_masks_gpu
    torch.cuda.empty_cache()

    # Close encoder / finalize output
    if output_format == 'png':
        # Zip the PNG sequence
        print(f"[GPU-FX] Zipping {frame_idx} PNG frames...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Zipping PNG sequence...'})
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for png_file in sorted(glob.glob(f"{png_output_dir}/*.png")):
                zf.write(png_file, os.path.basename(png_file))
        shutil.rmtree(png_output_dir)
        print(f"[GPU-FX] Created PNG zip: {output_path}")
    else:
        # Close ffmpeg encoder
        ffmpeg_encode.stdin.close()
        return_code = ffmpeg_encode.wait()
        if return_code != 0:
            stderr = ffmpeg_encode.stderr.read().decode() if ffmpeg_encode.stderr else ''
            print(f"[GPU-FX] FFmpeg error (code {return_code}): {stderr[:500]}")
            raise RuntimeError(f"FFmpeg encoding failed: {stderr[:200]}")
        print(f"[GPU-FX] Encoded {frame_idx} frames")

    self.update_state(state='PROCESSING', meta={'progress': 92, 'status': 'Uploading to CDN...'})

    # Upload to B2
    output_url = upload_video_to_b2(output_path)

    # Cleanup
    shutil.rmtree(masks_dir, ignore_errors=True)
    # Only remove video if it was downloaded to /tmp (not if it was a local path we received)
    if video_path.startswith('/tmp/simple_fx_video'):
        try:
            os.remove(video_path)
        except Exception:
            pass
    try:
        os.remove(output_path)
    except Exception:
        pass

    print(f"[GPU-FX] 🎉 COMPLETE! CDN URL: {output_url}")

    return {
        'status': 'success',
        'cdn_url': output_url,
    }


def upload_video_to_b2(file_path):
    """Upload video/zip to B2 and return CDN URL."""
    from b2sdk.v2 import B2Api, InMemoryAccountInfo

    B2_KEY_ID = os.getenv('B2_KEY_ID')
    B2_APP_KEY = os.getenv('B2_APP_KEY')
    B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
    B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

    if not B2_KEY_ID or not B2_APP_KEY:
        print("[B2] Warning: B2 credentials not set")
        return None

    # Get file extension from original file
    basename = os.path.basename(file_path)
    if basename.endswith('.zip'):
        ext = '.zip'
        folder = 'exports'
    elif basename.endswith('.webm'):
        ext = '.webm'
        folder = 'videos'
    else:
        ext = '.mp4'
        folder = 'videos'

    # Remove extension to get ID
    file_id = basename.replace('.mp4', '').replace('.webm', '').replace('.zip', '')
    timestamp = int(time.time())
    b2_filename = f"{folder}/{timestamp}_{file_id}{ext}"

    print(f"[B2] Uploading {ext} to {B2_BUCKET}...")
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account('production', B2_KEY_ID, B2_APP_KEY)
    bucket = b2_api.get_bucket_by_name(B2_BUCKET)

    with open(file_path, 'rb') as f:
        bucket.upload_bytes(f.read(), b2_filename)

    file_url = f"{B2_CDN_URL}/{b2_filename}"
    print(f"[B2] Upload complete: {file_url}")

    return file_url
