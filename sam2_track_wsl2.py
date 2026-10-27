#!/usr/bin/env python3
"""
SAM2 Tracking Script for WSL2 - Called from Windows Celery worker
Usage:
  Point mode:  python sam2_track_wsl2.py <video_path> --point x,y <output_masks_dir>
  Bbox mode:   python sam2_track_wsl2.py <video_path> --bbox x1,y1,x2,y2 <output_masks_dir>

Uses HYBRID approach:
- Frame 0: TensorRT FP16 (instant)
- Frames 1-N: PyTorch torch.compile() + BFloat16
"""

import os
import sys
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
import argparse
import subprocess

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
SAM2_PATH = Path(__file__).parent / "segment-anything-2"
sys.path.insert(0, str(SAM2_PATH))

# Import SAM2 PyTorch
from sam2.build_sam import build_sam2_video_predictor

# Try to import TensorRT predictor (optional for frame 0)
try:
    from sam2_trt_predictor import SAM2TensorRTPredictor
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("[SAM2-WSL2] TensorRT predictor not available, using PyTorch for all frames")

# =============================================================================
# RESOLUTION CONFIGURATION - Single source of truth
# =============================================================================
TARGET_RESOLUTION = 480  # Change this ONE value to switch between 360p, 480p, 720p, 1080p

# Try to import DALI for GPU-direct video decoding
try:
    from nvidia.dali import pipeline_def, fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    HAS_DALI = True
    print("[SAM2-WSL2] NVIDIA DALI available - GPU-direct video streaming enabled!")
except ImportError:
    HAS_DALI = False
    print("[SAM2-WSL2] DALI not available, using CPU frame loading")

# Try to import ReID tracker for identity verification
try:
    from sam2_reid_tracker import (
        ReIDVerifier, USE_REID_VERIFICATION, VERIFY_EVERY_N_FRAMES,
        classify_tracking_state, get_user_click_for_recovery
    )
    HAS_REID = True
    print(f"[SAM2-WSL2] ReID verification available! (verify every {VERIFY_EVERY_N_FRAMES} frames)")
except ImportError:
    HAS_REID = False
    USE_REID_VERIFICATION = False
    print("[SAM2-WSL2] ReID tracker not available, identity verification disabled")


# =============================================================================
# REVOLUTIONARY COMPRESSION CLASSES
# =============================================================================

def extract_to_h265(video_path, output_path, target_fps=None, max_height=TARGET_RESOLUTION):
    """
    Extract video to H.265 for GPU-accelerated decoding.
    Way smaller than JPG frames: ~70MB vs 4GB+ for 2000 frames.
    """
    import shlex

    # Get source video info
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate resize
    if orig_height > max_height:
        scale = max_height / orig_height
        new_width = int(orig_width * scale) // 2 * 2  # Ensure even
        new_height = max_height // 2 * 2
    else:
        new_width, new_height = orig_width, orig_height

    fps_filter = f",fps={target_fps}" if target_fps else ""

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-c:v', 'hevc_nvenc', '-cq', '18', '-preset', 'p4',
        '-vf', f'scale={new_width}:{new_height}{fps_filter}',
        '-pix_fmt', 'yuv420p',
        '-an',  # No audio
        output_path
    ]

    print(f"[H.265] Converting {orig_width}x{orig_height} -> {new_width}x{new_height}")
    subprocess.run(cmd, check=True, capture_output=True)

    size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"[H.265] Created {size_mb:.1f}MB video (vs ~{total_frames * 0.2:.0f}MB in JPGs)")

    return total_frames, orig_fps, orig_width, orig_height, new_width, new_height


class DALIFrameStreamer:
    """
    GPU-direct video streaming using NVIDIA DALI with RANDOM ACCESS.
    Decodes H.265 directly on GPU - ZERO CPU RAM usage!

    Supports __getitem__ for random frame access needed by SAM2.
    """

    def __init__(self, video_path, device_id=0, image_size=1024, max_height=TARGET_RESOLUTION):
        if not HAS_DALI:
            raise RuntimeError("DALI not installed! pip install nvidia-dali-cuda120")

        self.device_id = device_id
        self.image_size = image_size
        self.max_height = max_height

        # Convert to H.265 for optimal DALI decoding
        self.h265_path = self._convert_to_h265(video_path)
        self.video_path = self.h265_path

        # SAM2's normalization constants
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(device_id)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(device_id)

        # Build pipeline
        self._build_pipeline()

        # Cache for frames (optional - for repeated access)
        self._cache = {}
        self._cache_enabled = False

    def _convert_to_h265(self, input_path):
        """Convert video to DALI-friendly H.265 stream"""
        import tempfile

        # Get video info first
        cap = cv2.VideoCapture(input_path)
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        # Calculate resize
        if orig_height > self.max_height:
            scale = self.max_height / orig_height
            new_width = int(orig_width * scale) // 2 * 2
            new_height = self.max_height // 2 * 2
        else:
            new_width, new_height = orig_width, orig_height

        self.video_height = new_height
        self.video_width = new_width

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()

        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'hevc_nvenc', '-cq', '18', '-preset', 'p4',
            '-vf', f'scale={new_width}:{new_height}',
            '-pix_fmt', 'yuv420p',
            '-an',
            temp_path
        ]

        print(f"[DALI] Converting to H.265: {orig_width}x{orig_height} -> {new_width}x{new_height}")
        subprocess.run(cmd, capture_output=True, check=True)

        size_mb = os.path.getsize(temp_path) / (1024**2)
        print(f"[DALI] H.265 video: {size_mb:.1f}MB")

        return temp_path

    def _build_pipeline(self):
        @pipeline_def
        def video_pipe():
            video = fn.readers.video(
                device="gpu",
                filenames=[self.video_path],
                sequence_length=1,
                shard_id=0,
                num_shards=1,
                random_shuffle=False,
                initial_fill=2,
                image_type=types.RGB,
                dtype=types.FLOAT,
                normalized=False,
                name="video_reader"
            )
            # Resize to SAM2's internal size
            video = fn.resize(video, size=[self.image_size, self.image_size])
            # Normalize to 0-1
            video = video / 255.0
            return video

        self.pipe = video_pipe(
            batch_size=1,
            num_threads=2,
            device_id=self.device_id,
            prefetch_queue_depth=1  # Minimal VRAM
        )
        self.pipe.build()

        # Get total frames
        self._length = self.pipe.reader_meta("video_reader")["epoch_size"]
        self._current_idx = 0
        print(f"[DALI] Ready: {self._length} frames, VRAM per frame: ~{self.image_size*self.image_size*3*4/1024/1024:.1f}MB")

    def __len__(self):
        return self._length

    def __getitem__(self, frame_idx):
        """Random access to frame - used by SAM2 during propagation"""
        if frame_idx < 0 or frame_idx >= self._length:
            raise IndexError(f"Frame {frame_idx} out of bounds (0-{self._length-1})")

        # Check cache
        if self._cache_enabled and frame_idx in self._cache:
            return self._cache[frame_idx]

        # Seek to frame by resetting and skipping
        # This is necessary because DALI video reader doesn't support random seek directly
        if frame_idx < self._current_idx:
            self.pipe.reset()
            self._current_idx = 0

        # Skip to target frame
        while self._current_idx < frame_idx:
            self.pipe.run()
            self._current_idx += 1

        # Get the target frame
        output = self.pipe.run()
        self._current_idx += 1

        frame = output[0].as_tensor()

        # Convert to PyTorch tensor [B, T, H, W, C] -> [C, H, W]
        frame_torch = torch.from_dlpack(frame).squeeze(0).squeeze(0).permute(2, 0, 1)

        # Apply SAM2 normalization
        frame_torch = frame_torch.unsqueeze(0)  # Add batch dim for normalization
        frame_torch = (frame_torch - self.img_mean) / self.img_std
        frame_torch = frame_torch.squeeze(0)  # Remove batch dim

        # Cache if enabled
        if self._cache_enabled:
            self._cache[frame_idx] = frame_torch

        return frame_torch  # [C, H, W]

    def __iter__(self):
        """Sequential iteration (more efficient than random access)"""
        self.pipe.reset()
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self._length:
            raise StopIteration
        return self[self._current_idx]

    def preload_all(self):
        """Preload all frames to GPU (uses VRAM but fast access)"""
        print(f"[DALI] Preloading {self._length} frames to VRAM...")
        self._cache_enabled = True
        self.pipe.reset()
        self._current_idx = 0

        frames = []
        for i in tqdm(range(self._length), desc="DALI preload"):
            frames.append(self[i])

        # Stack into single tensor
        self._frames_tensor = torch.stack(frames)
        print(f"[DALI] Preloaded: {self._frames_tensor.shape}, VRAM: {self._frames_tensor.numel()*4/1024/1024:.1f}MB")
        return self._frames_tensor

    def cleanup(self):
        """Delete temp H.265 file"""
        if hasattr(self, 'h265_path') and os.path.exists(self.h265_path):
            os.unlink(self.h265_path)
            print(f"[DALI] Cleaned up temp file")


class DALIVideoLoader(DALIFrameStreamer):
    """Alias for backward compatibility"""
    pass


class DALIChunkedStreamer:
    """
    Async chunked GPU streaming with queue-based prefetching:
    - DALI loads raw uint8 frames (simple, reliable)
    - Background thread continuously prefetches chunks into queue
    - PyTorch does resize+normalize when accessing frames
    - Zero lag at chunk boundaries
    """

    def __init__(self, video_path, chunk_size=500, device_id=0, image_size=1024, max_height=TARGET_RESOLUTION):
        import queue
        import threading

        if not HAS_DALI:
            raise RuntimeError("DALI not installed! pip install nvidia-dali-cuda120")

        self.chunk_size = chunk_size
        self.device_id = device_id
        self.image_size = image_size
        self.max_height = max_height

        # SAM2 normalization constants
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(device_id)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(device_id)

        # Convert to H.265 for GPU decoding
        self.h265_path = self._convert_to_h265(video_path)

        # Build simple DALI pipeline (raw uint8 only)
        self._build_pipeline()

        # Chunk state
        self.current_chunk = None
        self.current_chunk_start = -1

        # Chunk cache for backward pass support (SAM2 does backward first!)
        # Dict: chunk_start -> tensor
        self.chunk_cache = {}
        self.max_cached_chunks = 3  # Keep last 3 chunks in VRAM (~4GB)

        # Async loading queue
        self.load_queue = queue.Queue(maxsize=2)
        # Set by the loader thread when it dies. The consumer raises this
        # instead of the bare queue.Empty it would otherwise see 30s later.
        self.loader_error = None
        self.stop_event = threading.Event()
        self.loading_complete = threading.Event()

        # Start background loader
        self._start_background_loader()

        chunk_mb = chunk_size * self.video_height * self.video_width * 3 / 1024**2
        print(f"[DALI-ASYNC] {self._length} frames, {chunk_size}-frame chunks, ~{chunk_mb:.0f}MB raw per chunk")
        print(f"[DALI-ASYNC] Background prefetcher started")

    def _convert_to_h265(self, input_path):
        """Convert video to DALI-friendly H.265 stream. Returns (path, orig_frame_count)"""
        import tempfile
        import requests

        # If input is URL, download first (ffmpeg has issues streaming from HTTPS)
        local_input = input_path
        downloaded_file = None
        if input_path.startswith('http://') or input_path.startswith('https://'):
            print(f"[DALI-ASYNC] Downloading video from URL...")
            downloaded_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            r = requests.get(input_path, timeout=300)
            r.raise_for_status()
            downloaded_file.write(r.content)
            downloaded_file.close()
            local_input = downloaded_file.name
            print(f"[DALI-ASYNC] Downloaded {len(r.content) / 1024 / 1024:.1f}MB to {local_input}")

        cap = cv2.VideoCapture(local_input)
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # ORIGINAL frame count (not converted!)
        cap.release()

        if orig_height > self.max_height:
            scale = self.max_height / orig_height
            new_width = int(orig_width * scale) // 2 * 2
            new_height = self.max_height // 2 * 2
        else:
            new_width, new_height = orig_width, orig_height

        self.video_height = new_height
        self.video_width = new_width

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()

        cmd = [
            'ffmpeg', '-y', '-i', local_input,
            '-c:v', 'hevc_nvenc', '-cq', '18', '-preset', 'p4',
            '-vf', f'scale={new_width}:{new_height}',
            '-pix_fmt', 'yuv420p',
            '-an',
            temp_path
        ]

        print(f"[DALI-ASYNC] Converting to H.265: {orig_width}x{orig_height} -> {new_width}x{new_height}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up downloaded file (H.265 used for both DALI and ffmpeg fallback)
        if downloaded_file:
            try:
                os.remove(downloaded_file.name)
            except:
                pass

        if result.returncode != 0:
            print(f"[DALI-ASYNC] FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg H.265 conversion failed: {result.stderr}")

        size_mb = os.path.getsize(temp_path) / (1024**2)
        print(f"[DALI-ASYNC] H.265 video: {size_mb:.1f}MB")

        return temp_path

    def _build_pipeline(self):
        """Build simple DALI pipeline - raw uint8 only"""
        cap = cv2.VideoCapture(self.h265_path)
        self._length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Clamp chunk_size to video length (DALI requires sequence_length <= video frames)
        actual_chunk_size = min(self.chunk_size, self._length)
        if actual_chunk_size < self.chunk_size:
            print(f"[DALI] Video has {self._length} frames, reducing chunk size to {actual_chunk_size}")
            self.chunk_size = actual_chunk_size

        @pipeline_def
        def video_pipe():
            # DEBUG: Log what DALI returns, find the real issue
            video, frame_num = fn.experimental.readers.video(
                device="gpu",
                filenames=[self.h265_path],
                sequence_length=self.chunk_size,
                step=self.chunk_size,  # Explicit step
                stride=1,
                pad_mode='none',
                read_ahead=False,
                enable_frame_num=True,
                random_shuffle=False,
                image_type=types.RGB,
                shard_id=0,
                num_shards=1,
                name="video_reader",
            )
            return video, frame_num

        self.pipe = video_pipe(
            batch_size=1,
            num_threads=1,  # Single thread to prevent race conditions
            device_id=self.device_id,
            prefetch_queue_depth=1,  # Minimal prefetch to prevent wraparound
        )
        self.pipe.build()

    def _start_background_loader(self):
        """Spawn daemon thread that continuously prefetches chunks using DALI"""
        import threading
        import traceback

        def loader_worker():
            try:
                chunk_num = 0
                total_frames_read = 0
                num_chunks = (self._length + self.chunk_size - 1) // self.chunk_size

                # DALI breaks on last 2 chunks - use OpenCV for those
                dali_chunk_limit = max(0, num_chunks - 2)
                print(f"[DALI-PREFETCH] {num_chunks} chunks total, DALI for 0-{dali_chunk_limit-1}, OpenCV for last 2")

                # === PHASE 1: DALI for chunks 0 to (num_chunks - 3) ===
                while not self.stop_event.is_set() and chunk_num < dali_chunk_limit:
                    chunk_start = chunk_num * self.chunk_size
                    chunk_end = min(chunk_start + self.chunk_size, self._length)
                    frames_needed = chunk_end - chunk_start

                    try:
                        print(f"[DALI] Running pipe for chunk {chunk_num}...")
                        output = self.pipe.run()
                        raw_tensor = torch.from_dlpack(output[0].as_tensor())
                        raw_tensor = raw_tensor.squeeze(0)  # [seq, H, W, C]
                        actual_frames = raw_tensor.shape[0]

                        if actual_frames > frames_needed:
                            raw_tensor = raw_tensor[:frames_needed]
                            actual_frames = frames_needed

                        total_frames_read += actual_frames
                        # Use timeout on put to avoid deadlock
                        self.load_queue.put((chunk_start, raw_tensor), timeout=60)
                        print(f"[DALI] Chunk {chunk_num}: frames {chunk_start}-{chunk_start+actual_frames-1}")
                        chunk_num += 1

                    except queue.Full:
                        print(f"[DALI] Queue full at chunk {chunk_num}, waiting...")
                        # Keep trying until stop is set
                        while not self.stop_event.is_set():
                            try:
                                self.load_queue.put((chunk_start, raw_tensor), timeout=5)
                                chunk_num += 1
                                break
                            except queue.Full:
                                continue
                    except Exception as e:
                        print(f"[DALI] Error at chunk {chunk_num}: {e}")
                        traceback.print_exc()
                        self.loader_error = f"DALI chunk {chunk_num}: {type(e).__name__}: {e}"
                        break
                # === PHASE 2: FFmpeg CUDA for last 2 chunks (GPU decode, fast seeking) ===
                if chunk_num < num_chunks:
                    print(f"[FFMPEG-CUDA] Taking over for chunks {chunk_num} to {num_chunks-1}")

                    # Get FPS for time-based seeking
                    cap = cv2.VideoCapture(self.h265_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cap.release()

                    while chunk_num < num_chunks and not self.stop_event.is_set():
                        chunk_start = chunk_num * self.chunk_size
                        chunk_end = min(chunk_start + self.chunk_size, self._length)
                        frames_needed = chunk_end - chunk_start

                        # Calculate start time in seconds
                        start_time = chunk_start / fps
                        print(f"[FFMPEG-CUDA] Chunk {chunk_num}: extracting {frames_needed} frames from {start_time:.2f}s")

                        # Use ffmpeg with CUDA for GPU-accelerated decoding
                        cmd = [
                            'ffmpeg', '-y',
                            '-hwaccel', 'cuda',           # GPU decode
                            '-ss', str(start_time),       # Seek BEFORE input (fast)
                            '-i', self.h265_path,
                            '-frames:v', str(frames_needed),
                            '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24',
                            'pipe:1'
                        ]

                        result = subprocess.run(cmd, capture_output=True)
                        if result.returncode != 0:
                            print(f"[FFMPEG-CUDA] Error: {result.stderr.decode()[:200]}")
                            break

                        # Parse raw frames
                        raw_data = result.stdout
                        expected_size = frames_needed * self.video_height * self.video_width * 3
                        if len(raw_data) < expected_size:
                            actual_frames = len(raw_data) // (self.video_height * self.video_width * 3)
                            print(f"[FFMPEG-CUDA] Got {actual_frames}/{frames_needed} frames")
                            if actual_frames == 0:
                                break
                            frames_needed = actual_frames

                        frames_np = np.frombuffer(raw_data, dtype=np.uint8)
                        frames_np = frames_np[:frames_needed * self.video_height * self.video_width * 3]
                        frames_np = frames_np.reshape(frames_needed, self.video_height, self.video_width, 3)
                        frames_tensor = torch.from_numpy(frames_np.copy()).to(f'cuda:{self.device_id}')

                        total_frames_read += frames_needed
                        self.load_queue.put((chunk_start, frames_tensor))
                        print(f"[FFMPEG-CUDA] Chunk {chunk_num}: frames {chunk_start}-{chunk_start+frames_needed-1} queued")
                        chunk_num += 1

                print(f"[LOADER] Finished: {chunk_num} chunks, {total_frames_read} frames")

            except Exception as e:
                print(f"[DALI-LOADER] Fatal error in loader thread: {e}")
                traceback.print_exc()
                self.loader_error = f"loader thread: {type(e).__name__}: {e}"

        self._loader_thread = threading.Thread(target=loader_worker, daemon=True)
        self._loader_thread.start()

    def _get_chunk(self, chunk_start):
        """Pull chunk from prefetch queue or cache"""
        if chunk_start == self.current_chunk_start:
            return

        # Check cache first (for backward pass)
        if chunk_start in self.chunk_cache:
            self.current_chunk = self.chunk_cache[chunk_start]
            self.current_chunk_start = chunk_start
            actual_end = min(chunk_start + self.chunk_size - 1, self._length - 1)
            print(f"[DALI-CACHE] Retrieved chunk {chunk_start}-{actual_end} from cache")
            return

        # Pull from prefetch queue
        try:
            while True:
                queued_start, chunk = self.load_queue.get(timeout=30)

                # Cache this chunk for potential backward pass
                self.chunk_cache[queued_start] = chunk

                # Evict old chunks if cache is too large
                if len(self.chunk_cache) > self.max_cached_chunks:
                    oldest = min(self.chunk_cache.keys())
                    if oldest != queued_start and oldest != chunk_start:
                        del self.chunk_cache[oldest]

                if queued_start == chunk_start:
                    break
                elif queued_start > chunk_start:
                    # We're behind - this shouldn't happen, but handle gracefully
                    print(f"[DALI-ASYNC] Warning: got chunk {queued_start}, wanted {chunk_start}")
                    # Check if wanted chunk is in cache now
                    if chunk_start in self.chunk_cache:
                        chunk = self.chunk_cache[chunk_start]
                        queued_start = chunk_start
                    break
                # If queued_start < chunk_start, continue pulling (skip stale)

            self.current_chunk = chunk
            self.current_chunk_start = queued_start
            actual_end = min(queued_start + self.chunk_size - 1, self._length - 1)
            print(f"[DALI-ASYNC] Activated chunk {queued_start}-{actual_end}")

        except Exception as e:
            # queue.Empty here means the loader thread is gone. Report what
            # actually killed it; the symptom alone is not diagnosable.
            if self.loader_error:
                print(f"[DALI-ASYNC] Loader died: {self.loader_error}")
                raise RuntimeError(f"video loader failed: {self.loader_error}") from e
            print(f"[DALI-ASYNC] Queue timeout for chunk {chunk_start}: {e}")
            raise

    def __getitem__(self, idx):
        """O(1) access - chunks are prefetched as raw uint8, processed per-frame"""
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Frame {idx} out of bounds (0-{self._length-1})")

        chunk_start = (idx // self.chunk_size) * self.chunk_size

        if self.current_chunk is None or chunk_start != self.current_chunk_start:
            self._get_chunk(chunk_start)

        chunk_idx = idx - self.current_chunk_start

        # Bounds check - DALI with pad_mode='none' may return shorter chunks
        if chunk_idx >= self.current_chunk.shape[0]:
            print(f"[DALI] WARNING: Frame {idx} beyond chunk bounds (chunk has {self.current_chunk.shape[0]} frames, wanted idx {chunk_idx})")
            chunk_idx = self.current_chunk.shape[0] - 1  # Clamp to last valid frame

        frame = self.current_chunk[chunk_idx]  # [H, W, C] uint8

        # Process SINGLE frame on-demand (instant on GPU - ~0.1ms)
        frame = frame.float() / 255.0
        frame = frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        frame = torch.nn.functional.interpolate(
            frame,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        frame = (frame - self.img_mean) / self.img_std
        return frame.squeeze(0)  # [C, H, W]

    def __len__(self):
        return self._length

    def cleanup(self):
        """Stop background loader and cleanup"""
        self.stop_event.set()
        if hasattr(self, '_loader_thread'):
            self._loader_thread.join(timeout=2)

        # Clear chunk cache
        if hasattr(self, 'chunk_cache'):
            self.chunk_cache.clear()
        self.current_chunk = None

        # Clean temp H.265 file
        if hasattr(self, 'h265_path') and os.path.exists(self.h265_path):
            try:
                os.unlink(self.h265_path)
                print(f"[DALI-ASYNC] Cleaned up temp file")
            except:
                pass


class RevolutionMaskCompressor:
    """
    Ultra-efficient mask compression: Delta + RLE + zlib.
    Achieves 100x compression vs raw PNGs for typical masks.
    """

    def __init__(self):
        import zlib
        self.prev_mask = None
        self.frame_count = 0
        self.compressed_masks = []

    def compress_mask(self, mask_tensor):
        """Compress mask using delta + RLE + zlib"""
        import zlib
        from itertools import groupby

        # Convert to numpy bool
        if torch.is_tensor(mask_tensor):
            mask = mask_tensor.cpu().numpy().astype(bool)
        else:
            mask = mask_tensor.astype(bool)

        if self.frame_count == 0 or self.prev_mask is None:
            # First frame: store full mask
            data = mask
            is_delta = False
        else:
            # Delta: XOR with previous (most bits will be 0)
            data = np.logical_xor(mask, self.prev_mask)
            is_delta = True

        self.prev_mask = mask.copy()
        self.frame_count += 1

        # RLE encode
        rle = self._rle_encode(data.flatten())

        # zlib compress
        compressed = zlib.compress(rle.tobytes(), level=6)

        # Store with metadata
        self.compressed_masks.append({
            'is_delta': is_delta,
            'shape': mask.shape,
            'data': compressed
        })

        return len(compressed)

    def _rle_encode(self, flat_array):
        """Run-length encoding for binary array"""
        from itertools import groupby
        rle = []
        for val, group in groupby(flat_array):
            length = sum(1 for _ in group)
            rle.extend([int(val), length])
        return np.array(rle, dtype=np.uint32)

    def save_to_file(self, output_path):
        """Save all compressed masks to a single file"""
        import pickle
        import zlib

        # Serialize and compress entire structure
        data = pickle.dumps(self.compressed_masks)
        compressed = zlib.compress(data, level=9)

        with open(output_path, 'wb') as f:
            f.write(compressed)

        size_kb = len(compressed) / 1024
        print(f"[MaskCompress] Saved {self.frame_count} masks to {size_kb:.1f}KB")
        return output_path

    @staticmethod
    def load_from_file(input_path):
        """Load and decompress masks from file"""
        import pickle
        import zlib
        from itertools import groupby

        with open(input_path, 'rb') as f:
            compressed = f.read()

        data = pickle.loads(zlib.decompress(compressed))

        masks = []
        prev_mask = None

        for entry in data:
            # Decompress RLE
            rle_data = np.frombuffer(zlib.decompress(entry['data']), dtype=np.uint32)

            # Decode RLE
            flat = []
            for i in range(0, len(rle_data), 2):
                val, length = rle_data[i], rle_data[i+1]
                flat.extend([bool(val)] * length)

            mask = np.array(flat, dtype=bool).reshape(entry['shape'])

            # Apply delta if needed
            if entry['is_delta'] and prev_mask is not None:
                mask = np.logical_xor(mask, prev_mask)

            prev_mask = mask.copy()
            masks.append((mask * 255).astype(np.uint8))

        return masks


class YUV420FrameStorage:
    """
    Load frames from disk and compress to YUV420 in SINGLE PASS.
    No double loading - compress as we load!
    """

    def __init__(self, frames_dir, image_size=1024, device="cuda"):
        import torch.nn.functional as F
        from PIL import Image
        import torchvision.transforms as transforms

        self.device = device
        self.compressed = []
        self.video_height = None
        self.video_width = None

        # SAM2's normalization constants
        img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        # Get frame paths
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        # SAM2's transform (resize to internal size)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        print(f"[YUV420] Loading & compressing {len(frame_files)} frames (single pass)...", flush=True)
        for fname in tqdm(frame_files, desc="YUV420 load+compress"):
            img = Image.open(os.path.join(frames_dir, fname)).convert("RGB")

            # Store original dimensions from first frame
            if self.video_height is None:
                self.video_width, self.video_height = img.size

            # Transform to tensor and normalize (like SAM2 does)
            tensor = transform(img).to(device)  # [3, H, W]
            tensor = (tensor - img_mean) / img_std

            # Compress immediately to YUV420
            compressed = self._compress(tensor)
            self.compressed.append(compressed)

            del tensor, img

        # Stats
        orig_size = len(self.compressed) * 3 * image_size * image_size * 4
        comp_size = sum(c['y'].numel() + c['u'].numel() + c['v'].numel() for c in self.compressed) * 2
        print(f"[YUV420] Done! {orig_size/1e9:.2f}GB -> {comp_size/1e9:.2f}GB ({orig_size/comp_size:.1f}:1)", flush=True)
        torch.cuda.empty_cache()

    def _compress(self, frame):
        """Compress [3, H, W] tensor to YUV420"""
        import torch.nn.functional as F

        frame = frame.to(self.device)
        r, g, b = frame[0], frame[1], frame[2]

        # RGB to YUV
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b

        # 4:2:0 subsampling - U/V at quarter resolution
        u_sub = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), 2).squeeze()
        v_sub = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), 2).squeeze()

        # Store as half precision
        return {
            'y': y.half(),
            'u': u_sub.half(),
            'v': v_sub.half()
        }

    def _decompress(self, compressed):
        """Decompress YUV420 back to [3, H, W] tensor"""
        import torch.nn.functional as F

        y = compressed['y'].float()
        u = F.interpolate(compressed['u'].unsqueeze(0).unsqueeze(0).float(),
                         scale_factor=2, mode='bilinear', align_corners=False).squeeze()
        v = F.interpolate(compressed['v'].unsqueeze(0).unsqueeze(0).float(),
                         scale_factor=2, mode='bilinear', align_corners=False).squeeze()

        # YUV to RGB
        r = y + 1.140 * v
        g = y - 0.395 * u - 0.581 * v
        b = y + 2.032 * u

        return torch.stack([r, g, b], dim=0)

    def __getitem__(self, index):
        return self._decompress(self.compressed[index])

    def __len__(self):
        return len(self.compressed)


class YUV420FrameStorageReordered:
    """
    Wrapper that provides frames from an existing YUV420FrameStorage in a different order.
    Used for backward tracking by reversing frame order.
    """

    def __init__(self, original_storage, frame_indices):
        """
        Args:
            original_storage: YUV420FrameStorage instance
            frame_indices: List of original frame indices in desired order
                           e.g., [100, 99, 98, ..., 0] for backward from frame 100
        """
        self.original = original_storage
        self.frame_indices = frame_indices
        self.video_height = original_storage.video_height
        self.video_width = original_storage.video_width

    def __getitem__(self, index):
        # Map new index to original frame index
        original_idx = self.frame_indices[index]
        return self.original[original_idx]

    def __len__(self):
        return len(self.frame_indices)


def init_state_reordered(predictor, original_yuv_frames, frame_indices, device="cuda"):
    """
    Initialize SAM2 inference state with reordered frames (for backward tracking).
    Reuses existing YUV420 compressed frames but accesses them in different order.
    """
    from collections import OrderedDict

    # Wrap existing frames with new order
    reordered_frames = YUV420FrameStorageReordered(original_yuv_frames, frame_indices)

    # Build inference_state
    inference_state = {}
    inference_state["images"] = reordered_frames
    inference_state["num_frames"] = len(reordered_frames)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = reordered_frames.video_height
    inference_state["video_width"] = reordered_frames.video_width
    inference_state["device"] = torch.device(device)
    inference_state["storage_device"] = torch.device(device)

    # Initialize tracking structures
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up visual backbone on frame 0 (which is original frame_idx_start)
    print("[SAM2-Reversed] Warming up visual backbone on reversed sequence...", flush=True)
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def init_state_yuv420(predictor, frames_dir, device="cuda"):
    """
    Initialize SAM2 inference state with YUV420 compressed frames.
    SKIPS SAM2's load_video_frames() entirely - no double loading!
    """
    from collections import OrderedDict

    # Load frames with YUV420 compression (single pass)
    yuv_frames = YUV420FrameStorage(frames_dir, image_size=predictor.image_size, device=device)

    # Build inference_state manually (same structure as SAM2's init_state)
    inference_state = {}
    inference_state["images"] = yuv_frames
    inference_state["num_frames"] = len(yuv_frames)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = yuv_frames.video_height
    inference_state["video_width"] = yuv_frames.video_width
    inference_state["device"] = torch.device(device)
    inference_state["storage_device"] = torch.device(device)

    # Initialize tracking structures (same as SAM2's init_state)
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up visual backbone on frame 0
    print("[SAM2-YUV420] Warming up visual backbone...", flush=True)
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def init_state_dali(predictor, video_path, device="cuda", max_height=TARGET_RESOLUTION, chunk_size=500):
    """
    Initialize SAM2 inference state with DALI chunked GPU streaming.
    ZERO CPU RAM - frames load directly from disk to GPU in chunks!

    Args:
        chunk_size: Frames per chunk (500 = ~1.3GB VRAM per chunk)
    """
    from collections import OrderedDict

    if not HAS_DALI:
        raise RuntimeError("DALI not available! pip install nvidia-dali-cuda120")

    # Create DALI CHUNKED streamer - O(1) within-chunk access, no OOM
    dali_streamer = DALIChunkedStreamer(
        video_path,
        chunk_size=chunk_size,
        device_id=0,
        image_size=predictor.image_size,
        max_height=max_height
    )

    # Build inference_state manually (same structure as SAM2's init_state)
    inference_state = {}
    inference_state["images"] = dali_streamer  # DALI replaces YUV420FrameStorage
    inference_state["num_frames"] = len(dali_streamer)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = dali_streamer.video_height
    inference_state["video_width"] = dali_streamer.video_width
    inference_state["device"] = torch.device(device)
    inference_state["storage_device"] = torch.device(device)

    # Initialize tracking structures (same as SAM2's init_state)
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Store reference for cleanup
    inference_state["_dali_streamer"] = dali_streamer

    # Warm up visual backbone on frame 0
    print("[SAM2-DALI] Warming up visual backbone (GPU-direct)...", flush=True)
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def rle_encode_mask(mask_uint8):
    """RLE compress binary mask (50:1 compression for masks)"""
    binary = (mask_uint8 > 127).astype(np.uint8).flatten()
    if len(binary) == 0:
        return [], mask_uint8.shape

    runs = []
    current_val = binary[0]
    current_len = 1

    for i in range(1, len(binary)):
        if binary[i] == current_val:
            current_len += 1
        else:
            runs.append(current_len if current_val == 1 else -current_len)
            current_val = binary[i]
            current_len = 1
    runs.append(current_len if current_val == 1 else -current_len)

    return runs, mask_uint8.shape


def rle_decode_mask(rle_runs, shape):
    """Decompress RLE back to binary mask"""
    if not rle_runs:
        return np.zeros(shape, dtype=np.uint8)

    flat = []
    for run in rle_runs:
        flat.extend([1 if run > 0 else 0] * abs(run))

    return (np.array(flat, dtype=np.uint8).reshape(shape) * 255)


# Configuration
SAM2_CHECKPOINT = SAM2_PATH / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# TensorRT engines - WSL2 paths
ENCODER_ENGINE = "/mnt/d/watermarkz/sam2_trt_inference/engines_wsl2/sam2_encoder_fp16.engine"
DECODER_ENGINE = "/mnt/d/watermarkz/sam2_trt_inference/engines_wsl2/sam2_decoder_fp16_dynamic.engine"


def enable_optimizations():
    """Enable PyTorch optimizations for WSL2"""
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_properties(0).major

        # TF32 on Ampere+
        if compute_capability >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Memory allocator
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Inductor - DISABLED for faster testing
        # torch._inductor.config.coordinate_descent_tuning = True
        # torch._inductor.config.triton.unique_kernel_names = True
        # torch._inductor.config.fx_graph_cache = True

        # SDPA
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Disable gradients
        torch.set_grad_enabled(False)

        # Async CUDA
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def extract_frames(video_path, output_dir, max_height=TARGET_RESOLUTION):
    """Extract frames from video, resized to 480p for faster loading"""
    import tempfile
    import requests

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Download video first if it's a URL (cv2 has issues with HTTPS streaming)
    local_path = video_path
    downloaded_file = None
    if video_path.startswith('http://') or video_path.startswith('https://'):
        print(f"[SAM2] Downloading video from URL...")
        downloaded_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        r = requests.get(video_path, timeout=300)
        r.raise_for_status()
        downloaded_file.write(r.content)
        downloaded_file.close()
        local_path = downloaded_file.name
        print(f"[SAM2] Downloaded {len(r.content) / 1024 / 1024:.1f}MB")

    cap = cv2.VideoCapture(local_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate resize dimensions (maintain aspect ratio, ensure even dimensions)
    if orig_height > max_height:
        scale = max_height / orig_height
        new_width = int(orig_width * scale) // 2 * 2  # Ensure even for video codecs
        new_height = max_height // 2 * 2              # Ensure even for video codecs
        print(f"[SAM2] Resizing {orig_width}x{orig_height} -> {new_width}x{new_height} (faster loading)")
    else:
        new_width, new_height = orig_width, orig_height

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if needed for faster loading
        if orig_height > max_height:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(f"{output_dir}/{frame_idx:05d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_idx += 1

    cap.release()

    # Clean up downloaded file
    if downloaded_file:
        try:
            os.remove(downloaded_file.name)
        except:
            pass

    print(f"[SAM2] Extracted {frame_idx} frames @ {fps:.1f} fps")
    return frame_idx, fps, orig_width, orig_height, new_width, new_height


def release_old_frames_complete(inference_state, keep_last=64):
    """
    Memory cleanup for SAM2 - SAFE version.

    IMPORTANT: Cannot truncate images tensor during propagation because
    propagate_in_video generator uses original frame indices.
    Only clean output_dict entries for memory savings.
    """
    try:
        # DON'T touch images tensor - generator uses original indices!
        # Only clean non_cond_frame_outputs to save some memory

        if "output_dict" in inference_state:
            output_dict = inference_state["output_dict"]

            # Only clean non-conditioning frame outputs (keep last N)
            if "non_cond_frame_outputs" in output_dict:
                keys = sorted(output_dict["non_cond_frame_outputs"].keys())
                if len(keys) > keep_last:
                    for old_idx in keys[:-keep_last]:
                        output_dict["non_cond_frame_outputs"].pop(old_idx, None)

            # DO NOT touch cond_frame_outputs - SAM2 needs them

        # Clean per-object output dictionaries
        if "output_dict_per_obj" in inference_state:
            for obj_id, obj_dict in inference_state["output_dict_per_obj"].items():
                if "non_cond_frame_outputs" in obj_dict:
                    keys = sorted(obj_dict["non_cond_frame_outputs"].keys())
                    if len(keys) > keep_last:
                        for old_idx in keys[:-keep_last]:
                            obj_dict["non_cond_frame_outputs"].pop(old_idx, None)

        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[SAM2-Memory] Cleanup warning: {e}")


def load_frames_as_tensors(frames_dir, start_idx, end_idx, img_mean, img_std, image_size):
    """
    Load frames from disk and convert to tensors matching SAM2's format.

    Returns list of tensors ready to extend inference_state["images"]
    """
    from PIL import Image
    import torchvision.transforms as transforms

    # SAM2 uses this transform internally
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    images = []
    for i in range(start_idx, end_idx):
        frame_path = os.path.join(frames_dir, f"{i:05d}.jpg")
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)

    return images


def track_video_pytorch_only(frames_dir, total_frames, output_masks_dir, points=None, labels=None, object_ids=None, bbox=None, frame_idx_start=0, video_path=None, initial_mask=None):
    """
    SAM2 video tracking using OFFICIAL memory management parameters.

    Uses start_frame_idx and max_frame_num_to_track for chunked processing
    instead of manual tensor manipulation.

    Args:
        points: List of (x, y) tuples for multiple click points
        labels: List of labels (1=foreground, 0=background) for each point
        object_ids: List of object IDs for each point (each click = separate object)
        video_path: Original video path for DALI streaming (optional)
        initial_mask: Pre-provided mask numpy array (from user eraser) - if provided, uses mask-based tracking
    """
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_optimizations()

    # Check if DALI streaming is enabled
    USE_DALI_STREAMING = os.getenv('USE_DALI_STREAMING', '0').lower() in ('1', 'true', 'yes', 'on')
    SAM2_MAX_HEIGHT = int(os.getenv('SAM2_MAX_HEIGHT', str(TARGET_RESOLUTION)))

    # Create output directory
    os.makedirs(output_masks_dir, exist_ok=True)

    CHUNK_SIZE = 128  # Process 128 frames at a time (memory optimized for long videos)

    # Build predictor
    print(f"[SAM2-Official] Building predictor with official memory management...")
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
        vos_optimized=False
    )
    torch.cuda.empty_cache()

    inference_state = None  # Initialize for finally block

    try:
        # Choose initialization method: DALI (GPU-direct) or YUV420 (CPU RAM)
        if USE_DALI_STREAMING and HAS_DALI and video_path:
            print(f"[SAM2-DALI] GPU-direct streaming enabled! ZERO CPU RAM!")
            print(f"[SAM2-DALI] Video: {video_path}, Max height: {SAM2_MAX_HEIGHT}")
            inference_state = init_state_dali(predictor, video_path, device=device, max_height=SAM2_MAX_HEIGHT)
            actual_frames = inference_state['num_frames']
            print(f"[SAM2-DALI] Ready! {actual_frames} frames streaming from GPU", flush=True)
        else:
            if USE_DALI_STREAMING and not HAS_DALI:
                print(f"[SAM2-DALI] WARNING: DALI requested but not installed! Falling back to YUV420")
            if USE_DALI_STREAMING and not video_path:
                print(f"[SAM2-DALI] WARNING: DALI requested but no video_path! Falling back to YUV420")
            # Initialize state with YUV420 compression - SKIPS SAM2's frame loading entirely!
            print(f"[SAM2-YUV420] Initializing with {total_frames} frames (single pass, no double loading)...")
            inference_state = init_state_yuv420(predictor, frames_dir, device=device)
            actual_frames = inference_state['num_frames']
            print(f"[SAM2-YUV420] Ready! {actual_frames} frames loaded", flush=True)

        # Clamp frame_idx_start to valid range (frontend may calculate wrong index if FPS differs)
        if frame_idx_start < 0 or frame_idx_start >= actual_frames:
            old_idx = frame_idx_start
            frame_idx_start = max(0, min(frame_idx_start, actual_frames - 1))
            print(f"[SAM2] WARNING: frame_idx_start {old_idx} out of range, clamped to {frame_idx_start}")

        # Add initial prompt
        if initial_mask is not None:
            # User provided a pre-erased mask - use mask-based propagation
            print(f"[SAM2-Mask] Using user-provided mask (shape: {initial_mask.shape}) at frame {frame_idx_start}")
            # Convert to binary mask (threshold at 127)
            mask_binary = (initial_mask > 127).astype(np.uint8)
            # SAM2 expects (H, W) mask with values 0 or 1
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx_start,
                obj_id=1,  # Single combined object
                mask=mask_binary
            )
            print(f"[SAM2-Mask] Added mask prompt at frame {frame_idx_start}, obj_ids: {out_obj_ids}")
        elif bbox is not None:
            box_array = np.array(bbox, dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx_start,
                obj_id=1,
                box=box_array
            )
            print(f"[SAM2-Official] Added bbox prompt at frame {frame_idx_start}")
        elif points is not None and len(points) > 0:
            # Handle multiple objects: each click can be a separate object
            if object_ids is not None and len(object_ids) == len(points):
                # Group points by object_id
                unique_obj_ids = sorted(set(object_ids))
                print(f"[SAM2-MultiObject] Found {len(unique_obj_ids)} objects: {unique_obj_ids}")

                for obj_id in unique_obj_ids:
                    # Get points for this object
                    obj_points = [points[i] for i in range(len(points)) if object_ids[i] == obj_id]
                    # points/object_ids lengths are checked above, labels was not:
                    # a burst of taps can deliver one more point than labels and
                    # labels[i] then walks off the end. Missing label means a tap,
                    # which is label 1.
                    obj_labels = [labels[i] if labels is not None and i < len(labels) else 1
                                  for i in range(len(points)) if object_ids[i] == obj_id]

                    points_array = np.array([[p[0], p[1]] for p in obj_points], dtype=np.float32)
                    labels_array = np.array(obj_labels, dtype=np.int32)

                    # SAM2 uses 1-based object IDs, so add 1 to our 0-based IDs
                    sam2_obj_id = obj_id + 1

                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx_start,
                        obj_id=sam2_obj_id,
                        points=points_array,
                        labels=labels_array
                    )
                    print(f"[SAM2-MultiObject] Added object {sam2_obj_id}: {len(obj_points)} point(s) at frame {frame_idx_start}")
            else:
                # Legacy: all points as single object (obj_id=1)
                points_array = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
                labels_array = np.array(labels, dtype=np.int32) if labels else np.ones(len(points), dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx_start,
                    obj_id=1,
                    points=points_array,
                    labels=labels_array
                )
                print(f"[SAM2-Official] Added point prompt at frame {frame_idx_start}")

        masks_saved = 0
        reid_verifier = None
        drift_count = 0
        corrections_made = 0

        # Initialize ReID verifier if available
        if HAS_REID and USE_REID_VERIFICATION:
            print(f"[ReID] Initializing identity verification...")
            reid_verifier = ReIDVerifier(object_type="general")  # Auto-detect face later

        # BIDIRECTIONAL PROPAGATION - simple internal reverse=True
        print(f"\n[SAM2-Bidirectional] Starting from frame {frame_idx_start}, propagating both directions")

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):

                # ============================================================
                # BACKWARD: frame_idx_start -> 0 (using internal reverse=True)
                # ============================================================
                backward_frames = frame_idx_start + 1  # Include click frame
                if frame_idx_start > 0:
                    print(f"\n[SAM2-Backward] Propagating frames {frame_idx_start} -> 0 ({backward_frames} frames)")

                    for frame_idx, obj_ids, mask_logits in tqdm(
                        predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=frame_idx_start,
                            max_frame_num_to_track=backward_frames,
                            reverse=True
                        ),
                        total=backward_frames,
                        desc="Backward"
                    ):
                        # Combine masks from all objects (logical OR)
                        if mask_logits.shape[0] > 1:
                            # Multiple objects: combine all masks
                            combined_mask = (mask_logits > 0.0).any(dim=0).cpu().numpy().squeeze()
                        else:
                            # Single object
                            combined_mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                        mask_uint8 = (combined_mask * 255).astype(np.uint8)

                        if np.sum(mask_uint8 > 127) < 100:
                            mask_uint8[:] = 0

                        cv2.imwrite(f"{output_masks_dir}/{frame_idx:05d}.png", mask_uint8)
                        masks_saved += 1

                        del mask_logits, combined_mask

                    print(f"[SAM2-Backward] Complete: {backward_frames} masks")
                    gc.collect()
                    torch.cuda.empty_cache()

                # ============================================================
                # FORWARD: frame_idx_start -> end
                # ============================================================
                forward_frames = actual_frames - frame_idx_start
                if forward_frames > 0:
                    if frame_idx_start > 0:
                        print(f"\n[SAM2-Forward] Propagating frames {frame_idx_start} -> {actual_frames - 1} ({forward_frames} frames)")
                    else:
                        print(f"\n[SAM2-Forward] Propagating frames 0 -> {actual_frames - 1} ({forward_frames} frames)")

                    # Open video for ReID frame extraction if needed
                    reid_cap = None
                    if reid_verifier is not None and video_path:
                        reid_cap = cv2.VideoCapture(video_path)

                    for frame_idx, obj_ids, mask_logits in tqdm(
                        predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=frame_idx_start,
                            max_frame_num_to_track=forward_frames,
                            reverse=False
                        ),
                        total=forward_frames,
                        desc="Forward"
                    ):
                        # Skip click frame if already saved by backward
                        if frame_idx_start > 0 and frame_idx == frame_idx_start:
                            continue

                        # Combine masks from all objects (logical OR)
                        if mask_logits.shape[0] > 1:
                            # Multiple objects: combine all masks
                            combined_mask = (mask_logits > 0.0).any(dim=0).cpu().numpy().squeeze()
                        else:
                            # Single object
                            combined_mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                        mask_uint8 = (combined_mask * 255).astype(np.uint8)

                        if np.sum(mask_uint8 > 127) < 100:
                            mask_uint8[:] = 0

                        # ============================================================
                        # ReID VERIFICATION - Detect and AUTO-CORRECT drift
                        # ============================================================
                        if reid_verifier is not None and reid_cap is not None:
                            # BUILD REFERENCE BANK from first 60 frames (4 viewpoints)
                            # Collect at frames relative to start: 0, 15, 30, 45
                            relative_frame = frame_idx - frame_idx_start
                            if len(reid_verifier.reference_bank) < 4 and np.sum(mask_uint8) > 1000:
                                viewpoint_frames = {0: "frontal", 15: "view_15", 30: "view_30", 45: "view_45"}
                                if relative_frame in viewpoint_frames:
                                    reid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                    ret, frame_bgr = reid_cap.read()
                                    if ret:
                                        viewpoint = viewpoint_frames[relative_frame]
                                        reid_verifier.add_reference_viewpoint(frame_bgr, mask_uint8, viewpoint)
                                        # Also set single reference on frame 0 for backward compat
                                        if relative_frame == 0:
                                            reid_verifier.set_reference(frame_bgr, mask_uint8)

                            # Verify identity every N frames (only after reference bank is built)
                            elif len(reid_verifier.reference_bank) >= 2 and frame_idx % VERIFY_EVERY_N_FRAMES == 0:
                                reid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, frame_bgr = reid_cap.read()
                                if ret:
                                    is_same, similarity = reid_verifier.verify(frame_bgr, mask_uint8)
                                    # similarity = -1.0 means uncertain (decode error), skip
                                    # Only count real verified drift (sim >= 0 and is_same=False)
                                    if not is_same and similarity >= 0:
                                        drift_count += 1
                                        tqdm.write(f"[ReID] DRIFT at frame {frame_idx} (sim={similarity:.3f})")

                                        # Search for object
                                        new_point, search_sim = reid_verifier.find_object_in_frame(
                                            frame_bgr, mask_hint=mask_uint8
                                        )

                                        # Classify tracking state
                                        state = classify_tracking_state(mask_uint8, similarity, search_sim)
                                        tqdm.write(f"[ReID] State: {state} (search_sim={search_sim:.3f})")

                                        if state == "LOST":
                                            # Object not found - ask user to click
                                            tqdm.write(f"[ReID] OBJECT LOST - requesting user input...")
                                            click_point = get_user_click_for_recovery(
                                                frame_bgr, frame_idx, mask_uint8
                                            )

                                            if click_point is not None:
                                                tqdm.write(f"[ReID] User clicked at {click_point} - re-prompting SAM2")
                                                predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=frame_idx,
                                                    obj_id=1,
                                                    points=np.array([click_point], dtype=np.float32),
                                                    labels=np.array([1], dtype=np.int32),
                                                    clear_old_points=True
                                                )
                                                corrections_made += 1
                                            else:
                                                tqdm.write(f"[ReID] User skipped frame {frame_idx}")

                                        elif state == "DRIFT" and new_point is not None:
                                            # Check if detection is close to current mask
                                            coords = np.argwhere(mask_uint8 > 127)
                                            if len(coords) > 0:
                                                mask_cy, mask_cx = coords.mean(axis=0)
                                                dist = np.sqrt((new_point[0] - mask_cx)**2 + (new_point[1] - mask_cy)**2)

                                                if dist <= 150:
                                                    # Close enough - auto-correct
                                                    tqdm.write(f"[ReID] RE-DETECTING at {new_point} ({dist:.0f}px away)")
                                                    predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=frame_idx,
                                                        obj_id=1,
                                                        points=np.array([new_point], dtype=np.float32),
                                                        labels=np.array([1], dtype=np.int32),
                                                        clear_old_points=True
                                                    )
                                                    reid_verifier.set_reference(frame_bgr, mask_uint8)
                                                    corrections_made += 1
                                                else:
                                                    # Too far - probably wrong object, skip
                                                    tqdm.write(f"[ReID] Skipping - {dist:.0f}px away (probably wrong object)")

                                        # OCCLUDED state: keep tracking, don't intervene

                        cv2.imwrite(f"{output_masks_dir}/{frame_idx:05d}.png", mask_uint8)
                        masks_saved += 1

                        del mask_logits, combined_mask

                    # Cleanup ReID video capture
                    if reid_cap is not None:
                        reid_cap.release()

                    print(f"[SAM2-Forward] Complete: {forward_frames} masks")
                    gc.collect()
                    torch.cuda.empty_cache()

        # Report ReID verification results
        if reid_verifier is not None:
            if corrections_made > 0:
                print(f"[ReID] 🔧 Made {corrections_made} auto-corrections (detected {drift_count} drift events)")
            elif drift_count > 0:
                print(f"[ReID] ⚠️  Detected {drift_count} drift events but could not auto-correct")
            else:
                print(f"[ReID] ✅ No drift detected - identity verified throughout video")

        print(f"[SAM2] Complete! Saved {masks_saved} masks to {output_masks_dir}")
        return masks_saved

    finally:
        # Cleanup DALI streamer if used (deletes temp H.265 file)
        if inference_state is not None and "_dali_streamer" in inference_state:
            inference_state["_dali_streamer"].cleanup()
        # Always cleanup predictor
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert_windows_to_wsl_path(path):
    """Convert Windows path to WSL2 path if needed"""
    if not path:
        return path
    # Handle D: or d: drive
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        return f'/mnt/{drive}' + path[2:].replace('\\', '/')
    elif '\\' in path:
        return path.replace('\\', '/')
    return path


def main():
    parser = argparse.ArgumentParser(description='SAM2 Video Tracking for WSL2')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('output_masks_dir', help='Output directory for masks')

    # Prompt options - mutually exclusive
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--point', help='Point coordinates as x,y')
    prompt_group.add_argument('--bbox', help='Bounding box as x1,y1,x2,y2')

    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index to start tracking from')
    parser.add_argument('--temp-frames-dir', default='/tmp/sam2_frames', help='Temp dir for frames')

    args = parser.parse_args()

    # Parse prompt
    point = None
    bbox = None

    if args.point:
        try:
            x, y = map(int, args.point.split(','))
            point = (x, y)
            print(f"[SAM2] Using point prompt: {point}")
        except:
            print(f"[ERROR] Invalid point format: {args.point}. Use x,y (e.g., 200,150)")
            sys.exit(1)
    elif args.bbox:
        try:
            coords = list(map(int, args.bbox.split(',')))
            if len(coords) != 4:
                raise ValueError("Need exactly 4 coordinates")
            bbox = coords  # [x1, y1, x2, y2]
            print(f"[SAM2] Using bbox prompt: {bbox}")
        except:
            print(f"[ERROR] Invalid bbox format: {args.bbox}. Use x1,y1,x2,y2 (e.g., 100,100,300,300)")
            sys.exit(1)

    # Convert Windows paths to WSL2 paths
    video_path = convert_windows_to_wsl_path(args.video_path)
    output_masks_dir = convert_windows_to_wsl_path(args.output_masks_dir)
    temp_frames_dir = convert_windows_to_wsl_path(args.temp_frames_dir)

    print(f"[SAM2] Video: {video_path}")
    print(f"[SAM2] Output: {output_masks_dir}")
    print(f"[SAM2] Frame idx: {args.frame_idx}")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Check if DALI streaming is enabled
    USE_DALI_STREAMING = os.getenv('USE_DALI_STREAMING', '0').lower() in ('1', 'true', 'yes', 'on')
    SAM2_MAX_HEIGHT = int(os.getenv('SAM2_MAX_HEIGHT', str(TARGET_RESOLUTION)))

    if USE_DALI_STREAMING and HAS_DALI:
        # DALI MODE: Skip frame extraction - stream directly from video!
        print(f"[SAM2-DALI] GPU-direct streaming mode - SKIPPING frame extraction!")

        # Get video info for coordinate scaling
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate target dimensions (same logic as DALI)
        if orig_h > SAM2_MAX_HEIGHT:
            scale_factor = SAM2_MAX_HEIGHT / orig_h
            new_w = int(orig_w * scale_factor) // 2 * 2
            new_h = SAM2_MAX_HEIGHT // 2 * 2
        else:
            new_w, new_h = orig_w, orig_h

        print(f"[SAM2-DALI] Video: {total_frames} frames @ {fps:.1f}fps, {orig_w}x{orig_h} -> {new_w}x{new_h}")
    else:
        # YUV420 MODE: Extract frames to disk
        total_frames, fps, orig_w, orig_h, new_w, new_h = extract_frames(video_path, temp_frames_dir)

    # Scale coordinates if frames were resized
    scale = new_h / orig_h if orig_h != new_h else 1.0
    if scale != 1.0:
        print(f"[SAM2] Scaling coordinates by {scale:.3f} (original {orig_w}x{orig_h} -> {new_w}x{new_h})")

        if point is not None:
            point = (int(point[0] * scale), int(point[1] * scale))
            print(f"[SAM2] Scaled point: {point}")

        if bbox is not None:
            bbox = [int(c * scale) for c in bbox]
            print(f"[SAM2] Scaled bbox: {bbox}")

    # Run tracking (pure PyTorch - simpler and still fast)
    # Convert single point to list format expected by function
    points_list = [point] if point else None
    labels_list = [1] if point else None  # 1 = foreground

    masks_saved = track_video_pytorch_only(
        temp_frames_dir,
        total_frames,
        output_masks_dir,
        points=points_list,
        labels=labels_list,
        bbox=bbox,
        frame_idx_start=args.frame_idx,
        video_path=video_path  # Pass for DALI streaming
    )

    # Output JSON for Windows to read
    result = {
        'status': 'success',
        'total_frames': total_frames,
        'masks_saved': masks_saved,
        'fps': fps,
        'masks_dir': output_masks_dir
    }
    print(f"\n[RESULT] {json.dumps(result)}")


if __name__ == '__main__':
    main()
