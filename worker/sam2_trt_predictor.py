"""
SAM2 TensorRT Python Implementation - FIXED
Properly handles all encoder/decoder tensor bindings
Fast, stable, Windows-compatible predictor for RTX 4090
"""
import os
import sys
import time

# Add TensorRT DLLs to PATH (Windows-specific fix)
tensorrt_bin = r"D:\watermarkz\TensorRT-10.13.3.9\bin"
if tensorrt_bin not in os.environ["PATH"]:
    os.environ["PATH"] = tensorrt_bin + os.pathsep + os.environ["PATH"]

import numpy as np
# Try tensorrt_bindings first (WSL2), fallback to tensorrt (Windows)
try:
    import tensorrt_bindings as trt
except ImportError:
    import tensorrt as trt
import cv2
from pathlib import Path
import logging
import torch
import pycuda.driver as cuda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAM2TensorRTPredictor:
    """Pure Python TensorRT SAM2 predictor with proper tensor binding"""

    def __init__(self, encoder_engine_path, decoder_engine_path):
        # Initialize CUDA context on THIS thread (critical for multi-threading)
        cuda.init()
        self.cuda_device = cuda.Device(0)
        self.cuda_context = self.cuda_device.make_context()
        logger.info(f"[CUDA] Initialized device: {self.cuda_device.name()}, compute: {self.cuda_device.compute_capability()}")

        self.encoder_engine_path = Path(encoder_engine_path)
        self.decoder_engine_path = Path(decoder_engine_path)

        # Use dynamic decoder engine
        if "dynamic" not in str(decoder_engine_path):
            logger.warning("[WARN] Using non-dynamic decoder engine - may have binding issues")
            logger.warning("[WARN] Recommended: use sam2_decoder_fp16_dynamic.engine")

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load engines
        self.encoder_engine = self._load_engine(self.encoder_engine_path)
        self.decoder_engine = self._load_engine(self.decoder_engine_path)

        # Create execution contexts
        self.encoder_context = self.encoder_engine.create_execution_context()
        self.decoder_context = self.decoder_engine.create_execution_context()

        # Enumerate encoder tensors
        self.encoder_tensors = self._enumerate_tensors(self.encoder_engine)
        self.encoder_inputs = [t for t in self.encoder_tensors if t['mode'] == 'INPUT']
        self.encoder_outputs = [t for t in self.encoder_tensors if t['mode'] == 'OUTPUT']

        logger.info(f"[ENCODER] Inputs: {[t['name'] for t in self.encoder_inputs]}")
        logger.info(f"[ENCODER] Outputs: {[t['name'] for t in self.encoder_outputs]}")

        # Enumerate decoder tensors
        self.decoder_tensors = self._enumerate_tensors(self.decoder_engine)
        self.decoder_inputs = [t for t in self.decoder_tensors if t['mode'] == 'INPUT']
        self.decoder_outputs = [t for t in self.decoder_tensors if t['mode'] == 'OUTPUT']

        logger.info(f"[DECODER] Inputs: {[t['name'] for t in self.decoder_inputs]}")
        logger.info(f"[DECODER] Outputs: {[t['name'] for t in self.decoder_outputs]}")

        # Image embeddings cache (all 3 encoder outputs)
        self.image_embed = None
        self.high_res_feats_0 = None
        self.high_res_feats_1 = None
        self.current_image_hash = None

        # Original image dimensions for coordinate normalization
        self.orig_w = None
        self.orig_h = None

        # The decoder is built with one optimization profile and will refuse any
        # batch above its max - set_input_shape returns False rather than raising,
        # so the context quietly keeps the previous request's size.
        self.max_prompts = 16
        try:
            prof = self.decoder_engine.get_tensor_profile_shape('point_coords', 0)
            self.max_prompts = int(prof[2][0])
        except Exception as e:
            logger.warning(f"[WARN] Could not read decoder profile, assuming {self.max_prompts}: {e}")
        logger.info(f"[DECODER] Max points per request: {self.max_prompts}")

        logger.info(f"[OK] Loaded encoder: {self.encoder_engine_path.name}")
        logger.info(f"[OK] Loaded decoder: {self.decoder_engine_path.name}")

        # Create separate CUDA streams for encoder and decoder to avoid race conditions
        # Using the same stream for both can cause interference between operations
        self.encoder_stream = cuda.Stream()
        self.decoder_stream = cuda.Stream()

    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _enumerate_tensors(self, engine):
        """Enumerate all input/output tensors in engine"""
        tensors = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)

            tensors.append({
                'index': i,
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'mode': 'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT',
                'size': int(np.prod(shape)) * trt.volume(trt.Dims([4]))  # FP32 = 4 bytes
            })

        return tensors

    def set_image(self, image):
        """
        Encode image and cache ALL embeddings (3 outputs)

        Args:
            image: numpy array (H, W, 3) RGB uint8
        """
        # Check if we already encoded this image
        image_hash = hash(image.tobytes())
        if image_hash == self.current_image_hash:
            return  # Already encoded

        # Store original image size for coordinate normalization
        self.orig_h, self.orig_w = image.shape[:2]

        # Preprocess image to 1024x1024
        img_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        # Normalize to [-1, 1] and transpose to CHW
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_chw, axis=0)  # Add batch dim -> (1, 3, 1024, 1024)
        img_batch = np.ascontiguousarray(img_batch)  # Ensure contiguous array for CUDA

        # Allocate GPU memory for input
        d_input = cuda.mem_alloc(img_batch.nbytes)
        cuda.memcpy_htod(d_input, img_batch)

        # Allocate GPU memory for ALL 3 encoder outputs
        encoder_outputs_gpu = {}
        encoder_outputs_host = {}

        for output_tensor in self.encoder_outputs:
            name = output_tensor['name']
            shape = output_tensor['shape']
            size = int(np.prod(shape)) * np.dtype(np.float32).itemsize

            # Allocate GPU buffer
            encoder_outputs_gpu[name] = cuda.mem_alloc(size)

            # Allocate host buffer
            encoder_outputs_host[name] = np.empty(shape, dtype=np.float32)

        # Bind input tensor
        input_name = self.encoder_inputs[0]['name']
        self.encoder_context.set_tensor_address(input_name, int(d_input))

        # Bind ALL output tensors
        for name, d_output in encoder_outputs_gpu.items():
            self.encoder_context.set_tensor_address(name, int(d_output))

        # Run inference with encoder stream
        self.encoder_context.execute_async_v3(self.encoder_stream.handle)
        self.encoder_stream.synchronize()  # Wait for execution to complete

        # Copy ALL outputs back to host
        for name, d_output in encoder_outputs_gpu.items():
            cuda.memcpy_dtoh(encoder_outputs_host[name], d_output)

        # Cache all embeddings (handle different possible names)
        self.image_embed = encoder_outputs_host.get('image_embed')
        if self.image_embed is None:
            self.image_embed = encoder_outputs_host.get('image_embeddings')

        self.high_res_feats_0 = encoder_outputs_host.get('high_res_feats_0')
        if self.high_res_feats_0 is None:
            self.high_res_feats_0 = encoder_outputs_host.get('high_res_feat_0')

        self.high_res_feats_1 = encoder_outputs_host.get('high_res_feats_1')
        if self.high_res_feats_1 is None:
            self.high_res_feats_1 = encoder_outputs_host.get('high_res_feat_1')

        self.current_image_hash = image_hash

        logger.info(f"[OK] Encoded image - image_embed: {self.image_embed.shape if self.image_embed is not None else 'None'}")
        logger.info(f"[OK] high_res_feats_0: {self.high_res_feats_0.shape if self.high_res_feats_0 is not None else 'None'}")
        logger.info(f"[OK] high_res_feats_1: {self.high_res_feats_1.shape if self.high_res_feats_1 is not None else 'None'}")

        # Synchronize both PyCUDA and PyTorch CUDA before freeing memory
        # This ensures all async operations are complete before cleanup
        self.encoder_stream.synchronize()  # PyCUDA encoder stream
        torch.cuda.synchronize()   # PyTorch CUDA context

        # Free GPU memory to prevent memory leak
        d_input.free()
        for d_output in encoder_outputs_gpu.values():
            d_output.free()

    def get_image_embeddings(self):
        """
        Get cached TensorRT encoder outputs for hybrid PyTorch/TensorRT pipeline

        Returns:
            tuple: (image_embed, high_res_feats_0, high_res_feats_1) as numpy arrays
        """
        if self.image_embed is None:
            raise ValueError("Call set_image() first to encode an image")

        return self.image_embed, self.high_res_feats_0, self.high_res_feats_1

    def predict(self, point_coords, point_labels, mask_input=None):
        """
        Predict mask from point prompts

        Args:
            point_coords: numpy array of shape (N, 2) with (x, y) coordinates
            point_labels: numpy array of shape (N,) with 1 for positive, 0 for negative
            mask_input: optional numpy array [1, 1, 256, 256] - previous mask for tracking

        Returns:
            masks: numpy array (H, W) with predicted mask
            scores: confidence scores
        """
        if self.image_embed is None:
            raise ValueError("Call set_image() first to encode an image")

        # Normalize coordinates to 1024x1024 space (SAM2 expects this)
        points = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.float32)

        # This preview decoder tops out at max_prompts. The render path uses
        # PyTorch SAM2 with every point, so trimming here only affects the
        # on-screen mask - and the newest clicks are the ones refining it.
        cap = getattr(self, 'max_prompts', 16)
        if len(points) > cap:
            logger.warning(f"[TRT] {len(points)} points over engine max {cap}, previewing the last {cap}")
            points = points[-cap:]
            labels = labels[-cap:]

        # Scale coordinates from original image size to 1024x1024
        scale_x = 1024.0 / self.orig_w
        scale_y = 1024.0 / self.orig_h
        points[:, 0] *= scale_x  # Scale x coordinates
        points[:, 1] *= scale_y  # Scale y coordinates

        logger.info(f"[TRT] Normalized {len(points)} points to 1024x1024 space (orig: {self.orig_w}x{self.orig_h})")

        # Reshape to (N, 2, 2) format - each prompt has 2 point slots
        num_prompts = len(points)
        points_reshaped = np.zeros((num_prompts, 2, 2), dtype=np.float32)
        points_reshaped[:, 0, :] = points  # First point (actual point)
        points_reshaped[:, 1, :] = points  # Second point (padding/duplicate)

        labels_reshaped = np.zeros((num_prompts, 2), dtype=np.float32)
        labels_reshaped[:, 0] = labels  # First label (actual label)
        labels_reshaped[:, 1] = 0  # Second label (padding)

        # Create mask_input and has_mask_input flag
        if mask_input is None:
            # No previous mask - use zeros
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0.0], dtype=np.float32)  # 0 = no previous mask
        else:
            # Use provided mask for tracking
            # Ensure correct shape [1, 1, 256, 256]
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError(f"mask_input must be shape (1, 1, 256, 256), got {mask_input.shape}")
            mask_input = np.ascontiguousarray(mask_input, dtype=np.float32)
            has_mask_input = np.array([1.0], dtype=np.float32)  # 1 = using previous mask

        # Allocate GPU memory for ALL decoder inputs
        decoder_inputs_gpu = {}

        # Input 1: image_embed (from encoder)
        if self.image_embed is not None:
            d_image_embed = cuda.mem_alloc(self.image_embed.nbytes)
            cuda.memcpy_htod(d_image_embed, np.ascontiguousarray(self.image_embed))
            decoder_inputs_gpu['image_embed'] = d_image_embed
            decoder_inputs_gpu['image_embeddings'] = d_image_embed  # Try both names

        # Input 2: high_res_feats_0 (from encoder)
        if self.high_res_feats_0 is not None:
            d_high_res_0 = cuda.mem_alloc(self.high_res_feats_0.nbytes)
            cuda.memcpy_htod(d_high_res_0, np.ascontiguousarray(self.high_res_feats_0))
            decoder_inputs_gpu['high_res_feats_0'] = d_high_res_0
            decoder_inputs_gpu['high_res_feat_0'] = d_high_res_0  # Try both names

        # Input 3: high_res_feats_1 (from encoder)
        if self.high_res_feats_1 is not None:
            d_high_res_1 = cuda.mem_alloc(self.high_res_feats_1.nbytes)
            cuda.memcpy_htod(d_high_res_1, np.ascontiguousarray(self.high_res_feats_1))
            decoder_inputs_gpu['high_res_feats_1'] = d_high_res_1
            decoder_inputs_gpu['high_res_feat_1'] = d_high_res_1  # Try both names

        # Input 4: point_coords (reshaped to TIER IV format)
        points_contig = np.ascontiguousarray(points_reshaped)
        d_points = cuda.mem_alloc(points_contig.nbytes)
        cuda.memcpy_htod(d_points, points_contig)
        decoder_inputs_gpu['point_coords'] = d_points

        # Input 5: point_labels (reshaped to TIER IV format)
        labels_contig = np.ascontiguousarray(labels_reshaped)
        d_labels = cuda.mem_alloc(labels_contig.nbytes)
        cuda.memcpy_htod(d_labels, labels_contig)
        decoder_inputs_gpu['point_labels'] = d_labels

        # Input 6: mask_input (expand batch dimension to match num_prompts)
        mask_input_batch = np.tile(mask_input, (num_prompts, 1, 1, 1))
        d_mask_input = cuda.mem_alloc(mask_input_batch.nbytes)
        cuda.memcpy_htod(d_mask_input, mask_input_batch)
        decoder_inputs_gpu['mask_input'] = d_mask_input

        # Input 7: has_mask_input (scalar, always shape (1,))
        d_has_mask = cuda.mem_alloc(has_mask_input.nbytes)
        cuda.memcpy_htod(d_has_mask, has_mask_input)
        decoder_inputs_gpu['has_mask_input'] = d_has_mask

        # IMPORTANT: Set dynamic input shapes before inference
        logger.info(f"[TRT] Setting dynamic shapes for batch size: {num_prompts}")
        def _set_shapes():
            return (
                self.decoder_context.set_input_shape('point_coords', (num_prompts, 2, 2))
                and self.decoder_context.set_input_shape('point_labels', (num_prompts, 2))
                and self.decoder_context.set_input_shape('mask_input', (num_prompts, 1, 256, 256))
            )

        if _set_shapes() is False:
            logger.warning(f"[TRT] set_input_shape({num_prompts}) refused, retrying")
            time.sleep(0.25)
            if _set_shapes() is False:
                raise RuntimeError(
                    f"TensorRT would not accept batch size {num_prompts} "
                    f"(engine max {getattr(self, 'max_prompts', 'unknown')})")

        # Allocate GPU memory for outputs - MUST query shapes AFTER setting input shapes
        # because outputs have dynamic dimensions tied to the batch size
        decoder_outputs_gpu = {}
        decoder_outputs_host = {}

        for output_tensor in self.decoder_outputs:
            name = output_tensor['name']
            # Query actual shape after input shapes are set (critical for dynamic outputs!)
            actual_shape = self.decoder_context.get_tensor_shape(name)
            size = int(np.prod(actual_shape)) * np.dtype(np.float32).itemsize

            decoder_outputs_gpu[name] = cuda.mem_alloc(size)
            decoder_outputs_host[name] = np.empty(actual_shape, dtype=np.float32)

        # Bind ALL decoder inputs
        for input_tensor in self.decoder_inputs:
            name = input_tensor['name']
            if name in decoder_inputs_gpu:
                self.decoder_context.set_tensor_address(name, int(decoder_inputs_gpu[name]))
            else:
                logger.warning(f"[WARN] Decoder input '{name}' not found in prepared inputs")

        # Bind ALL decoder outputs
        for name, d_output in decoder_outputs_gpu.items():
            self.decoder_context.set_tensor_address(name, int(d_output))

        # Run decoder with decoder stream
        self.decoder_context.execute_async_v3(self.decoder_stream.handle)
        self.decoder_stream.synchronize()  # Wait for execution to complete

        # Copy outputs back
        for name, d_output in decoder_outputs_gpu.items():
            cuda.memcpy_dtoh(decoder_outputs_host[name], d_output)

        # Extract mask and IoU from outputs (try different possible names)
        mask_logits = decoder_outputs_host.get('masks')
        if mask_logits is None:
            mask_logits = decoder_outputs_host.get('output_mask')
        if mask_logits is None:
            mask_logits = decoder_outputs_host.get('low_res_masks')

        iou_predictions = decoder_outputs_host.get('iou_predictions')
        if iou_predictions is None:
            iou_predictions = decoder_outputs_host.get('output_confidence')
        if iou_predictions is None:
            iou_predictions = decoder_outputs_host.get('iou_scores')

        if mask_logits is None:
            raise RuntimeError(f"Could not find mask output. Available: {list(decoder_outputs_host.keys())}")

        # Post-process mask (sigmoid + threshold)
        # MULTI-CLICK FIX: Combine masks respecting positive/negative labels
        # Positive points (label=1) add to mask, negative points (label=0) subtract
        num_prompts = mask_logits.shape[0]
        if num_prompts > 1:
            # Multiple clicks - combine with label awareness
            all_masks_probs = 1 / (1 + np.exp(-mask_logits[:, 0]))  # Sigmoid on all batch items

            # Separate positive and negative masks based on labels
            positive_mask = np.zeros(all_masks_probs.shape[1:], dtype=bool)
            negative_mask = np.zeros(all_masks_probs.shape[1:], dtype=bool)

            if len(labels) != num_prompts:
                logger.warning(
                    f"[TRT] decoder returned {num_prompts} masks for {len(labels)} points")
            # Walk the masks the decoder actually produced, never the point
            # count. A missing label means a plain tap, which is label 1.
            for i in range(num_prompts):
                label = labels[i] if i < len(labels) else 1
                mask_binary_i = all_masks_probs[i] > 0.5
                if label == 1:  # Positive - include this region
                    positive_mask = positive_mask | mask_binary_i
                else:  # Negative (label=0) - exclude this region
                    negative_mask = negative_mask | mask_binary_i

            # Final mask = positive regions MINUS negative regions
            mask_combined = positive_mask & ~negative_mask
            mask_binary = mask_combined.astype(np.uint8)
            pos_count = int(np.sum(labels == 1))
            neg_count = int(np.sum(labels == 0))
            logger.info(f"[TRT] Combined {num_prompts} masks: {pos_count} positive, {neg_count} negative")
        else:
            # Single click - use original logic
            mask_logits_2d = mask_logits[0, 0]
            mask_probs = 1 / (1 + np.exp(-mask_logits_2d))  # Sigmoid
            mask_binary = (mask_probs > 0.5).astype(np.uint8)

        # Extract IoU score (ensure it's a scalar)
        if iou_predictions is not None:
            iou_score = float(iou_predictions.flatten()[0])
            # Validate IoU range - negative values indicate memory corruption
            if iou_score < 0.0 or iou_score > 1.0:
                logger.warning(f"[WARN] Invalid IoU: {iou_score:.3f} - likely GPU memory corruption, resetting to 0.0")
                iou_score = 0.0
        else:
            iou_score = 0.0

        logger.info(f"[OK] Predicted mask: {mask_binary.shape}, IoU: {iou_score:.3f}")

        # Synchronize both PyCUDA and PyTorch CUDA before freeing memory
        self.decoder_stream.synchronize()  # PyCUDA decoder stream
        torch.cuda.synchronize()   # PyTorch CUDA context

        # Free GPU memory to prevent memory leak
        for d_buf in set(decoder_inputs_gpu.values()):
            d_buf.free()
        for d_buf in decoder_outputs_gpu.values():
            d_buf.free()

        return mask_binary, iou_score

    def predict_with_embeddings(self, custom_image_embed, point_coords, point_labels):
        """
        Predict mask using CUSTOM image embeddings (for memory-fused features)

        This method allows using memory-fused embeddings from PyTorch SAM2 memory attention
        instead of the original cached embeddings from set_image().

        Args:
            custom_image_embed: numpy array [1, 256, 64, 64] - memory-fused features
            point_coords: numpy array of shape (N, 2) with (x, y) coordinates
            point_labels: numpy array of shape (N,) with 1 for positive, 0 for negative

        Returns:
            masks: numpy array (H, W) with predicted mask
            scores: confidence scores
        """
        if self.high_res_feats_0 is None or self.high_res_feats_1 is None:
            raise ValueError("Call set_image() first to encode high-res features")

        # Temporarily replace cached embeddings with memory-fused version
        original_embed = self.image_embed
        self.image_embed = custom_image_embed

        # Run prediction using the standard predict() method
        mask, score = self.predict(point_coords, point_labels)

        # Restore original embeddings
        self.image_embed = original_embed

        return mask, score

    def __del__(self):
        """Cleanup CUDA context when predictor is destroyed"""
        try:
            # Don't pop context - causes issues with PyTorch's CUDA context
            # Context will be cleaned up automatically when object is destroyed
            if hasattr(self, 'cuda_context'):
                logger.info("[CUDA] Context will be auto-cleaned")
        except Exception as e:
            logger.warning(f"[CUDA] Cleanup failed: {e}")


def test_sam2_trt():
    """Test the TensorRT SAM2 predictor"""
    import time

    # Paths to TensorRT engines (use dynamic decoder)
    encoder_engine = r"D:\watermarkz\sam2_trt_inference\engines\sam2_encoder_fp16.engine"
    decoder_engine = r"D:\watermarkz\sam2_trt_inference\engines\sam2_decoder_fp16_dynamic.engine"

    # Initialize predictor
    print("[*] Initializing SAM2 TensorRT predictor...")
    predictor = SAM2TensorRTPredictor(encoder_engine, decoder_engine)

    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Benchmark encoding
    print("\n[*] Benchmarking encoder...")
    start = time.perf_counter()
    predictor.set_image(test_image)
    encode_time = (time.perf_counter() - start) * 1000
    print(f"   Encoding time: {encode_time:.2f}ms")

    # Benchmark decoding
    print("\n[*] Benchmarking decoder...")
    point_coords = np.array([[256, 256]])  # Center point
    point_labels = np.array([1])  # Positive point

    times = []
    for i in range(10):
        start = time.perf_counter()
        mask, score = predictor.predict(point_coords, point_labels)
        decode_time = (time.perf_counter() - start) * 1000
        times.append(decode_time)
        if i == 0:
            print(f"   First run: {decode_time:.2f}ms")

    avg_time = np.mean(times[1:])  # Skip first run (warmup)
    print(f"   Average decode time: {avg_time:.2f}ms")
    print(f"   Total pipeline: {encode_time + avg_time:.2f}ms")

    print(f"\n[OK] SAM2 TensorRT working! Mask shape: {mask.shape}")
    print(f"[*] Target: <20ms total latency on RTX 4090")


if __name__ == "__main__":
    test_sam2_trt()
