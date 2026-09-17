#!/bin/bash
# =============================================================================
# MarkRemoverAI - unified entrypoint
# =============================================================================
# Merges docker-entrypoint-sam2workers.sh (SAM2, from sam2workers-v11) and
# docker-entrypoint.sh (ProPainter TensorRT, from v11).
#
# Boot sequence:
#   1. preflight                 nvidia-smi, REDIS_URL, B2 creds
#   2. detect compute capability and pick an arch-keyed engine cache
#   3. build every TRT engine that is missing, for THIS GPU
#   4. re-gate FP8 / TRT feature flags on what actually built
#   5. launch the selected processes as tmux windows
#   6. block on tmux wait-for and restart only what dies
#
# ROLE = all | clicker | sam2 | propainter | api
# =============================================================================

set -e

echo "============================================================"
echo "MarkRemoverAI - Unified Worker"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Preflight
# -----------------------------------------------------------------------------
if ! nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found or GPU not accessible!"
    echo "        Make sure you are running with --gpus all"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

# REDIS_URL is not optional: gpu_coordinator.py raises RuntimeError at import
# without it, and it is also the Celery broker for all four queues.
if [ -z "$REDIS_URL" ]; then
    echo "[ERROR] REDIS_URL environment variable not set!"
    echo "        Example: -e REDIS_URL='redis://:password@host:port/0'"
    exit 1
fi

if [ -z "$B2_KEY_ID" ] || [ -z "$B2_APP_KEY" ]; then
    echo "[WARN] B2_KEY_ID or B2_APP_KEY not set - mask uploads will fail!"
fi

# -----------------------------------------------------------------------------
# 2. Detect compute capability, pick an arch-keyed engine cache
# -----------------------------------------------------------------------------
# THE core fix for "runs on any GPU". A TensorRT engine is tied to the compute
# capability it was built on. Every engine baked into the source images was
# sm_86 or sm_89; on a 5090 (sm_120) they simply fail to deserialize.
# Keying the cache directory by capability means a wrong-arch engine can never
# be picked up, and mounting /engines makes the build a one-time cost.
# Explicit override wins over ALL probing. The group is provisioned 5090-only,
# so setting SAM2_ENGINE_ARCH=120 as a group env var makes the arch key immune
# to whatever driver the next random Salad host ships. Some old-driver consumer
# boxes cannot report Blackwell's compute_cap at all; every probe (nvidia-smi
# OR torch) then guesses wrong, mis-keys the cache to e.g. sm86, never finds the
# baked sm120 engines, and burns ~13 min rebuilding into a dir that never
# matches next boot. Pinning removes the guess entirely.
if [ -n "$SAM2_ENGINE_ARCH" ]; then
    CC=$(echo "$SAM2_ENGINE_ARCH" | tr -d '. ')
    CC_MAJOR="${CC%?}"
    CC_RAW="${CC_MAJOR}.${CC#$CC_MAJOR}"
    echo "[GPU] Arch PINNED via SAM2_ENGINE_ARCH -> sm_${CC} (skipping all probing)"
else
    CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')
    if [ -z "$CC_RAW" ]; then
        # Older nvidia-smi builds have no `compute_cap` query field, so the line
        # above comes back empty on some otherwise-fine 5090 nodes. Do NOT fall
        # through to torch here: torch.cuda.get_device_capability() can report the
        # arch torch was COMPILED for, not the card present, mis-keying to sm86.
        # Map from the device NAME instead - it names the real silicon, and the
        # name query is far more universally supported than compute_cap.
        NAME_FOR_ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
        echo "[WARN] nvidia-smi compute_cap empty; mapping arch from name: '$NAME_FOR_ARCH'"
        case "$NAME_FOR_ARCH" in
            *5090*|*5080*|*5070*|*"PRO 6000"*|*B200*|*B100*) CC_RAW="12.0" ;;
            *4090*|*4080*|*4070*|*Ada*|*L40*|*L4*)           CC_RAW="8.9"  ;;
            *3090*|*3080*|*3070*|*A5000*|*A6000*|*A40*|*A10*) CC_RAW="8.6"  ;;
            *A100*|*A30*)                                     CC_RAW="8.0"  ;;
            *)  # This group is provisioned 5090-only, so an unrecognised name is
                # far likelier a naming quirk on a 5090 than real older silicon.
                CC_RAW="12.0"
                echo "[WARN] unrecognised GPU '$NAME_FOR_ARCH' - defaulting sm_120 (group is 5090-only)" ;;
        esac
    fi
    CC=$(echo "$CC_RAW" | tr -d '.')
    CC_MAJOR=$(echo "$CC_RAW" | cut -d. -f1)
fi

# The cache key carries BOTH the compute capability and the TensorRT version.
# A serialized engine is tied to both; keying on sm120 alone would mean a future
# base-image bump (TRT 10.9 -> 10.10) silently loads engines that cannot
# deserialize - the same wrong-artifact bug this image exists to prevent.
CACHE_KEY="sm${CC}-trt${TENSORRT_VERSION:-unknown}"
ENGINE_DIR="${SAM2_ENGINE_DIR:-/engines}/${CACHE_KEY}"
mkdir -p "$ENGINE_DIR"

# Engines built during THIS boot, so only genuinely new ones get uploaded.
BUILT=()

echo "[GPU] Detected : $GPU_NAME"
echo "[GPU] VRAM     : ${GPU_MEMORY}MB"
echo "[GPU] Compute  : ${CC_RAW}  (sm_${CC})"
echo "[GPU] Engines  : $ENGINE_DIR"
echo ""

# -----------------------------------------------------------------------------
# 3a. Pull cached engines from B2
# -----------------------------------------------------------------------------
# SaladCloud has no persistent volumes - storage is wiped on every stop or node
# reallocation. Without this, all four trtexec builds re-run every time (~25 min
# of paid GPU) before the worker is useful. Anything pulled here makes the
# matching build block below a no-op, because each is guarded on file existence.
# Fast path: if the engines are already present (baked into the image for this
# arch), skip the B2 pull entirely - nothing to download, boot straight through.
BAKED_OK=0
if [ "${ENGINE_CACHE_REFRESH:-0}" != "1" ] \
   && [ -f "$ENGINE_DIR/sam2_encoder_fp16.engine" ] \
   && [ -f "$ENGINE_DIR/sam2_decoder_fp16_dynamic.engine" ] \
   && [ -f "$ENGINE_DIR/rfcnet_dcnv4_fp16.engine" ] \
   && [ -f "$ENGINE_DIR/neuflow_things_fp16.engine" ]; then
    BAKED_OK=1
    echo "[BAKED] all 4 engines already present in $ENGINE_DIR - no build, no B2 pull"
fi

if [ "$BAKED_OK" = "1" ]; then
    :  # engines baked in, skip cache logic entirely
elif [ "${ENGINE_CACHE_B2:-1}" = "1" ] && [ "${ENGINE_CACHE_REFRESH:-0}" != "1" ]; then
    echo "[CACHE] checking b2://${B2_BUCKET:-watermarkz}/${ENGINE_CACHE_PREFIX:-engines}/${CACHE_KEY}/"
    python3 /app/engine_cache.py pull "$CACHE_KEY" "$ENGINE_DIR" || true
elif [ "${ENGINE_CACHE_REFRESH:-0}" = "1" ]; then
    echo "[CACHE] ENGINE_CACHE_REFRESH=1 - skipping pull, forcing a rebuild"
    rm -f "$ENGINE_DIR"/*.engine
else
    echo "[CACHE] ENGINE_CACHE_B2=0 - cache disabled, building locally"
fi
echo ""

# -----------------------------------------------------------------------------
# 3b. Build whatever is still missing
# -----------------------------------------------------------------------------
# ---- SAM2 (clicker) ----------------------------------------------------------
SAM2_ONNX_DIR="/app/sam2_trt_inference/sam2_pytorch2onnx/output"
ENCODER_ONNX="$SAM2_ONNX_DIR/sam2.1_hiera_tiny_encoder.onnx"
DECODER_ONNX="$SAM2_ONNX_DIR/sam2.1_hiera_tiny_decoder.onnx"
ENCODER_ENGINE="$ENGINE_DIR/sam2_encoder_fp16.engine"
DECODER_ENGINE="$ENGINE_DIR/sam2_decoder_fp16_dynamic.engine"

if [ ! -f "$ENCODER_ENGINE" ]; then
    if [ -f "$ENCODER_ONNX" ]; then
        echo "[BUILD] SAM2 encoder engine missing for sm_${CC}, building (~5 min)..."
        trtexec \
            --onnx="$ENCODER_ONNX" \
            --saveEngine="$ENCODER_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --builderOptimizationLevel=5
        echo "[OK] SAM2 encoder engine built: $ENCODER_ENGINE"
        BUILT+=("sam2_encoder_fp16.engine")
    else
        echo "[FATAL] SAM2 encoder ONNX missing: $ENCODER_ONNX"
        echo "        The image is built wrong - this is exactly the bug that"
        echo "        made sam2workers-v11 unable to build its own engines."
        exit 1
    fi
else
    echo "[OK] SAM2 encoder engine exists: $ENCODER_ENGINE"
fi

if [ ! -f "$DECODER_ENGINE" ]; then
    if [ -f "$DECODER_ONNX" ]; then
        echo "[BUILD] SAM2 decoder engine missing for sm_${CC}, building (~5 min)..."
        # Dynamic shape profile lifted verbatim from the v11 entrypoint:
        # 1 -> 4 -> 16 click points.
        trtexec \
            --onnx="$DECODER_ONNX" \
            --saveEngine="$DECODER_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --builderOptimizationLevel=5 \
            --minShapes=point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1 \
            --optShapes=point_coords:4x2x2,point_labels:4x2,mask_input:4x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1 \
            --maxShapes=point_coords:16x2x2,point_labels:16x2,mask_input:16x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1
        echo "[OK] SAM2 decoder engine built: $DECODER_ENGINE"
        BUILT+=("sam2_decoder_fp16_dynamic.engine")
    else
        echo "[FATAL] SAM2 decoder ONNX missing: $DECODER_ONNX"
        exit 1
    fi
else
    echo "[OK] SAM2 decoder engine exists: $DECODER_ENGINE"
fi

export SAM2_ENCODER_ENGINE="$ENCODER_ENGINE"
export SAM2_DECODER_ENGINE="$DECODER_ENGINE"

# ---- RFCNet DCNv4 (ProPainter) ----------------------------------------------
# The app reads a hardcoded path, so the real engine lives in the arch-keyed
# cache and the expected location is a symlink into it.
RFCNET_ONNX="/app/engines/rfcnet/rfcnet_dcnv4.onnx"
RFCNET_ENGINE="$ENGINE_DIR/rfcnet_dcnv4_fp16.engine"
RFCNET_LINK="/app/engines/rfcnet/rfcnet_dcnv4_fp16.engine"
DCNV4_PLUGIN="${DCNV4_PLUGIN_PATH:-/app/libdcnv4_plugin.so}"
RFCNET_TRT_OK=0

# trtexec renamed --plugins to --staticPlugins across the TRT 10 line. v11 built
# against 10.7 and used --plugins; this image is 10.9. Detect rather than assume,
# so the same script keeps working if the base image moves again.
TRTEXEC_HELP="$(trtexec --help 2>&1 || true)"
if echo "$TRTEXEC_HELP" | grep -q -- "--staticPlugins"; then
    PLUGIN_FLAG="--staticPlugins"
elif echo "$TRTEXEC_HELP" | grep -q -- "--plugins"; then
    PLUGIN_FLAG="--plugins"
else
    PLUGIN_FLAG="--staticPlugins"
fi

if [ -f "$RFCNET_ENGINE" ]; then
    echo "[OK] RFCNet DCNv4 engine exists: $RFCNET_ENGINE"
    RFCNET_TRT_OK=1
elif [ ! -f "$RFCNET_ONNX" ]; then
    echo "[WARN] RFCNet ONNX missing: $RFCNET_ONNX - PyTorch fallback"
elif [ ! -f "$DCNV4_PLUGIN" ]; then
    echo "[WARN] DCNv4 plugin missing: $DCNV4_PLUGIN - PyTorch fallback"
else
    echo "[BUILD] RFCNet DCNv4 engine missing for sm_${CC}, building (~5 min)..."
    echo "[PLUGIN] Loading DCNv4 plugin: $DCNV4_PLUGIN  (via $PLUGIN_FLAG)"
    # Shapes lifted verbatim from the v11 entrypoint:
    #   masked_flows=[B,T,2,H,W], masks=[B,T,1,H,W]
    if trtexec \
            --onnx="$RFCNET_ONNX" \
            --saveEngine="$RFCNET_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --builderOptimizationLevel=5 \
            "${PLUGIN_FLAG}=$DCNV4_PLUGIN" \
            --minShapes=masked_flows:1x8x2x256x256,masks:1x8x1x256x256 \
            --optShapes=masked_flows:1x8x2x480x640,masks:1x8x1x480x640 \
            --maxShapes=masked_flows:1x16x2x720x1280,masks:1x16x1x720x1280 ; then
        echo "[OK] RFCNet DCNv4 engine built: $RFCNET_ENGINE"
        BUILT+=("rfcnet_dcnv4_fp16.engine")
        RFCNET_TRT_OK=1
    else
        echo "[WARN] RFCNet engine build FAILED - PyTorch fallback"
        rm -f "$RFCNET_ENGINE"
    fi
fi

if [ "$RFCNET_TRT_OK" = "1" ]; then
    mkdir -p "$(dirname "$RFCNET_LINK")"
    ln -sfn "$RFCNET_ENGINE" "$RFCNET_LINK"
fi

# ---- NeuFlow (ProPainter optical flow) --------------------------------------
NEUFLOW_ONNX="/app/faster-propainter-main/models/neuflow_things.onnx"
NEUFLOW_ENGINE="$ENGINE_DIR/neuflow_things_fp16.engine"
NEUFLOW_LINK="/app/faster-propainter-main/models/neuflow_things_fp16.engine"
NEUFLOW_TRT_OK=0

if [ -f "$NEUFLOW_ENGINE" ]; then
    echo "[OK] NeuFlow engine exists: $NEUFLOW_ENGINE"
    NEUFLOW_TRT_OK=1
elif [ ! -f "$NEUFLOW_ONNX" ]; then
    echo "[WARN] NeuFlow ONNX missing: $NEUFLOW_ONNX"
    echo "       It ships inside faster-propainter-main/models/ (42.2MB), so if it"
    echo "       is absent the image was built wrong - check the COPY of"
    echo "       faster-propainter-main and the *.engine cleanup step."
    echo "       Continuing with the PyTorch RAFT fallback (slower, more VRAM)."
else
    echo "[BUILD] NeuFlow engine missing for sm_${CC}, building (~10 min)..."
    # --tacticSources=+CUDNN retained from v11: the original notes flag that
    # CUBLAS tactics cause issues for this network.
    if trtexec \
            --onnx="$NEUFLOW_ONNX" \
            --saveEngine="$NEUFLOW_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --tacticSources=+CUDNN \
            --builderOptimizationLevel=5 ; then
        echo "[OK] NeuFlow engine built: $NEUFLOW_ENGINE"
        BUILT+=("neuflow_things_fp16.engine")
        NEUFLOW_TRT_OK=1
    else
        echo "[WARN] NeuFlow engine build FAILED - PyTorch RAFT fallback"
        rm -f "$NEUFLOW_ENGINE"
    fi
fi

if [ "$NEUFLOW_TRT_OK" = "1" ]; then
    mkdir -p "$(dirname "$NEUFLOW_LINK")"
    ln -sfn "$NEUFLOW_ENGINE" "$NEUFLOW_LINK"
fi

# -----------------------------------------------------------------------------
# 3c. Push newly built engines back to B2
# -----------------------------------------------------------------------------
# Only engines built during THIS boot are uploaded, so a warm start does zero
# writes. The first 5090 node pays the ~25 min build once; every node after it
# pulls ~90MB in about a minute and never builds again.
if [ "${#BUILT[@]}" -gt 0 ]; then
    if [ "${ENGINE_CACHE_B2:-1}" = "1" ]; then
        echo ""
        echo "[CACHE] uploading ${#BUILT[@]} newly built engine(s) for ${CACHE_KEY}"
        python3 /app/engine_cache.py push "$CACHE_KEY" "$ENGINE_DIR" "${BUILT[@]}" ||             echo "[CACHE] upload had failures - continuing, engines are valid locally"
    else
        echo "[CACHE] ENGINE_CACHE_B2=0 - built ${#BUILT[@]} engine(s), not uploading"
    fi
else
    echo "[CACHE] no engines built this boot - nothing to upload"
fi
echo ""

# -----------------------------------------------------------------------------
# 4. Re-gate feature flags on what actually exists
# -----------------------------------------------------------------------------
# v12 shipped ENABLE_FP8_*=1 unconditionally. FP8 tensor cores need sm_89+;
# on anything older TensorRT either refuses the engine or silently degrades.
if [ "$CC" -lt 89 ] 2>/dev/null; then
    echo "[GATE] sm_${CC} < sm_89: disabling FP8"
    export ENABLE_FP8_TRANSFORMER=0
    export ENABLE_FP8_ENCODER=0
    export ENABLE_FP8_DECODER=0
    export ENABLE_FP8_RFCNET=0
fi

# Never advertise a TRT path whose engine is not present - that turns a clean
# PyTorch fallback into a hard crash.
if [ "$RFCNET_TRT_OK" != "1" ]; then
    export FORCE_TRT_RFCNET=0
    export ENABLE_DCNV4_RFCNET=0
    export ENABLE_FP8_RFCNET=0
fi
if [ "$NEUFLOW_TRT_OK" != "1" ]; then
    export USE_NEUFLOW=0
fi

# Concurrency from VRAM (from the v11 entrypoint). A 5090 has 32GB -> 4.
if [ "$GPU_MEMORY" -ge 20000 ]; then
    CONCURRENCY=${CELERY_CONCURRENCY:-4}
elif [ "$GPU_MEMORY" -ge 10000 ]; then
    CONCURRENCY=${CELERY_CONCURRENCY:-2}
else
    CONCURRENCY=${CELERY_CONCURRENCY:-1}
    # Small cards cannot hold 4 resident SAM2 TRT contexts alongside ProPainter.
    if [ "${NUM_WORKERS:-4}" -gt 1 ]; then
        echo "[GATE] ${GPU_MEMORY}MB VRAM: reducing NUM_WORKERS to 1"
        NUM_WORKERS=1
    fi
fi

echo ""
echo "============================================================"
echo "TensorRT Engine Status  (sm_${CC})"
echo "============================================================"
[ -f "$ENCODER_ENGINE" ] && echo "  SAM2 encoder : ENABLED  $(du -h "$ENCODER_ENGINE" | cut -f1)" || echo "  SAM2 encoder : MISSING"
[ -f "$DECODER_ENGINE" ] && echo "  SAM2 decoder : ENABLED  $(du -h "$DECODER_ENGINE" | cut -f1)" || echo "  SAM2 decoder : MISSING"
[ "$RFCNET_TRT_OK"  = "1" ] && echo "  RFCNet DCNv4 : ENABLED  $(du -h "$RFCNET_ENGINE" | cut -f1)" || echo "  RFCNet DCNv4 : PyTorch fallback"
[ "$NEUFLOW_TRT_OK" = "1" ] && echo "  NeuFlow      : ENABLED  $(du -h "$NEUFLOW_ENGINE" | cut -f1)" || echo "  NeuFlow      : PyTorch RAFT fallback"
echo "  FP8          : ${ENABLE_FP8_TRANSFORMER}/${ENABLE_FP8_ENCODER}/${ENABLE_FP8_DECODER}/${ENABLE_FP8_RFCNET} (xf/enc/dec/rfc)"
echo "============================================================"
echo ""

export PYTHONPATH=/app:/app/sam2_trt_inference:/app/segment-anything-2:/app/faster-propainter-main

# -----------------------------------------------------------------------------
# Notification helpers (kept from both source entrypoints)
# -----------------------------------------------------------------------------
send_notification() {
    MSG="$1"
    if [ -n "$NOTIFY_WEBHOOK_URL" ]; then
        curl -s -X POST "$NOTIFY_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"$MSG\"}" || true
    fi
}

get_crash_reason() {
    case "$1" in
        0)   echo "Normal shutdown (exit 0)" ;;
        1)   echo "Error (exit 1)" ;;
        2)   echo "Bash misuse (exit 2)" ;;
        126) echo "Permission denied (exit 126)" ;;
        127) echo "Command not found (exit 127)" ;;
        130) echo "SIGINT/Ctrl+C (exit 130)" ;;
        134) echo "SIGABRT (exit 134)" ;;
        137) echo "OOM Killed (exit 137)" ;;
        139) echo "Segfault (exit 139)" ;;
        141) echo "Broken pipe (exit 141)" ;;
        143) echo "SIGTERM (exit 143)" ;;
        *)   echo "Unknown (exit $1)" ;;
    esac
}

WORKER_NAME="${NOTIFY_WORKER_NAME:-MRAI-$(hostname)}"
RESTART_COUNT=0
SESSION=mrai
ROLE="${ROLE:-all}"
NUM_WORKERS=${NUM_WORKERS:-4}

# -----------------------------------------------------------------------------
# 5. Launch
# -----------------------------------------------------------------------------
# Each window is registered in WINDOWS as "name:signal:command" so the monitor
# loop below can restart exactly the one that died.
WINDOWS=()

cmd_clicker() { echo "python /app/start_object_server.py --worker-id $1 --port $((5555 + $1))"; }
cmd_sam2()    { echo "celery -A wsl_sam2_worker worker -Q wsl_sam2,wsl_yolo --loglevel=info --pool=solo --without-heartbeat --without-gossip --without-mingle"; }
cmd_prop()    { echo "celery -A server_production2.celery worker -Q celery,propainter --loglevel=info --pool=${CELERY_POOL:-threads} --concurrency=${CONCURRENCY} --without-heartbeat --without-gossip --without-mingle"; }
cmd_api()     { echo "python /app/server_production2.py"; }

case "$ROLE" in
    all)
        for i in $(seq 0 $((NUM_WORKERS - 1))); do
            WINDOWS+=("clicker_$i:clicker-$i-done:$(cmd_clicker $i)")
        done
        WINDOWS+=("sam2:sam2-done:$(cmd_sam2)")
        WINDOWS+=("propainter:propainter-done:$(cmd_prop)")
        WINDOWS+=("api:api-done:$(cmd_api)")
        ;;
    clicker)
        for i in $(seq 0 $((NUM_WORKERS - 1))); do
            WINDOWS+=("clicker_$i:clicker-$i-done:$(cmd_clicker $i)")
        done
        ;;
    sam2)       WINDOWS+=("sam2:sam2-done:$(cmd_sam2)") ;;
    propainter) WINDOWS+=("propainter:propainter-done:$(cmd_prop)") ;;
    api)        WINDOWS+=("api:api-done:$(cmd_api)") ;;
    *)
        echo "[ERROR] Unknown ROLE='$ROLE' (expected: all|clicker|sam2|propainter|api)"
        exit 1
        ;;
esac

start_window() {
    local NAME="$1" SIG="$2" CMD="$3" MODE="$4"
    local WRAPPED="$CMD; echo \$? > /tmp/exit_${NAME}; tmux wait-for -S ${SIG}"
    if [ "$MODE" = "new" ]; then
        tmux new-session -d -s "$SESSION" -n "$NAME" "$WRAPPED"
    else
        tmux kill-window -t "${SESSION}:${NAME}" 2>/dev/null || true
        tmux new-window -t "$SESSION" -n "$NAME" "$WRAPPED"
    fi
}

echo "============================================================"
echo "[START] ROLE=$ROLE  ->  ${#WINDOWS[@]} process(es) in tmux session '$SESSION'"
echo "============================================================"

tmux kill-session -t "$SESSION" 2>/dev/null || true

FIRST=1
for W in "${WINDOWS[@]}"; do
    NAME="${W%%:*}";  REST="${W#*:}"
    SIG="${REST%%:*}"; CMD="${REST#*:}"
    if [ "$FIRST" = "1" ]; then
        start_window "$NAME" "$SIG" "$CMD" new
        FIRST=0
    else
        start_window "$NAME" "$SIG" "$CMD" add
    fi
    echo "  [WINDOW] $NAME"
    sleep 2
done

echo ""
echo "  Attach : docker exec -it <container> tmux attach -t $SESSION"
echo "  Switch : Ctrl+B then window number     Detach: Ctrl+B then D"
echo "============================================================"
echo ""

send_notification "🟢 **$WORKER_NAME** READY | GPU: $GPU_NAME (${GPU_MEMORY}MB, sm_${CC}) | ROLE=$ROLE | RFCNet-TRT=$RFCNET_TRT_OK NeuFlow-TRT=$NEUFLOW_TRT_OK"

# -----------------------------------------------------------------------------
# 6. Monitor - block with zero CPU, restart only what died
# -----------------------------------------------------------------------------
while true; do
    echo "[MONITOR] Waiting for any process to exit..."

    for W in "${WINDOWS[@]}"; do
        REST="${W#*:}"; SIG="${REST%%:*}"
        tmux wait-for "$SIG" &
    done
    wait -n
    # Drop the other waiters so they do not accumulate across restarts.
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true

    CRASHED=""
    for W in "${WINDOWS[@]}"; do
        NAME="${W%%:*}"
        if [ -f "/tmp/exit_${NAME}" ]; then
            EXIT_CODE=$(cat "/tmp/exit_${NAME}")
            rm -f "/tmp/exit_${NAME}"
            CRASHED="$W"
            break
        fi
    done

    if [ -z "$CRASHED" ]; then
        if ! tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[FATAL] tmux session died entirely, exiting container"
            send_notification "🔴 **$WORKER_NAME** tmux session died - container restarting"
            exit 1
        fi
        echo "[WARN] No exit file found, continuing"
        continue
    fi

    NAME="${CRASHED%%:*}"; REST="${CRASHED#*:}"
    SIG="${REST%%:*}";     CMD="${REST#*:}"
    RESTART_COUNT=$((RESTART_COUNT + 1))
    REASON=$(get_crash_reason "$EXIT_CODE")

    echo "[CRASH] $NAME exited: $REASON (restart #$RESTART_COUNT)"
    send_notification "🔴 **$WORKER_NAME** - $NAME crashed: $REASON | restarting (#$RESTART_COUNT)"

    # Reclaim VRAM before bringing the process back.
    python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" 2>/dev/null || true
    sleep 3

    start_window "$NAME" "$SIG" "$CMD" add
    send_notification "🔄 **$WORKER_NAME** - $NAME restarted (#$RESTART_COUNT)"
done
