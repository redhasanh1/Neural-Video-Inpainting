"""Real CUDA health checks and fatal-GPU handling.

nvidia-smi printing a device name proves nothing. On 2026-10-25 this box came
back from a memory recycle reporting "No GPU (0MB VRAM)", announced itself
READY anyway, and spent four and a half hours accepting renders it could not
run, failing each one at SAM2 model load and spending a user credit on it.

So anything that gates work on the GPU has to actually allocate, run a kernel
and synchronize. And when CUDA dies underneath a running worker, the node is
gone: ask SaladCloud to move us rather than retrying into a poisoned context.
"""

import os

FATAL_FLAG = '/tmp/gpu-fatal'
HEALTH_FILE = '/tmp/gpu-health'
HEALTH_MAX_AGE = 900

# Errors that mean the GPU or its driver has gone away. Ordinary out-of-memory
# is deliberately absent: that is the VRAM watchdog's problem and a render that
# was simply too big must not cost us the whole node. "device-side assert" is
# absent for the same reason in reverse, it is a bug in our own kernel indices
# and moving node would just loop.
FATAL_PATTERNS = (
    'cumemalloc failed',
    'cuda error: unknown error',
    'failed to initialize nvml',
    'gpu has fallen off the bus',
    'no cuda-capable device',
    'cuda driver version is insufficient',
    'system has unsupported display driver',
)

# A bare "unknown error" is far too broad to move a node on, so a match also
# has to name the GPU stack somewhere in the message.
GPU_WORDS = ('cuda', 'cumemalloc', 'nvml', 'nvidia', 'gpu')


def cuda_healthy():
    """Allocate, run a kernel and synchronize. Returns (ok, detail, vram_mb)."""
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return False, 'no CUDA device', 0
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        probe = torch.ones((256, 256), dtype=torch.float32, device='cuda')
        total = (probe + 1).sum()
        torch.cuda.synchronize()
        del probe
        if int(total.item()) != 256 * 256 * 2:
            return False, 'CUDA returned the wrong answer', vram
        return True, name, vram
    except Exception as exc:
        return False, '{}: {}'.format(type(exc).__name__, exc), 0


def is_fatal(exc):
    """True when this error means the node's GPU is gone rather than busy."""
    name = type(exc).__name__.lower()
    text = '{}: {}'.format(type(exc).__name__, exc).lower()
    if 'outofmemory' in name or 'out of memory' in text:
        return False
    if not any(word in text for word in GPU_WORDS):
        return False
    return any(pattern in text for pattern in FATAL_PATTERNS)


def mark_fatal(reason):
    """Leave a flag the liveness probe can see, so the node gets replaced
    even if we die before the reallocate call lands."""
    try:
        with open(FATAL_FLAG, 'w') as handle:
            handle.write(str(reason)[:500])
    except Exception:
        pass


def write_health(ok):
    """Record the last in-process CUDA verdict for the liveness probe.

    The worker already holds a live CUDA context, so checking here costs
    nothing. A probe that imported torch every thirty seconds would add
    ~500MB of RSS per run to a container that OOM-killed at 9553MB of its
    8192MB limit the day before, and could fail on VRAM pressure during a
    perfectly good render and move the node out from under it.
    """
    import time
    try:
        with open(HEALTH_FILE, 'w') as handle:
            handle.write('{} {}'.format(int(time.time()), 'ok' if ok else 'bad'))
    except Exception:
        pass


def reallocate(reason):
    """Ask SaladCloud to move this container to another node.

    Uses the instance metadata service, so no API key has to live in the
    container. Returns True when SaladCloud accepted the request.
    """
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            'http://169.254.169.254/v1/reallocate',
            data=json.dumps({'reason': str(reason)[:200]}).encode(),
            method='POST',
            headers={'Metadata': 'true', 'Content-Type': 'application/json'})
        # IMDS is link-local; a proxy env var would send this to the wrong place.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=20) as resp:
            return resp.status in (200, 202, 204)
    except Exception as exc:
        print('[GPU] reallocate request failed: {}'.format(exc), flush=True)
        return False


def probe_ok():
    """Cheap liveness verdict for the exec probe: no torch, no CUDA context."""
    import subprocess
    import time
    if os.path.exists(FATAL_FLAG):
        return False, 'fatal flag set'
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return False, 'nvidia-smi exit {}'.format(out.returncode)
        vram = int(out.stdout.strip().splitlines()[0].strip())
        if vram < 1000:
            return False, 'only {}MB VRAM visible'.format(vram)
    except Exception as exc:
        return False, 'nvidia-smi failed: {}'.format(exc)
    try:
        with open(HEALTH_FILE) as handle:
            stamp, verdict = handle.read().split()
        age = time.time() - int(stamp)
        if verdict != 'ok' and age < HEALTH_MAX_AGE:
            return False, 'worker reported CUDA bad {}s ago'.format(int(age))
    except Exception:
        pass
    return True, 'ok'


if __name__ == '__main__':
    import sys
    if '--full' in sys.argv:
        ok, detail, vram = cuda_healthy()
        print('CUDA {}: {} ({}MB)'.format('OK' if ok else 'FAIL', detail, vram))
    else:
        ok, detail = probe_ok()
        print('PROBE {}: {}'.format('OK' if ok else 'FAIL', detail))
    sys.exit(0 if ok else 1)
