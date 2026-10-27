"""Restart the container when renders stop being consumed.

The render workers have died several times without anything noticing: the
container keeps reporting ready, masks keep being produced, and jobs pile up
behind a propainter worker that no longer exists. The last time this happened
30 renders were queued before anyone looked. The tmux supervisor is meant to
restart a dead window but has not been reliable, so this watches the only thing
that actually matters - whether queued work is moving - and exits the container
when it is not. Salad's restart policy then rebuilds everything from scratch.

Deliberately dumb: no assumptions about which process died or why.
"""
import json
import os
import signal
import sys
import time

import redis

import gpu_preflight

REDIS_URL = os.getenv('REDIS_URL')
QUEUES = ('propainter', 'wsl_sam2', 'celery')
CHECK_EVERY = int(os.getenv('WATCHDOG_INTERVAL', '60'))
STALL_AFTER = int(os.getenv('WATCHDOG_STALL_SECONDS', '600'))
# A single wedged render does not back the queue up: the other jobs get taken,
# depth falls to zero, and the stall check above stays quiet while one task
# holds the GPU lock forever. That is exactly how the box went to 116% memory
# with nothing queued. So watch the lock holder's own progress too.
WEDGE_AFTER = int(os.getenv('WATCHDOG_WEDGE_SECONDS', '1200'))
# How many renders may fail back to back, with nothing succeeding in
# between, before we treat this box as broken rather than busy.
FAIL_STREAK = int(os.getenv('WATCHDOG_FAIL_STREAK', '3'))
ENABLED = os.getenv('WATCHDOG_ENABLED', '1') == '1'
WORKER = os.getenv('NOTIFY_WORKER_NAME') or os.uname().nodename
# A watchdog that dies quietly is worse than none - it leaves you believing
# something is watching. Publish a short-lived key so the box can be asked
# from outside whether its watchdog is still breathing.
ALIVE_KEY = f'watchdog:alive:{WORKER}'
ALIVE_TTL = 180


def gpu_vram_mb():
    """Total VRAM in MB per nvidia-smi, or 0 when the GPU has gone away.

    Deliberately shells out instead of importing torch: this process has to
    stay small, and a dead driver is visible to nvidia-smi either way.
    """
    try:
        import subprocess
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return 0
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return 0


def beat(r, **fields):
    try:
        r.setex(ALIVE_KEY, ALIVE_TTL,
                json.dumps(dict(worker=WORKER, at=int(time.time()), **fields)))
    except Exception:
        pass


def log(msg):
    print(f"[WATCHDOG] {msg}", flush=True)


def connect():
    return redis.from_url(
        REDIS_URL, decode_responses=True,
        socket_keepalive=True, socket_connect_timeout=10,
        socket_timeout=30, health_check_interval=30, retry_on_timeout=True,
    )


def lock_progress(r):
    """(task_id, progress fingerprint) for whoever holds the GPU lock.

    The fingerprint is the task's whole result blob, so any progress update at
    all counts as movement. The lock value itself is useless for this: its
    heartbeat rewrites acquired_at every 30s, so it keeps changing even when
    the render behind it has not moved in ten minutes.
    """
    raw = r.get('gpu:lock')
    if not raw:
        return None, None
    try:
        task_id = json.loads(raw).get('task_id')
    except Exception:
        return None, None
    if not task_id:
        return None, None
    return task_id, r.get(f'celery-task-meta-{task_id}') or 'nometa'


def restart(r, why, **fields):
    log(f"{why} - restarting container")
    beat(r, state='restarting', why=why, **fields)
    sys.stdout.flush()
    # If signalling PID 1 ever fails the watchdog must not quietly carry on
    # pretending to guard the box: that is the same silent-no-op failure it
    # exists to catch. Say so loudly, publish it, and take this process down so
    # the missing heartbeat is visible from outside.
    try:
        os.kill(1, signal.SIGTERM)
        time.sleep(30)
        os.kill(1, signal.SIGKILL)
    except BaseException as exc:
        log(f"FATAL: could not signal PID 1 ({type(exc).__name__}: {exc}); "
            f"watchdog is not able to restart this container")
        beat(r, state='impotent', why=str(exc)[:120])
        sys.stdout.flush()
        os._exit(1)


def completed_count(r):
    """How many results exist. If work is flowing this keeps changing."""
    n = 0
    for _ in r.scan_iter(match='celery-task-meta-*', count=1000):
        n += 1
    return n


def result_health(r, seen=None):
    """Return (successes, failures, fatal_text_from_new_failures, ids).

    completed_count() counts every result blob whatever its status, which is
    exactly why a box that failed every render in three seconds read as a
    healthy, fast, empty queue for four and a half hours on 2026-10-25. Work
    "flowing" is not the same as work succeeding.

    Only failures that appeared since the previous look can set fatal_text.
    Result blobs live for a day, so scoring every one of them would let a
    GPU error from this morning justify reallocating the node this evening
    over three unrelated failures.
    """
    ok = bad = 0
    fatal = None
    ids = set()
    for key in r.scan_iter(match='celery-task-meta-*', count=1000):
        raw = r.get(key)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        task_id = key[len('celery-task-meta-'):]
        ids.add(task_id)
        status = data.get('status')
        if status == 'SUCCESS':
            ok += 1
        elif status == 'FAILURE':
            bad += 1
            if seen is not None and task_id in seen:
                continue
            try:
                text = json.dumps(data.get('result'))[:600]
            except Exception:
                continue
            if gpu_preflight.is_fatal(RuntimeError(text)):
                fatal = text[:200]
    return ok, bad, fatal, ids


def main():
    if not ENABLED:
        log("disabled via WATCHDOG_ENABLED=0")
        return
    if not REDIS_URL:
        log("no REDIS_URL; not watching")
        return

    log(f"watching {QUEUES}, stall threshold {STALL_AFTER}s, "
        f"wedge threshold {WEDGE_AFTER}s")
    r = connect()
    stalled_since = None
    last_done = -1
    last_ok = last_bad = -1
    fail_streak = 0
    seen_ids = None         # task ids already scored; None means first look
    wedge_mark = None       # (task_id, fingerprint) last seen on the lock
    wedge_since = None
    beat(r, state='starting')

    while True:
        time.sleep(CHECK_EVERY)
        try:
            depth = sum(r.llen(q) for q in QUEUES)
            done = completed_count(r)

            # Work is only stalled when something is waiting AND nothing has
            # finished since the last look. A busy render that takes ten
            # minutes still changes nothing in the queue, so the completed
            # count is what separates "slow" from "dead".
            if depth > 0 and done == last_done:
                if stalled_since is None:
                    stalled_since = time.time()
                    log(f"queue has {depth} job(s) and nothing completed; watching")
                elif time.time() - stalled_since >= STALL_AFTER:
                    restart(r, f"STALLED: {depth} job(s) queued, nothing "
                               f"completed for {int(time.time()-stalled_since)}s",
                            depth=depth, completed=done)
                    return
            else:
                if stalled_since is not None:
                    log(f"recovered: depth={depth} completed={done}")
                stalled_since = None
            last_done = done

            task_id, fingerprint = lock_progress(r)
            mark = (task_id, fingerprint) if task_id else None
            if mark and mark == wedge_mark:
                if wedge_since is None:
                    wedge_since = time.time()
                elif time.time() - wedge_since >= WEDGE_AFTER:
                    restart(r, f"WEDGED: {task_id} has held the GPU lock with "
                               f"no progress for {int(time.time()-wedge_since)}s",
                            task=task_id, depth=depth)
                    return
            else:
                wedge_since = None
            wedge_mark = mark

            # A run of failures with nothing succeeding in between means the
            # box is broken, not busy. The stall and wedge checks above both
            # miss this because the queue drains perfectly well when every
            # job fails in three seconds.
            ok_n, bad_n, fatal_text, seen_ids = result_health(r, seen_ids)
            if last_ok >= 0:
                if ok_n > last_ok:
                    fail_streak = 0
                elif bad_n > last_bad:
                    fail_streak += bad_n - last_bad
            last_ok, last_bad = ok_n, bad_n

            if fail_streak >= FAIL_STREAK:
                why = (f"{fail_streak} renders failed with none succeeding"
                       + (f"; GPU fatal: {fatal_text}" if fatal_text else ""))
                log(f"FAILING: {why}")
                beat(r, state='failing', depth=depth, completed=done,
                     fail_streak=fail_streak, vram=gpu_vram_mb())
                if fatal_text:
                    gpu_preflight.mark_fatal(fatal_text)
                    if gpu_preflight.reallocate(f'repeated GPU failures: {why}'):
                        log("asked SaladCloud to move this container")
                        return
                restart(r, why, depth=depth)
                return

            beat(r, state='stalling' if stalled_since else
                       ('wedging' if wedge_since else 'ok'),
                 depth=depth, completed=done,
                 stalled_for=int(time.time()-stalled_since) if stalled_since else 0,
                 wedged_for=int(time.time()-wedge_since) if wedge_since else 0,
                 holder=task_id, vram=gpu_vram_mb(), fail_streak=fail_streak)
        except Exception as exc:
            log(f"check failed ({exc}); will retry")
            try:
                r = connect()
            except Exception:
                pass


if __name__ == '__main__':
    main()
