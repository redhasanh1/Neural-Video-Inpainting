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

REDIS_URL = os.getenv('REDIS_URL')
QUEUES = ('propainter', 'wsl_sam2', 'celery')
CHECK_EVERY = int(os.getenv('WATCHDOG_INTERVAL', '60'))
STALL_AFTER = int(os.getenv('WATCHDOG_STALL_SECONDS', '600'))
# A single wedged render does not back the queue up: the other jobs get taken,
# depth falls to zero, and the stall check above stays quiet while one task
# holds the GPU lock forever. That is exactly how the box went to 116% memory
# with nothing queued. So watch the lock holder's own progress too.
WEDGE_AFTER = int(os.getenv('WATCHDOG_WEDGE_SECONDS', '1200'))
ENABLED = os.getenv('WATCHDOG_ENABLED', '1') == '1'
WORKER = os.getenv('NOTIFY_WORKER_NAME') or os.uname().nodename
# A watchdog that dies quietly is worse than none - it leaves you believing
# something is watching. Publish a short-lived key so the box can be asked
# from outside whether its watchdog is still breathing.
ALIVE_KEY = f'watchdog:alive:{WORKER}'
ALIVE_TTL = 180


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

            beat(r, state='stalling' if stalled_since else
                       ('wedging' if wedge_since else 'ok'),
                 depth=depth, completed=done,
                 stalled_for=int(time.time()-stalled_since) if stalled_since else 0,
                 wedged_for=int(time.time()-wedge_since) if wedge_since else 0,
                 holder=task_id)
        except Exception as exc:
            log(f"check failed ({exc}); will retry")
            try:
                r = connect()
            except Exception:
                pass


if __name__ == '__main__':
    main()
