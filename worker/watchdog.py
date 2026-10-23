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

    log(f"watching {QUEUES}, stall threshold {STALL_AFTER}s")
    r = connect()
    stalled_since = None
    last_done = -1
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
                    log(f"STALLED: {depth} job(s) queued, nothing completed for "
                        f"{int(time.time()-stalled_since)}s - restarting container")
                    beat(r, state='restarting', depth=depth, completed=done)
                    sys.stdout.flush()
                    os.kill(1, signal.SIGTERM)
                    time.sleep(30)
                    os.kill(1, signal.SIGKILL)
                    return
            else:
                if stalled_since is not None:
                    log(f"recovered: depth={depth} completed={done}")
                stalled_since = None
            last_done = done
            beat(r, state='stalling' if stalled_since else 'ok',
                 depth=depth, completed=done,
                 stalled_for=int(time.time()-stalled_since) if stalled_since else 0)
        except Exception as exc:
            log(f"check failed ({exc}); will retry")
            try:
                r = connect()
            except Exception:
                pass


if __name__ == '__main__':
    main()
