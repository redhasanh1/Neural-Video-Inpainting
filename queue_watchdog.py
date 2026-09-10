"""
Restarts the GPU worker when it stops consuming.

The failure this exists for: the Salad container reports state=running and
ready=true while the Celery process inside it has stopped taking work. Nothing
in the stack notices — the queue simply stops draining, and every job submitted
sits at "Tracking in progress..." forever. It ran that way for hours before
anyone looked at the broker.

The signal is unambiguous: a queue with items in it whose depth does not fall.
Run this as a Railway cron or a small always-on service.
"""
import os
import sys
import time

import redis
import requests

REDIS_URL = os.environ["REDIS_URL"]
SALAD_KEY = os.environ["SALAD_API_KEY"]
SALAD_ORG = os.environ.get("SALAD_ORG", "waterz")
SALAD_PROJECT = os.environ.get("SALAD_PROJECT", "default")
SALAD_GROUP = os.environ.get("SALAD_GROUP", "mrai-5090-prod")
NTFY = os.environ.get("NTFY_TOPIC", "watermarkz-workers-9f31c")

QUEUES = ["wsl_sam2", "propainter", "wsl_yolo"]
# Two consecutive stalled samples, so a job that legitimately takes a while to
# be picked up is not mistaken for a wedged worker.
STALL_SAMPLES = 2
SAMPLE_SECONDS = 90

SALAD_BASE = (
    f"https://api.salad.com/api/public/organizations/{SALAD_ORG}"
    f"/projects/{SALAD_PROJECT}/containers/{SALAD_GROUP}"
)
# Cloudflare rejects urllib on POST; a browser-ish UA on requests is fine.
HEADERS = {"Salad-Api-Key": SALAD_KEY, "User-Agent": "objectremoverai-watchdog/1"}


def depths(client):
    return {q: client.llen(q) for q in QUEUES}


def notify(message):
    try:
        requests.post(f"https://ntfy.sh/{NTFY}", data=message.encode(), timeout=10)
    except Exception as exc:
        print(f"[watchdog] notify failed: {exc}")


def restart_worker():
    r = requests.get(f"{SALAD_BASE}/instances", headers=HEADERS, timeout=30)
    r.raise_for_status()
    instances = r.json().get("instances", [])
    if not instances:
        notify("watchdog: queue stalled and no worker instance exists")
        return False

    instance_id = instances[0]["id"]
    r = requests.post(
        f"{SALAD_BASE}/instances/{instance_id}/restart", headers=HEADERS, timeout=40
    )
    ok = r.status_code in (200, 202, 204)
    print(f"[watchdog] restart {instance_id}: HTTP {r.status_code}")
    return ok


def main():
    client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=20)

    previous = depths(client)
    stalled = 0

    while True:
        time.sleep(SAMPLE_SECONDS)
        current = depths(client)

        # Stalled means: work is waiting, and not one queue moved.
        waiting = any(v > 0 for v in current.values())
        unchanged = current == previous
        stalled = stalled + 1 if (waiting and unchanged) else 0

        print(f"[watchdog] {current} stalled={stalled}", flush=True)

        if stalled >= STALL_SAMPLES:
            total = sum(current.values())
            minutes = STALL_SAMPLES * SAMPLE_SECONDS // 60
            print(f"[watchdog] {total} job(s) stuck for {minutes}m — restarting")
            notify(f"Worker wedged: {total} job(s) unmoved for {minutes}m. Restarting.")
            if restart_worker():
                notify("Worker restart requested.")
            stalled = 0
            time.sleep(120)  # let it boot before sampling again
            previous = depths(client)
            continue

        previous = current


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
