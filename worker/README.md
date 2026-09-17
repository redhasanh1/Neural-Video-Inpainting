# GPU worker sources

These are the files the SaladCloud GPU worker actually executes. They live in
the `humblewoslayer/watermarkz-unified` image (base `v7-3090`) and were not
previously tracked here, which made it easy to patch the wrong file: the worker
runs `server_production2.py`, NOT the repo-root `server_production.py` (that one
is the Railway API).

| file | what runs it |
|---|---|
| `server_production2.py`        | `celery -A server_production2.celery worker -Q celery,propainter` and `python /app/server_production2.py` |
| `start_object_server.py`       | 4x clicker processes - consumes `sam2:selection:request` (click-to-select) |
| `wsl_sam2_worker.py`           | `celery -A wsl_sam2_worker worker -Q wsl_sam2,wsl_yolo` |
| `docker-entrypoint-unified.sh` | container entrypoint, launches the above in tmux by ROLE |

Deployed by appending them over the base image:

    tar --format ustar -czf patch.tar.gz app/    # app/<file>
    crane append -b <repo>:v7-3090-netfix -f patch.tar.gz -t <repo>:v7-3090-workerfix

Known gap: the entrypoint watchdog only restarts a process that *exits*
(`wait -n` on a tmux signal). A process that hangs while alive is never
restarted, which is why a wedged clicker silently stopped serving clicks.
