"""
GPU Coordinator - Redis-based distributed lock for GPU exclusivity.

Ensures only ONE Celery worker (SAM2 or ProPainter) uses the GPU at a time.
Workers wait (pause) while another worker has the lock - tasks stay in queue.

Usage:
    from gpu_coordinator import GPUCoordinator

    coord = GPUCoordinator(worker_type='sam2')  # or 'propainter'

    with coord.gpu_exclusive(task_id):
        # GPU-intensive work here
        ...
"""

import redis
import time
import threading
import os
import json
from contextlib import contextmanager

# Configuration from environment
REDIS_URL = os.getenv('REDIS_URL')
LOCK_TTL = int(os.getenv('GPU_LOCK_TTL', '120'))  # 2 minutes - auto-expire if worker dies
HEARTBEAT_INTERVAL = int(os.getenv('GPU_HEARTBEAT_INTERVAL', '30'))  # Extend TTL every 30s
POLL_INTERVAL = 2  # Check lock every 2 seconds

# Redis keys
LOCK_KEY = 'gpu:lock'
LAST_TYPE_KEY = 'gpu:lock:last_type'


def _resilient_redis(url):
    """Redis client that survives the proxy dropping a connection mid-command."""
    opts = dict(
        decode_responses=True,
        socket_keepalive=True,
        socket_connect_timeout=10,
        socket_timeout=30,
        health_check_interval=30,
        retry_on_timeout=True,
    )
    try:
        from redis.retry import Retry
        from redis.backoff import ExponentialBackoff
        from redis.exceptions import ConnectionError as _CE, TimeoutError as _TE
        opts['retry'] = Retry(ExponentialBackoff(cap=3, base=0.1), 5)
        opts['retry_on_error'] = [_CE, _TE, ConnectionResetError]
    except Exception as exc:
        print(f"[GPU-COORD] retry unavailable ({exc}); keepalive only")
    return redis.from_url(url, **opts)


class GPUCoordinator:
    """
    Coordinates GPU access between SAM2 and ProPainter Celery workers.

    Features:
    - Redis SETNX lock with TTL (auto-expires on crash)
    - Heartbeat thread extends TTL for long tasks
    - Fair scheduling (alternates between worker types when both waiting)
    """

    def __init__(self, worker_type: str, worker_id: str = None):
        """
        Initialize GPU coordinator.

        Args:
            worker_type: 'sam2' or 'propainter'
            worker_id: Unique identifier (defaults to type + PID)
        """
        if not REDIS_URL:
            raise RuntimeError("REDIS_URL environment variable not set")

        # Every render passes through this lock, and acquire_lock polls Redis in
        # a loop while waiting for the GPU - so this is the connection most
        # exposed to Railway's proxy resetting it. A bare client has no socket
        # timeout and no retry, so a reset landing mid-read propagated straight
        # out of parse_response and failed the whole render; one user lost two
        # renders to exactly that. retry_on_error covers a reset that arrives
        # during a command, not just while connecting.
        self.redis = _resilient_redis(REDIS_URL)
        self.worker_type = worker_type
        self.worker_id = worker_id or f"{worker_type}-{os.getpid()}"
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()

    def _lock_value(self, task_id: str) -> str:
        """Create JSON lock value with metadata."""
        return json.dumps({
            'worker_type': self.worker_type,
            'worker_id': self.worker_id,
            'task_id': task_id,
            'acquired_at': time.time(),
            'last_heartbeat': time.time()
        })

    WAIT_STALE_AFTER = 60       # a waiter that has not refreshed in 60s is gone

    def _waiting_key(self, worker_type: str) -> str:
        return f'gpu:lock:waiting:{worker_type}'

    def _mark_waiting(self):
        """Announce that this worker is waiting, with a timestamp.

        A plain INCR/DECR counter cannot survive a worker being killed mid-wait:
        the decrement never runs and the count stays high forever. Every later
        task then yields to a waiter that does not exist and times out after an
        hour - which took the service down for five hours and could only be
        fixed by resetting the key by hand. A sorted set scored by time is
        self-healing: a dead worker's entry simply ages out.
        """
        try:
            self.redis.zadd(self._waiting_key(self.worker_type),
                            {self.worker_id: time.time()})
        except Exception as exc:
            print(f"[GPU-LOCK] could not mark waiting: {exc}")

    def _unmark_waiting(self):
        try:
            self.redis.zrem(self._waiting_key(self.worker_type), self.worker_id)
        except Exception as exc:
            print(f"[GPU-LOCK] could not clear waiting: {exc}")

    def _count_waiting(self, worker_type: str) -> int:
        """Live waiters of a type, ignoring and pruning stale entries."""
        key = self._waiting_key(worker_type)
        cutoff = time.time() - self.WAIT_STALE_AFTER
        try:
            self.redis.zremrangebyscore(key, '-inf', cutoff)
            return int(self.redis.zcard(key) or 0)
        except Exception as exc:
            # An old deployment left a plain counter here; drop it and start clean.
            if 'WRONGTYPE' in str(exc):
                try:
                    self.redis.delete(key)
                    print(f"[GPU-LOCK] replaced legacy counter at {key}")
                except Exception:
                    pass
            else:
                print(f"[GPU-LOCK] could not count waiters: {exc}")
            return 0

    def _other_type_waiting(self) -> bool:
        """Check if the other worker type has tasks waiting."""
        other = 'propainter' if self.worker_type == 'sam2' else 'sam2'
        return self._count_waiting(other) > 0

    def acquire_lock(self, task_id: str, timeout: int = 3600) -> bool:
        """
        Acquire GPU lock with fair scheduling.

        Blocks until lock is acquired or timeout.
        Uses fair scheduling to alternate between worker types.

        Args:
            task_id: Celery task ID
            timeout: Max seconds to wait (default 1 hour)

        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        log_interval = 10  # Log waiting status every 10 seconds
        last_log_time = 0

        # Register in waiting queue (timestamped, so it cannot leak)
        self._mark_waiting()

        try:
            while time.time() - start_time < timeout:
                # Check fair scheduling - yield if we just ran and other type is waiting
                last_holder = self.redis.get(LAST_TYPE_KEY)
                should_yield = (
                    last_holder == self.worker_type and
                    self._other_type_waiting()
                )

                if not should_yield:
                    # Try to acquire lock with SETNX
                    lock_value = self._lock_value(task_id)
                    if self.redis.set(LOCK_KEY, lock_value, nx=True, ex=LOCK_TTL):
                        # Lock acquired!
                        self.redis.set(LAST_TYPE_KEY, self.worker_type)
                        print(f"[GPU-LOCK] Acquired by {self.worker_id} for task {task_id}")
                        self._start_heartbeat(task_id)
                        return True

                # Lock held by someone else - log status periodically
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    current = self.redis.get(LOCK_KEY)
                    if current:
                        try:
                            holder = json.loads(current)
                            other_type = holder.get('worker_type', 'unknown')
                            other_task = holder.get('task_id', 'unknown')[:8]
                            wait_time = int(current_time - start_time)
                            print(f"[GPU-LOCK] Waiting {wait_time}s... Lock held by {other_type} (task: {other_task}...)")
                        except json.JSONDecodeError:
                            pass
                    elif should_yield:
                        print(f"[GPU-LOCK] Yielding to other worker type (fair scheduling)")
                    last_log_time = current_time

                # Wait before retry, refreshing our own entry so a long but
                # genuine wait is never pruned as stale.
                self._mark_waiting()
                time.sleep(POLL_INTERVAL)

            print(f"[GPU-LOCK] TIMEOUT after {timeout}s waiting for lock")
            return False

        finally:
            # Remove from waiting queue
            self._unmark_waiting()

    def release_lock(self, task_id: str):
        """
        Release GPU lock after task completion.

        Verifies ownership before releasing to prevent accidental
        release of another worker's lock.
        """
        self._stop_heartbeat.set()

        # Verify we own the lock before releasing
        current = self.redis.get(LOCK_KEY)
        if current:
            try:
                holder = json.loads(current)
                if holder.get('worker_id') == self.worker_id:
                    self.redis.delete(LOCK_KEY)
                    print(f"[GPU-LOCK] Released by {self.worker_id} (task: {task_id})")
                else:
                    print(f"[GPU-LOCK] WARNING: Lock owned by {holder.get('worker_id')}, not releasing")
            except json.JSONDecodeError:
                # Corrupted lock data - safe to delete
                self.redis.delete(LOCK_KEY)
                print(f"[GPU-LOCK] Released (corrupted lock data)")

        # Clean up heartbeat thread
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def _start_heartbeat(self, task_id: str):
        """Start background thread to extend lock TTL for long tasks."""
        self._stop_heartbeat.clear()

        def heartbeat_loop():
            while not self._stop_heartbeat.wait(HEARTBEAT_INTERVAL):
                try:
                    # Extend lock TTL (only if we still own it)
                    current = self.redis.get(LOCK_KEY)
                    if current:
                        holder = json.loads(current)
                        if holder.get('worker_id') == self.worker_id:
                            lock_value = self._lock_value(task_id)
                            self.redis.set(LOCK_KEY, lock_value, xx=True, ex=LOCK_TTL)
                except Exception as e:
                    print(f"[GPU-LOCK] Heartbeat error: {e}")

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    @contextmanager
    def gpu_exclusive(self, task_id: str, timeout: int = 3600):
        """
        Context manager for exclusive GPU access.

        Usage:
            with coord.gpu_exclusive(task_id):
                # GPU work here
                ...

        Args:
            task_id: Celery task ID
            timeout: Max seconds to wait for lock

        Raises:
            RuntimeError: If lock cannot be acquired within timeout
        """
        acquired = self.acquire_lock(task_id, timeout=timeout)
        if not acquired:
            raise RuntimeError(f"[GPU-LOCK] Failed to acquire GPU lock for task {task_id} within {timeout}s")
        try:
            yield
        finally:
            self.release_lock(task_id)

    def force_release(self):
        """
        Force release any lock (for admin/debugging).

        WARNING: Only use this for stuck locks after verifying
        no worker is actually using the GPU.
        """
        self.redis.delete(LOCK_KEY)
        print(f"[GPU-LOCK] FORCE RELEASED by {self.worker_id}")

    def get_lock_status(self) -> dict:
        """Get current lock status for monitoring."""
        current = self.redis.get(LOCK_KEY)
        waiting_sam2 = self._count_waiting('sam2')
        waiting_propainter = self._count_waiting('propainter')

        status = {
            'locked': current is not None,
            'waiting_sam2': waiting_sam2,
            'waiting_propainter': waiting_propainter,
        }

        if current:
            try:
                holder = json.loads(current)
                status['holder'] = holder
                status['age_seconds'] = time.time() - holder.get('acquired_at', 0)
            except json.JSONDecodeError:
                status['holder'] = 'corrupted'

        return status


# Singleton instances for each worker type
_coordinators = {}

def get_coordinator(worker_type: str) -> GPUCoordinator:
    """Get or create a GPU coordinator for the given worker type."""
    if worker_type not in _coordinators:
        _coordinators[worker_type] = GPUCoordinator(worker_type)
    return _coordinators[worker_type]
