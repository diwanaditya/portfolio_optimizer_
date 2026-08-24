"""
Gunicorn production server configuration -- runs the FastAPI app behind
Gunicorn's process manager with Uvicorn's ASGI worker class. This is the
standard, battle-tested way to run a FastAPI app in production: Gunicorn
handles worker process management (restart on crash, graceful reload,
pre-fork model), Uvicorn's worker handles the actual ASGI/async serving.

Uvicorn alone (`uvicorn app:app`) is fine for development but runs a
single process -- Gunicorn is what gives you multiple worker processes
sharing one port, which is what an actual production load needs.
"""
import multiprocessing
import os

# Bind: 0.0.0.0 here is correct BECAUSE this process is meant to sit
# behind a reverse proxy (see nginx.conf) that terminates TLS and is the
# only thing actually exposed to the internet -- gunicorn itself should
# never be the internet-facing edge.
bind = f"{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8000')}"

worker_class = "uvicorn.workers.UvicornWorker"

# Default: 2x CPU cores + 1 (standard Gunicorn sizing guideline), capped
# at 8 for a typical small-to-mid deployment -- override via WORKERS env
# var once you have real load data telling you the right number.
_default_workers = min((multiprocessing.cpu_count() * 2) + 1, 8)
workers = int(os.environ.get("WORKERS", _default_workers))

worker_connections = 1000
timeout = 30                  # kill and restart a worker stuck longer than this
graceful_timeout = 30         # time to finish in-flight requests on reload/shutdown
keepalive = 5

# Restart workers periodically to guard against slow memory leaks in any
# dependency -- jitter avoids all workers restarting simultaneously.
max_requests = 2000
max_requests_jitter = 200

accesslog = "-"    # stdout -- let the container runtime/log aggregator collect it
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# JSON access log format when LOG_FORMAT=json, matching settings.py's
# log_format option -- makes gunicorn's own access logs parseable by the
# same log aggregation pipeline as the app's structured logs.
if os.environ.get("LOG_FORMAT", "json") == "json":
    access_log_format = (
        '{"remote_addr":"%(h)s","method":"%(m)s","path":"%(U)s",'
        '"status":%(s)s,"response_time_ms":%(D)s,"user_agent":"%(a)s"}'
    )

preload_app = True   # load the app once in the master process before forking workers,
                       # saves memory (shared via copy-on-write) and fails fast on import errors
