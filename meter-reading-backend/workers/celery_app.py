"""
celery_app.py

Defines the shared Celery application object and its queue routing rules.
Both torch_worker.py and paddle_worker.py import THIS object rather than
creating their own -- that's what makes them "the same team" reading the
same task board, instead of two separate systems that can't see each
other's work.

This file defines no actual tasks. It's pure configuration:
  1. Where the task queue lives (the broker).
  2. Where task RESULTS get stored once a task finishes (the result backend).
  3. Which task goes to which queue (task_routes).
"""

from celery import Celery

# Redis serves two roles here, at two different URLs' worth of database
# index (0 and 1) just to keep them visually separate in logs -- they
# could be the same index, this is purely for readability while debugging:
#
#   - BROKER: the "task board" itself. When the API enqueues a task,
#     the task's name + arguments get written here. A worker listening
#     on the matching queue picks it up from here.
#   - RESULT BACKEND: once a worker finishes a task, where does the
#     return value go? Redis again here (simplest option) -- Celery
#     also supports Postgres, RabbitMQ, etc. as backends, but for a
#     small project reusing Redis for both avoids running a second service.
REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery(
    "meter_reading",   # just an internal name for this Celery app, shows up in logs
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# task_routes is the rule that answers: "a task with this name -- which
# queue does it go on?" This is what keeps torch work and paddle work
# physically separated. Without this, Celery would dump every task onto
# one default queue, and whichever worker happened to be free would grab
# it -- which is exactly what we can't allow, since a paddle worker
# picking up a torch task would try to import ultralytics into a process
# that never should have it (or vice versa).
#
# The pattern "workers.torch_worker.*" matches any task whose full name
# starts with that string -- i.e. any task DEFINED in torch_worker.py.
# Celery names tasks after the Python module they're defined in by
# default, so this works automatically as long as the task functions
# actually live in torch_worker.py / paddle_worker.py.
celery_app.conf.task_routes = {
    "workers.torch_worker.*": {"queue": "torch_queue"},
    "workers.paddle_worker.*": {"queue": "paddle_queue"},
}

# Belt-and-suspenders: even if a task somehow doesn't match either
# pattern above, send it to torch_queue rather than silently defaulting
# to a queue nobody's listening on (which would make the task hang
# forever with no error).
celery_app.conf.task_default_queue = "torch_queue"