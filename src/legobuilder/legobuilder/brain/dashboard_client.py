"""Non-blocking HTTP client for posting events to the dashboard server.

Provides a factory that returns a fire-and-forget post_event(event_dict)
function.  Uses a daemon thread + queue so the ROS spin loop is never
blocked by HTTP I/O.  If the dashboard server is unreachable, events
are silently dropped.
"""

import json
import queue
import threading
import urllib.request


def create_post_event_fn(server_url: str):
    """Return a non-blocking ``post_event(event_dict)`` callable.

    Arguments
    ---------
    server_url : str
        Base URL of the dashboard server (e.g. ``http://localhost:8001``).

    Returns
    -------
    callable
        ``post_event(event_dict)`` that enqueues the event for async POST.
    """
    endpoint = f"{server_url.rstrip('/')}/events"
    _queue: queue.Queue = queue.Queue(maxsize=100)

    def _worker():
        while True:
            event = _queue.get()
            if event is None:
                break
            try:
                data = json.dumps(event, default=str).encode()
                req = urllib.request.Request(
                    endpoint,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=1)
            except Exception:
                pass  # fire-and-forget

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    def post_event(event_dict):
        try:
            _queue.put_nowait(event_dict)
        except queue.Full:
            pass  # drop if backed up

    return post_event
