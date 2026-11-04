"""Throttled command-line progress on stderr, leaving stdout usable for results."""

import sys
import time


class CLIProgress:
    def __init__(self, label, *, stream=None, interval=1.0):
        self.label = label
        self.stream = sys.stderr if stream is None else stream
        self.interval = interval
        self.started = time.monotonic()
        self.last = None
        self.finished = False

    def __call__(self, fraction):
        if self.finished:
            return
        now = time.monotonic()
        if self.last is None or fraction >= 1 or now - self.last >= self.interval:
            print(
                f"{self.label}: {fraction:.1%} · {now - self.started:.1f}s elapsed",
                file=self.stream,
                flush=True,
            )
            self.last = now
        self.finished = fraction >= 1
