"""Lightweight profiling utilities for timing code blocks.

Provides a reusable TimingContext context manager that measures
elapsed wall-clock time using time.perf_counter.
"""

import time


class TimingContext:
    """Context manager that measures elapsed wall-clock time.

    Stores the elapsed time in milliseconds on the elapsed_ms attribute
    after exiting the context.  Optionally logs at debug level via an
    injected logger.

    Attributes
    ----------
    label : str
        Human-readable label for the timing measurement.
    elapsed_ms : float
        Wall-clock time in milliseconds (set after __exit__).
    """

    def __init__(self, label: str, logger=None):
        """Initialise the timing context.

        Arguments
        ---------
        label : str
            Human-readable label for the timing measurement.
        logger : optional
            ROS-compatible logger. If provided, logs elapsed time
            at debug level on exit.
        """
        self.label = label
        self.elapsed_ms = 0.0
        self.logger = logger

    def __enter__(self):
        """Start the timer and return the context instance."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop the timer and optionally log the elapsed time."""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self.logger:
            self.logger.debug(
                f"[TIMING] {self.label}: {self.elapsed_ms:.2f}ms"
            )
