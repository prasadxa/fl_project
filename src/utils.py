"""
Shared Utilities — utils.py
============================
Common helpers used across server.py and client.py.

Currently provides:
    suppress_flwr_deprecation_warnings()  — filters Flower 1.26.x verbose
                                            "DEPRECATED FEATURE" log records
                                            while keeping all INFO messages.
"""

from __future__ import annotations

import logging as _logging


class _NoDeprecationFilter(_logging.Filter):
    """
    Logging filter that silently drops any log record whose message contains
    the word 'DEPRECATED'.

    Flower 1.26.x emits verbose 'DEPRECATED FEATURE' records for
    start_server / start_numpy_client even when the compat-layer equivalents
    are used.  This filter keeps INFO-level round progress, model saves, and
    all other messages while suppressing only that noise.
    """

    def filter(self, record: _logging.LogRecord) -> bool:  # type: ignore[override]
        return "DEPRECATED" not in record.getMessage()


def suppress_flwr_deprecation_warnings() -> None:
    """
    Attach _NoDeprecationFilter to the root 'flwr' logger.

    Call once at process startup — before any flwr import triggers logging —
    in both server.py and client.py:

        from utils import suppress_flwr_deprecation_warnings
        suppress_flwr_deprecation_warnings()

    The filter is idempotent: calling it multiple times has no additional
    effect because Python's logging framework deduplicates identical filter
    objects on the same logger.
    """
    logger = _logging.getLogger("flwr")
    # Avoid adding the same filter class twice if the function is called
    # more than once in the same process (e.g. during testing).
    for existing in logger.filters:
        if isinstance(existing, _NoDeprecationFilter):
            return
    logger.addFilter(_NoDeprecationFilter())
