"""Privacy-safe diagnostics for handled failures."""

import logging


logger = logging.getLogger(__name__)


def log_failure(stage: str, error: BaseException) -> None:
    """Log a fixed stage and exception metadata, never the exception message."""
    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        status = None
    if status is None:
        logger.warning("Failure stage=%s exception=%s", stage, type(error).__name__)
    else:
        logger.warning(
            "Failure stage=%s exception=%s http_status=%d",
            stage,
            type(error).__name__,
            status,
        )
