from __future__ import annotations
import logging, sys
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

def _safe_add_logger_name(logger, method_name, event_dict):
    name = getattr(logger, "name", None)
    if name:
        event_dict["logger"] = name
    return event_dict
def configure_logging(json_output: bool = True) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    if not STRUCTLOG_AVAILABLE:
        return
    # NOTE: structlog.stdlib.add_logger_name reads `logger.name` off the *underlying*
    # logger object. That attribute exists on stdlib logging.Logger instances, which is
    # what you get from `logger_factory=structlog.stdlib.LoggerFactory()`. This module
    # uses `PrintLoggerFactory()` instead (plain stdout, no stdlib logging config
    # required), whose PrintLogger has no `.name` attribute — combining the two raises
    # `AttributeError: 'PrintLogger' object has no attribute 'name'` on the very first
    # log call, which means on server startup, before any request is served. Found this
    # by actually running the app end-to-end, not by inspection — it's present verbatim
    # in the original file. Fix: drop add_logger_name from the processor chain and bind
    # the name explicitly in get_logger() below instead, which works with any factory.
    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            _safe_add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict, logger_factory=structlog.PrintLoggerFactory(), cache_logger_on_first_use=True,
    )

def get_logger(name: str = "ophthalmoai"):
    if STRUCTLOG_AVAILABLE:
        # .bind(logger=name) replaces the removed add_logger_name processor: every event
        # from this logger still carries which module emitted it, just via a bound
        # context key instead of a processor that assumed a stdlib Logger.
        return structlog.get_logger(name).bind(logger=name)
    return _StdlibLoggerAdapter(logging.getLogger(name))

class _StdlibLoggerAdapter:
    def __init__(self, logger): self._logger = logger
    def _fmt(self, event, **kwargs):
        if not kwargs: return event
        return f"{event} | " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    def info(self, event, **kwargs): self._logger.info(self._fmt(event, **kwargs))
    def warning(self, event, **kwargs): self._logger.warning(self._fmt(event, **kwargs))
    def error(self, event, **kwargs): self._logger.error(self._fmt(event, **kwargs))
    def exception(self, event, **kwargs): self._logger.exception(self._fmt(event, **kwargs))
