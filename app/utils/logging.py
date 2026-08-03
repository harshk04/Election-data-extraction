"""Centralized logging utilities."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from app.config.settings import get_settings


_STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class ContextAwareFormatter(logging.Formatter):
    """Formatter that appends non-standard LogRecord fields as key-value context."""

    def format(self, record: logging.LogRecord) -> str:
        base_message = super().format(record)
        context = self._extract_context(record)
        if not context:
            return base_message

        context_text = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
        return f"{base_message} | {context_text}"

    @staticmethod
    def _extract_context(record: logging.LogRecord) -> dict[str, Any]:
        """Return custom LogRecord fields added through `extra`."""
        return {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_FIELDS and not key.startswith("_")
        }


def configure_logging() -> None:
    """Configure application-wide logging."""
    settings = get_settings()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    formatter = ContextAwareFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_logger.level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if settings.log_to_file:
        log_path = _prepare_log_path(settings.log_dir, settings.log_file_name)
        file_handler = logging.FileHandler(filename=log_path, encoding="utf-8")
        file_handler.setLevel(root_logger.level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log an exception with consistent structured context."""
    logger.exception(message, extra={key: str(value) for key, value in context.items()})


def _prepare_log_path(log_dir: Path, log_file_name: str) -> Path:
    """Ensure the log directory exists and return the per-run log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = Path(log_file_name).stem or "electoral_roll_ocr"
    return log_dir / f"{safe_stem}_{timestamp}.log"
