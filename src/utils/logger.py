import inspect
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_FILE = os.getenv("DEFAULT_LOG_FILE", "logs/app.log")
DEFAULT_LOG_SIZE = os.getenv("DEFAULT_LOG_SIZE", "10MB")
DEFAULT_LOG_BACKUP_COUNT = int(os.getenv("DEFAULT_LOG_BACKUP_COUNT", "10"))
DEFAULT_LOG_FORMAT = os.getenv("DEFAULT_LOG_FORMAT", "%(level_icon)s [%(levelname)s] %(message)s")
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SUCCESS_LEVEL = 25
_FILE_HANDLERS: dict[str, RotatingFileHandler] = {}

logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def _success(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = _success  # type: ignore[attr-defined]


class IconFormatter(logging.Formatter):
    """Add a status icon before the logging level."""

    LEVEL_ICONS = {
        "DEBUG": "🐛",
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.level_icon = self.LEVEL_ICONS.get(record.levelname, "")
        return super().format(record)


def _resolve_log_file(log_file: str | Path) -> Path:
    log_path = Path(log_file)
    if log_path.is_absolute():
        return log_path
    return PROJECT_ROOT / log_path


def _parse_size(size: int | str) -> int:
    if isinstance(size, int):
        return size

    normalized = size.strip().upper()
    units = {
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
    }

    for unit, multiplier in units.items():
        if normalized.endswith(unit):
            value = normalized[: -len(unit)].strip()
            return int(float(value) * multiplier)

    return int(normalized)


def _caller_logger_name(stack_depth: int = 2) -> str:
    frame_info = inspect.stack()[stack_depth]
    caller_path = Path(frame_info.filename).resolve()

    try:
        return caller_path.relative_to(PROJECT_ROOT).with_suffix("").as_posix().replace("/", ".")
    except ValueError:
        return caller_path.stem


def _get_file_handler(
    *,
    log_file: str | Path,
    max_bytes: int | str,
    backup_count: int,
    level: int,
    handler_key_prefix: str = "",
) -> RotatingFileHandler:
    log_path = _resolve_log_file(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler_key = f"{handler_key_prefix}{log_path.resolve()}"
    if handler_key not in _FILE_HANDLERS:
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=_parse_size(max_bytes),
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(IconFormatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
        file_handler._handler_key = handler_key
        _FILE_HANDLERS[handler_key] = file_handler

    file_handler = _FILE_HANDLERS[handler_key]
    file_handler.setLevel(level)
    return file_handler


def get_logger(
    name: str | None = None,
    level: int = logging.INFO,
    log_file: str | Path = DEFAULT_LOG_FILE,
    max_bytes: int | str = DEFAULT_LOG_SIZE,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> logging.Logger:
    """Create a console and rotating-file logger."""
    logger = logging.getLogger(name or _caller_logger_name())
    logger.setLevel(level)
    logger.propagate = False

    formatter = IconFormatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if not any(getattr(handler, "_handler_key", None) == "console" for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        console_handler._handler_key = "console"
        logger.addHandler(console_handler)

    file_handler = _get_file_handler(
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        level=level,
    )

    if not any(getattr(handler, "_handler_key", None) == file_handler._handler_key for handler in logger.handlers):
        logger.addHandler(file_handler)

    return logger


def get_file_logger(
    name: str | None = None,
    level: int = logging.INFO,
    log_file: str | Path = DEFAULT_LOG_FILE,
    max_bytes: int | str = DEFAULT_LOG_SIZE,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> logging.Logger:
    """Create a rotating-file logger without console output."""
    logger = logging.getLogger(f"{name or _caller_logger_name()}.__file_only__")
    logger.setLevel(level)
    logger.propagate = False

    file_handler = _get_file_handler(
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        level=level,
        handler_key_prefix="file-only:",
    )

    if not any(getattr(handler, "_handler_key", None) == file_handler._handler_key for handler in logger.handlers):
        logger.addHandler(file_handler)

    return logger


def log_info(message: str, *args, **kwargs) -> None:
    get_logger(_caller_logger_name()).info(message, *args, **kwargs)


def log_success(message: str, *args, **kwargs) -> None:
    get_logger(_caller_logger_name()).success(message, *args, **kwargs)


def log_warning(message: str, *args, **kwargs) -> None:
    get_logger(_caller_logger_name()).warning(message, *args, **kwargs)


def log_error(message: str, *args, **kwargs) -> None:
    get_logger(_caller_logger_name()).error(message, *args, **kwargs)


def file_info(message: str, *args, **kwargs) -> None:
    get_file_logger(_caller_logger_name()).info(message, *args, **kwargs)


def file_success(message: str, *args, **kwargs) -> None:
    get_file_logger(_caller_logger_name()).success(message, *args, **kwargs)


def file_warning(message: str, *args, **kwargs) -> None:
    get_file_logger(_caller_logger_name()).warning(message, *args, **kwargs)


def file_error(message: str, *args, **kwargs) -> None:
    get_file_logger(_caller_logger_name()).error(message, *args, **kwargs)


logger = get_logger("app")
