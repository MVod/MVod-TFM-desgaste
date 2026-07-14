# app/logger.py
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
LOG_FILE  = LOGS_DIR / "lora_sam.log"

_FMT      = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
    r" \| (\w+)\s*"
    r" \| ([^|]+)"
    r" \| (.+)$"
)

LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "#64748b",
    "INFO":     "#94a3b8",
    "WARNING":  "#f59e0b",
    "ERROR":    "#ef4444",
    "CRITICAL": "#dc2626",
}


def _setup() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("lora_sam")
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)


_setup()


def get_logger(name: str) -> logging.Logger:
    """Returns a child logger under the lora_sam namespace."""
    return logging.getLogger(f"lora_sam.{name}")


def get_recent_logs(n: int = 200, level_filter: str = "ALL") -> list[dict]:
    """
    Returns last n matching log entries as dicts: {ts, level, module, message, color}.
    Reads from the end of the file for efficiency.
    """
    if not LOG_FILE.exists():
        return []
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    parsed: list[dict] = []
    for line in reversed(text.splitlines()):
        m = _LOG_RE.match(line)
        if not m:
            continue
        ts, level, module, message = m.groups()
        level  = level.strip()
        module = module.strip().replace("lora_sam.", "")
        if level_filter != "ALL" and level != level_filter:
            continue
        parsed.append({
            "ts":      ts,
            "level":   level,
            "module":  module,
            "message": message,
            "color":   LEVEL_COLORS.get(level, "#94a3b8"),
        })
        if len(parsed) >= n:
            break

    parsed.reverse()
    return parsed


def get_log_path() -> Path:
    return LOG_FILE
