"""Журналирование: файл с ротацией + консоль, перехват необработанных исключений."""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading

from .settings import config_dir

_TAG = "_fourier_atlas_handler"
FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def log_dir() -> Path:
    return config_dir() / "logs"


def setup_logging(directory=None, level=logging.INFO, console: bool = True) -> Path | None:
    """Идемпотентна. Возвращает путь к файлу журнала (None, если каталог недоступен)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _TAG, False):
            root.removeHandler(h)
            h.close()
    root.setLevel(level)
    fmt = logging.Formatter(FMT)
    path = None
    try:
        d = Path(directory) if directory else log_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / "fourier_atlas.log"
        fh = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        setattr(fh, _TAG, True)
        root.addHandler(fh)
    except OSError as e:
        path = None
        print(f"Журнал в файл недоступен: {e}", file=sys.stderr)
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        setattr(ch, _TAG, True)
        root.addHandler(ch)
    return path


def install_excepthook(on_error=None) -> None:
    """Все необработанные исключения (в т.ч. в потоках) попадают в журнал;
    on_error(текст) — необязательный показ пользователю."""
    log = logging.getLogger("fourier_atlas")

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Необработанное исключение", exc_info=(exc_type, exc, tb))
        if on_error:
            try:
                on_error(f"{exc_type.__name__}: {exc}")
            except Exception:                                  # noqa: BLE001
                pass

    sys.excepthook = hook
    threading.excepthook = lambda a: log.critical("Исключение в потоке %s", a.thread,
                                                   exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
