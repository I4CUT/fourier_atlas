"""Фоновое выполнение функций (загрузка файлов, позже — экспорт) без блокировки UI."""
from __future__ import annotations
import logging
import threading
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

log = logging.getLogger(__name__)


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()
        self.setAutoDelete(False)          # владелец (окно) держит ссылку, пока не придёт сигнал

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:             # noqa: BLE001 — любые ошибки должны дойти до UI
            log.exception("Ошибка в фоновой задаче")
            self.signals.failed.emit(str(e) or e.__class__.__name__)
        else:
            self.signals.done.emit(result)


def start(worker: Worker) -> None:
    QThreadPool.globalInstance().start(worker)


class _ProgressSignals(QObject):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class ProgressWorker(QRunnable):
    """Фоновая задача с прогрессом и отменой. fn вызывается как fn(*args, progress=..., cancel=...)."""

    def __init__(self, fn, *args, cancel_exc=(), **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs, self.cancel_exc = fn, args, kwargs, tuple(cancel_exc)
        self.signals = _ProgressSignals()
        self._cancel = threading.Event()
        self.setAutoDelete(False)

    def cancel(self) -> None:
        self._cancel.set()

    def run(self):
        try:
            result = self.fn(*self.args, progress=self.signals.progress.emit,
                             cancel=self._cancel.is_set, **self.kwargs)
        except self.cancel_exc:
            self.signals.cancelled.emit()
        except Exception as e:                   # noqa: BLE001
            log.exception("Ошибка в фоновой задаче")
            self.signals.failed.emit(str(e) or e.__class__.__name__)
        else:
            self.signals.done.emit(result)
