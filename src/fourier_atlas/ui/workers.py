"""Фоновое выполнение функций (загрузка файлов, позже — экспорт) без блокировки UI."""
from __future__ import annotations
import logging
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
