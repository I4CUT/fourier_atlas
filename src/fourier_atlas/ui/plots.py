"""Общие настройки PyQtGraph и базовый класс «ленивой» панели."""
from __future__ import annotations
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget

pg.setConfigOptions(antialias=True, background=(18, 20, 28), foreground=(190, 195, 210))

BLUE, ORANGE, GRAY, GREEN = (110, 150, 230), (255, 190, 60), (110, 115, 130), (90, 200, 150)


def make_plot(xlabel: str = "", ylabel: str = "") -> pg.PlotWidget:
    pw = pg.PlotWidget()
    pi = pw.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.2)
    pi.setLabel("bottom", xlabel)
    pi.setLabel("left", ylabel)
    return pw


class LazyPanel(QWidget):
    """Панель обновляется только когда видима; частые события склеиваются таймером."""
    delay_ms = 60

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.hub = hub
        self._dirty = True
        self._replaced = True
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._run)
        hub.changed.connect(self.refresh)
        hub.replaced.connect(self._on_replaced)

    def refresh(self, *_):
        self._dirty = True
        if self.isVisible():
            self._timer.start(self.delay_ms)

    def _on_replaced(self):
        self._replaced = True
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self._timer.start(0)

    def _run(self):
        if not self.isVisible():
            return
        replaced = self._replaced
        self._dirty = self._replaced = False
        self.update_view(replaced)

    def update_view(self, replaced: bool):          # переопределяется
        raise NotImplementedError
