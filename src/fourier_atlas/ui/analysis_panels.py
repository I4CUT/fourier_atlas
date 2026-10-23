"""График сходимости и сравнение исходного/восстановленного контуров."""
from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout

from ..analytics import compare_reconstruction, contour_diagonal, epicycles_for_relative_error
from ..core import convergence_by_amplitude, convergence_curve, reconstruct
from .plots import BLUE, GRAY, GREEN, ORANGE, LazyPanel, make_plot

FLOOR_PCT = 1e-9


class ConvergencePanel(LazyPanel):
    delay_ms = 120

    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.pw = make_plot("Число вращающихся эпициклов N", "Относительная ошибка (RMSE), % диагонали")
        self.pw.addLegend(offset=(10, 10))
        self.curve_amp = self.pw.plot([], [], pen=pg.mkPen(BLUE, width=2), name="по убыванию |c| (как в анимации)")
        self.curve_band = self.pw.plot([], [], pen=pg.mkPen(GREEN, width=2, style=Qt.DashLine), name="полоса |k| ≤ M (N = 2M)")
        self.marker = self.pw.plot([], [], pen=None, symbol="o", symbolSize=11, symbolBrush=ORANGE,
                                   name="текущее состояние")
        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(ORANGE, width=1, style=Qt.DashLine))
        self.pw.addItem(self.vline)
        self.cb_log = QCheckBox("Логарифмическая шкала ошибки")
        self.cb_log.setChecked(True)
        self.cb_log.toggled.connect(self._on_log)
        self.info = QLabel()
        self.info.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(self.cb_log)
        top.addWidget(self.info, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.pw, 1)
        self.pw.getPlotItem().setLogMode(y=True)

    def _on_log(self, on):
        self.pw.getPlotItem().setLogMode(y=on)
        self.pw.getPlotItem().enableAutoRange()

    def update_view(self, replaced: bool):
        ed = self.hub.editor
        spec, z = ed.spectrum, ed.z
        diag = contour_diagonal(z)
        if replaced:                              # кривые зависят только от исходного спектра
            ns, e = convergence_by_amplitude(spec)
            ms, eb = convergence_curve(spec)
            self.curve_amp.setData(ns.astype(float), np.maximum(e / diag * 100, FLOOR_PCT))
            self.curve_band.setData(2.0 * ms, np.maximum(eb / diag * 100, FLOOR_PCT))
            parts = []
            for pct in (5, 1, 0.1, 0.01):
                n = epicycles_for_relative_error(spec, z, pct)
                parts.append(f"≤{pct:g}%: {n if n is not None else '—'}")
            self._targets = "Нужно эпициклов для ошибки " + ";  ".join(parts)
            self.pw.getPlotItem().enableAutoRange()
        rotating = int(np.count_nonzero(ed.ctrl.chain.freqs != 0))
        now = 100.0 * float(np.sqrt(np.mean(np.abs(z - reconstruct(spec, len(z))) ** 2))) / diag
        self.marker.setData([float(rotating)], [max(now, FLOOR_PCT)])
        self.vline.setValue(rotating)
        self.info.setText(f"{self._targets}.   Сейчас: {rotating} эпициклов, ошибка {now:.4g}%"
                          " (кривые — по исходному спектру, маркер — по текущему состоянию с правками)")


class ComparePanel(LazyPanel):
    delay_ms = 150

    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.pw = make_plot("x", "y")
        self.pw.setAspectLocked(True)
        self.pw.addLegend(offset=(10, 10))
        self.errs = self.pw.plot([], [], pen=pg.mkPen((200, 80, 80, 140), width=1), connect="pairs")
        self.orig = self.pw.plot([], [], pen=pg.mkPen(GRAY, width=2), name="исходный")
        self.rec = self.pw.plot([], [], pen=pg.mkPen(ORANGE, width=2), name="восстановленный")
        self.cb_err = QCheckBox("Показать отклонения (красные отрезки)")
        self.cb_err.toggled.connect(self.refresh)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(self.cb_err)
        top.addWidget(self.info, 1)
        lay.addLayout(top)
        lay.addWidget(self.pw, 1)

    def update_view(self, replaced: bool):
        ed = self.hub.editor
        z = ed.z
        recon, e = compare_reconstruction(z, ed.spectrum)
        zc, rc = np.append(z, z[0]), np.append(recon, recon[0])
        self.orig.setData(zc.real, zc.imag)
        self.rec.setData(rc.real, rc.imag)
        if self.cb_err.isChecked():
            idx = np.arange(0, len(z), max(1, len(z) // 128))
            pts = np.column_stack([z[idx], recon[idx]]).ravel()
            self.errs.setData(pts.real, pts.imag, connect="pairs")
        else:
            self.errs.setData([], [])
        if replaced:
            self.pw.getPlotItem().enableAutoRange()
        diag = contour_diagonal(z)
        self.info.setText(f"RMSE {e.rmse:.5g} ({e.relative_rmse * 100:.4g}% диагонали);  максимум {e.max_error:.5g};  "
                          f"Хаусдорф {e.hausdorff:.5g} ({e.hausdorff / diag * 100:.4g}%).  "
                          "Единицы — нормированный контур (max|z| = 1).")
