"""Спектр амплитуд и фаз. Двойной клик по линии — включить/отключить гармонику."""
from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout

from ..analytics import spectrum_plot_data
from .harmonic_menu import show_harmonic_menu
from .plots import BLUE, GRAY, ORANGE, LazyPanel, make_plot


def _stems(k, amp, floor):
    xs = np.repeat(k.astype(float), 2)
    ys = np.column_stack([np.full(len(k), floor), amp]).ravel()
    return xs, ys


class SpectrumPanel(LazyPanel):
    delay_ms = 40

    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.amp_pw = make_plot("k", "|c_k|")
        self.ph_pw = make_plot("k", "arg c_k, рад")
        self.ph_pw.setXLink(self.amp_pw)
        self.ph_pw.setYRange(-3.3, 3.3)
        self.stem_on = self.amp_pw.plot([], [], pen=pg.mkPen(BLUE, width=2), connect="pairs")
        self.stem_edit = self.amp_pw.plot([], [], pen=pg.mkPen(ORANGE, width=2), connect="pairs")
        self.stem_off = self.amp_pw.plot([], [], pen=pg.mkPen(GRAY, width=2), connect="pairs")
        kw = dict(pen=None, symbol="o", symbolSize=5)
        self.ph_on = self.ph_pw.plot([], [], symbolBrush=BLUE, **kw)
        self.ph_edit = self.ph_pw.plot([], [], symbolBrush=ORANGE, **kw)
        self.ph_off = self.ph_pw.plot([], [], symbolBrush=GRAY, **kw)

        self.cb_log = QCheckBox("Логарифмическая амплитуда")
        self.cb_log.toggled.connect(self._on_log)
        self.sp_range = QSpinBox(minimum=1, maximum=4096, value=60)
        self.sp_range.valueChanged.connect(self._apply_range)
        top = QHBoxLayout()
        top.addWidget(self.cb_log)
        top.addWidget(QLabel("Показывать |k| ≤"))
        top.addWidget(self.sp_range)
        top.addWidget(QLabel("Синий — включена, оранжевый — изменена, серый — отключена. "
                             "Двойной клик по линии — вкл/выкл гармоники."))
        top.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.amp_pw, 3)
        lay.addWidget(self.ph_pw, 2)
        for pw in (self.amp_pw, self.ph_pw):
            pw.scene().sigMouseClicked.connect(lambda ev, pw=pw: self._on_click(ev, pw))
            pw.getPlotItem().setMenuEnabled(False)          # свои меню вместо стандартного pyqtgraph
            pw.setContextMenuPolicy(Qt.CustomContextMenu)
            pw.customContextMenuRequested.connect(lambda pos, pw=pw: self._on_context_menu(pw, pos))
        self._apply_range()

    def _apply_range(self, *_):
        m = self.sp_range.value()
        self.amp_pw.setXRange(-m - 1, m + 1, padding=0)

    def _on_log(self, on: bool):
        self.amp_pw.getPlotItem().setLogMode(y=on)
        self.refresh()

    def _on_context_menu(self, pw, pos):
        vb = pw.getPlotItem().vb
        k = int(round(vb.mapSceneToView(pw.mapToScene(pos)).x()))
        spec = self.hub.editor.spectrum
        if k in spec._index:
            show_harmonic_menu(self, self.hub.editor, [k], pw.mapToGlobal(pos))

    def _on_click(self, ev, pw):
        if not ev.double() or ev.button() != Qt.LeftButton:
            return
        vb = pw.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        k = int(round(vb.mapSceneToView(ev.scenePos()).x()))
        spec = self.hub.editor.spectrum
        try:
            self.hub.editor.set_enabled(k, not bool(spec.enabled[spec.index_of(k)]))
        except (KeyError, ValueError):
            pass

    def update_view(self, replaced: bool):
        d = spectrum_plot_data(self.hub.editor.spectrum)
        log = self.cb_log.isChecked()
        floor = max(float(d.amp.max()), 1e-300) * 1e-12 if log else 0.0
        amp = np.maximum(d.amp, floor) if log else d.amp
        groups = ((self.stem_on, self.ph_on, d.enabled & ~d.edited),
                  (self.stem_edit, self.ph_edit, d.enabled & d.edited),
                  (self.stem_off, self.ph_off, ~d.enabled))
        for stem, ph, m in groups:
            xs, ys = _stems(d.k[m], amp[m], floor)
            stem.setData(xs, ys, connect="pairs")
            ok = m & ~np.isnan(d.phase)
            ph.setData(d.k[ok].astype(float), d.phase[ok])
        if replaced:
            self.amp_pw.getPlotItem().enableAutoRange(axis="y")
