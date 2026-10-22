"""Главное окно (этап 2): холст + управление анимацией."""
from __future__ import annotations
import logging
import time
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QDockWidget, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                               QSlider, QLabel, QComboBox, QCheckBox, QGroupBox)

from ..animation import AnimationController
from ..core import reconstruct_at, rmse
from ..data.samples import SAMPLES, make_sample
from .canvas import EpicycleCanvas
from .formula_panel import FormulaPanel

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Atlas")
        self.resize(1100, 860)
        name = next(iter(SAMPLES))
        self.z, spec = make_sample(name)
        self.ctrl = AnimationController(spec, max_vectors=64, speed=0.08)
        self.canvas = EpicycleCanvas(self.ctrl)
        self.canvas.set_target(self.z)
        self._build_panel()
        self.formula = FormulaPanel(self.ctrl)
        dock = QDockWidget("Ряд Фурье", self)
        dock.setWidget(self.formula)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self._last = time.perf_counter()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)
        self._update_info()

    # ---------- UI ----------
    def _build_panel(self):
        self.shape = QComboBox(); self.shape.addItems(list(SAMPLES))
        self.shape.currentTextChanged.connect(self._on_shape)
        self.btn_play = QPushButton("▶ Пуск"); self.btn_play.clicked.connect(self._on_toggle)
        btn_reset = QPushButton("⟲ Сброс"); btn_reset.clicked.connect(self._on_reset)

        self.s_speed = QSlider(Qt.Horizontal, minimum=-200, maximum=200, value=8)
        self.s_speed.valueChanged.connect(lambda v: setattr(self.ctrl, "speed", v / 100))
        self.s_phase = QSlider(Qt.Horizontal, minimum=0, maximum=1000)
        self.s_phase.valueChanged.connect(self._on_phase)
        self.s_vec = QSlider(Qt.Horizontal, minimum=1, maximum=len(self.ctrl.spectrum.freqs), value=64)
        self.s_vec.valueChanged.connect(self._on_vectors)
        self.s_trail = QSlider(Qt.Horizontal, minimum=1, maximum=100, value=100)
        self.s_trail.valueChanged.connect(lambda v: setattr(self.ctrl, "trail_fraction", v / 100))

        def check(text, attr, on=True):
            c = QCheckBox(text); c.setChecked(on)
            c.toggled.connect(lambda s: (setattr(self.canvas, attr, s), self.canvas.update()))
            return c

        self.info = QLabel(); self.info.setWordWrap(True)
        box = QGroupBox("Анимация"); lay = QVBoxLayout(box)
        for w in (QLabel("Контур"), self.shape, self.btn_play, btn_reset,
                  QLabel("Скорость (периодов/с ×100, <0 — назад)"), self.s_speed,
                  QLabel("Фаза"), self.s_phase,
                  QLabel("Число векторов"), self.s_vec,
                  QLabel("Длина следа"), self.s_trail,
                  check("Окружности", "show_circles"), check("Векторы", "show_vectors"),
                  check("Исходный контур", "show_target"), self.info):
            lay.addWidget(w)
        lay.addStretch(1)
        box.setFixedWidth(280)
        root = QWidget(); h = QHBoxLayout(root)
        h.addWidget(self.canvas, 1); h.addWidget(box)
        self.setCentralWidget(root)

    # ---------- обработчики ----------
    def _on_toggle(self):
        self.btn_play.setText("⏸ Пауза" if self.ctrl.toggle() else "▶ Пуск")
        self._last = time.perf_counter()

    def _on_reset(self):
        self.ctrl.reset()
        self.btn_play.setText("▶ Пуск")
        self._sync_phase_slider()
        self.canvas.update()

    def _on_phase(self, v):
        self.ctrl.seek(v / 1000)
        self.canvas.update()

    def _on_vectors(self, v):
        self.ctrl.max_vectors = v
        self._update_info()
        self.formula.refresh()
        self.canvas.update()

    def _on_shape(self, name):
        log.info("Контур: %s", name)
        self.z, spec = make_sample(name)
        self.ctrl.set_spectrum(spec)
        self.canvas.set_target(self.z)
        self._update_info()
        self.formula.refresh()

    def _tick(self):
        now = time.perf_counter()
        dt, self._last = min(now - self._last, 0.1), now
        if self.ctrl.advance(dt):
            self._sync_phase_slider()
            self.canvas.update()

    def _sync_phase_slider(self):
        self.s_phase.blockSignals(True)
        self.s_phase.setValue(int(self.ctrl.phase * 1000))
        self.s_phase.blockSignals(False)

    def _update_info(self):
        c = self.ctrl.chain
        t = np.arange(len(self.z)) / len(self.z)
        err = rmse(self.z, reconstruct_at(c.freqs, c.coeffs, t))
        self.info.setText(f"Векторов: {len(c.freqs)}\nRMSE: {err:.5f}")
