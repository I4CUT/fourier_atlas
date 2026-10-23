"""Главное окно: выбор страны/файла, подготовка контура, анимация, формула."""
from __future__ import annotations
import logging
import time
from dataclasses import replace
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QCheckBox, QDockWidget, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ..animation import AnimationController
from ..core import reconstruct_at, rmse
from ..data.catalog import Catalog
from ..data.samples import contour_to_spectrum
from ..geo import FILE_FILTER, GeoLoadError, PrepareOptions, load_file, prepare_region
from .canvas import EpicycleCanvas
from .formula_panel import FormulaPanel
from .workers import Worker, start

log = logging.getLogger(__name__)
PROJECTIONS = [("Авто", "auto"), ("Равнопромежуточная (cos φ)", "equirectangular"),
               ("Plate carrée (lon/lat)", "plate_carree")]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Atlas")
        self.resize(1150, 880)
        self.opts = PrepareOptions()
        self.catalog = Catalog.from_defaults()
        self._workers: set[Worker] = set()
        self.current = self.catalog.get(self.catalog.default_name())
        self.prepared, self.z, spec = self._prepare(self.current)

        self.ctrl = AnimationController(spec, max_epicycles=100, speed=0.08)
        self.canvas = EpicycleCanvas(self.ctrl)
        self.canvas.set_target(self.z)
        self._build_panel()
        self.formula = FormulaPanel(self.ctrl)
        dock = QDockWidget("Ряд Фурье", self)
        dock.setWidget(self.formula)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        self._fill_combo(select=self.catalog.default_name())
        self._sync_part_spin()
        self._update_info()
        if self.catalog.warning:
            self.status.setText(self.catalog.warning)
        self._last = time.perf_counter()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    # ---------- подготовка контура ----------
    def _prepare(self, region):
        prepared = prepare_region(region, self.opts)
        z, spec = contour_to_spectrum(prepared.xy)
        return prepared, z, spec

    def _apply(self, region, reset_part: bool = True) -> bool:
        opts = replace(self.opts, part=1) if reset_part else self.opts
        old = self.opts
        self.opts = opts
        try:
            prepared, z, spec = self._prepare(region)
        except (GeoLoadError, ValueError) as e:
            self.opts = old
            log.warning("Не удалось подготовить «%s»: %s", region.name, e)
            self._error(str(e))
            return False
        self.current, self.prepared, self.z = region, prepared, z
        self.ctrl.set_spectrum(spec)
        self.canvas.set_target(z)
        self._sync_part_spin()
        self._update_info()
        self.formula.refresh()
        self.status.setText("")
        return True

    # ---------- UI ----------
    def _build_panel(self):
        self.cmb = QComboBox()
        self.cmb.setEditable(True)
        self.cmb.setInsertPolicy(QComboBox.NoInsert)
        comp = self.cmb.completer()
        comp.setFilterMode(Qt.MatchContains)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        self.cmb.activated.connect(self._on_pick)
        btn_import = QPushButton("Импорт файла…")
        btn_import.clicked.connect(self._on_import)

        self.s_part = QSpinBox(minimum=1, maximum=1)
        self.s_part.valueChanged.connect(self._on_part)
        self.lbl_parts = QLabel("из 1")
        self.cmb_proj = QComboBox()
        for title, key in PROJECTIONS:
            self.cmb_proj.addItem(title, key)
        self.cmb_proj.currentIndexChanged.connect(self._on_proj)

        self.btn_play = QPushButton("▶ Пуск")
        self.btn_play.clicked.connect(self._on_toggle)
        btn_reset = QPushButton("⟲ Сброс")
        btn_reset.clicked.connect(self._on_reset)

        self.s_speed = QSlider(Qt.Horizontal, minimum=-200, maximum=200, value=8)
        self.s_speed.valueChanged.connect(lambda v: setattr(self.ctrl, "speed", v / 100))
        self.s_phase = QSlider(Qt.Horizontal, minimum=0, maximum=1000)
        self.s_phase.valueChanged.connect(self._on_phase)
        self.sp_vec = QSpinBox(minimum=0, maximum=1024, value=100)
        self.sp_vec.setToolTip("Число вращающихся эпициклов, 0–1024.\n0 — только центр контура.\n"
                               "Значения выше числа гармоник контура включают все гармоники.")
        self.sp_vec.valueChanged.connect(self._on_vectors)
        self.s_trail = QSlider(Qt.Horizontal, minimum=1, maximum=100, value=100)
        self.s_trail.valueChanged.connect(lambda v: setattr(self.ctrl, "trail_fraction", v / 100))

        def check(text, attr):
            c = QCheckBox(text)
            c.setChecked(True)
            c.toggled.connect(lambda s: (setattr(self.canvas, attr, s), self.canvas.update()))
            return c

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #c9a227;")

        part_row = QWidget()
        pr = QHBoxLayout(part_row)
        pr.setContentsMargins(0, 0, 0, 0)
        pr.addWidget(QLabel("Часть:"))
        pr.addWidget(self.s_part)
        pr.addWidget(self.lbl_parts)
        pr.addStretch(1)

        box = QGroupBox("Контур и анимация")
        lay = QVBoxLayout(box)
        for w in (QLabel("Страна / контур (поиск по вводу)"), self.cmb, btn_import, part_row,
                  QLabel("Проекция"), self.cmb_proj, self.btn_play, btn_reset,
                  QLabel("Скорость (периодов/с ×100, <0 — назад)"), self.s_speed,
                  QLabel("Фаза"), self.s_phase, QLabel("Число эпициклов (0–1024)"), self.sp_vec,
                  QLabel("Длина следа"), self.s_trail,
                  check("Окружности", "show_circles"), check("Векторы", "show_vectors"),
                  check("Исходный контур", "show_target"), self.info, self.status):
            lay.addWidget(w)
        lay.addStretch(1)
        box.setFixedWidth(300)
        root = QWidget()
        h = QHBoxLayout(root)
        h.addWidget(self.canvas, 1)
        h.addWidget(box)
        self.setCentralWidget(root)

    def _fill_combo(self, select: str | None = None):
        self.cmb.blockSignals(True)
        self.cmb.clear()
        self.cmb.addItems(self.catalog.names())
        if select is not None:
            self.cmb.setCurrentIndex(max(0, self.cmb.findText(select)))
        self.cmb.blockSignals(False)

    def _sync_part_spin(self):
        self.s_part.blockSignals(True)
        self.s_part.setMaximum(max(1, self.prepared.n_parts))
        self.s_part.setValue(self.prepared.part_rank)
        self.s_part.blockSignals(False)
        self.lbl_parts.setText(f"из {self.prepared.n_parts}")

    # ---------- обработчики ----------
    def _on_pick(self, index: int):
        name = self.cmb.itemText(index)
        log.info("Выбран контур: %s", name)
        self._apply(self.catalog.get(name))

    def _on_part(self, v: int):
        self.opts = replace(self.opts, part=v)
        if not self._apply(self.current, reset_part=False):
            self._sync_part_spin()

    def _on_proj(self, _):
        self.opts = replace(self.opts, projection=self.cmb_proj.currentData())
        self._apply(self.current, reset_part=False)

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Импорт геоданных", "", FILE_FILTER)
        if not path:
            return
        self.status.setText(f"Загрузка: {path} …")
        w = Worker(load_file, path)
        self._workers.add(w)
        w.signals.done.connect(lambda regs, w=w: self._import_done(regs, w))
        w.signals.failed.connect(lambda msg, w=w: self._import_failed(msg, w))
        start(w)

    def _import_done(self, regions, worker):
        self._workers.discard(worker)
        keys = [self.catalog.add(r) for r in regions]
        log.info("Импортировано объектов: %d", len(keys))
        self._fill_combo(select=keys[0])
        self.status.setText(f"Импортировано объектов: {len(keys)}")
        self._apply(self.catalog.get(keys[0]))

    def _import_failed(self, msg, worker):
        self._workers.discard(worker)
        self._error(msg)

    def _error(self, msg: str):
        self.status.setText(msg)
        QMessageBox.warning(self, "Ошибка", msg)

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
        self.ctrl.max_epicycles = v
        self._update_info()
        self.formula.refresh()
        self.canvas.update()

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
        p = self.prepared
        rotating = int(np.count_nonzero(c.freqs != 0))
        self.info.setText(f"Эпициклов: {rotating}\nRMSE: {err:.5f}\n"
                          f"Часть {p.part_rank} из {p.n_parts}\n"
                          f"Точек исходного контура: {p.raw_points}\nПроекция: {p.projection}")
