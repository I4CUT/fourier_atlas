"""Главное окно: выбор страны/файла, анимация, анализ, проекты, экспорт, настройки."""
from __future__ import annotations
import logging
import re
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from PySide6.QtCore import QByteArray, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDockWidget, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSlider,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..analytics import SpectrumEditor
from ..animation import AnimationController
from ..core import from_complex, reconstruct_at, rmse
from ..data.catalog import Catalog
from ..data.samples import contour_to_spectrum
from ..export import (ExportCancelled, ExportError, ExportJob, ProjectError, Style, build_state,
                      export_coefficients_csv, export_contour_csv, load_project, run_export, save_project,
                      spectrum_from_state, validate_job)
from ..geo import FILE_FILTER, GeoLoadError, PrepareOptions, load_file, prepare_region
from ..geo.model import Region
from ..geo.prepare import PreparedContour
from ..logging_setup import log_dir
from ..settings import Settings
from .analysis_panels import ComparePanel, ConvergencePanel
from .audio_panel import AudioPanel
from .canvas import EpicycleCanvas
from .export_dialog import ExportDialog
from .formula_panel import FormulaPanel
from .harmonic_menu import show_harmonic_menu
from .hub import SpectrumHub
from .spectrum_panel import SpectrumPanel
from .table_panel import TablePanel
from .workers import ProgressWorker, Worker, start

log = logging.getLogger(__name__)
PROJECTIONS = [("Авто", "auto"), ("Равнопромежуточная (cos φ)", "equirectangular"),
               ("Plate carrée (lon/lat)", "plate_carree")]
PROJECT_FILTER = "Проект Fourier Atlas (*.fatlas.json *.json);;Все файлы (*)"


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_.") or "contour"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Atlas")
        self.resize(1300, 960)
        self.settings = Settings.load()
        st = self.settings
        self.opts = PrepareOptions(projection=st.get("projection"))
        self.catalog = Catalog.from_defaults()
        self._workers: set[Worker] = set()
        self._export_worker: ProgressWorker | None = None
        start_name = st.get("country")
        if start_name not in self.catalog.names():
            start_name = self.catalog.default_name()
        self.current = self.catalog.get(start_name)
        self.prepared, self.z, spec = self._prepare(self.current)

        self.ctrl = AnimationController(spec, max_epicycles=st.get("max_epicycles"), speed=st.get("speed"),
                                        trail_fraction=st.get("trail_fraction"))
        self.canvas = EpicycleCanvas(self.ctrl)
        self.canvas.show_circles = st.get("show_circles")
        self.canvas.show_vectors = st.get("show_vectors")
        self.canvas.show_target = st.get("show_target")
        self.canvas.set_target(self.z)
        self._build_panel()
        self.editor = SpectrumEditor(self.ctrl, self.z)
        self.hub = SpectrumHub(self.editor, self)
        self.formula = FormulaPanel(self.ctrl)
        tabs = QTabWidget()
        tabs.addTab(self.formula, "Ряд Фурье")
        tabs.addTab(SpectrumPanel(self.hub), "Спектр")
        tabs.addTab(TablePanel(self.hub), "Коэффициенты")
        tabs.addTab(ConvergencePanel(self.hub), "Сходимость")
        tabs.addTab(ComparePanel(self.hub), "Сравнение")
        self.audio_panel = AudioPanel(self.hub, self.ctrl, self.settings)
        self.audio_panel.save_requested.connect(lambda kind, mode: self._export_media(kind, mode))
        tabs.addTab(self.audio_panel, "Звук")
        self.dock = QDockWidget("Анализ", self)
        self.dock.setObjectName("analysis_dock")
        self.dock.setWidget(tabs)
        self.dock.setMinimumHeight(330)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.canvas.harmonic_menu_requested.connect(self._canvas_menu)
        self.hub.changed.connect(self._after_change)
        self.hub.replaced.connect(self._after_change)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.editor.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.editor.redo)
        self._build_menus()

        self._fill_combo(select=start_name)
        self._sync_part_spin()
        self._update_info()
        self._restore_window()
        if self.catalog.warning:
            self.status.setText(self.catalog.warning)
        self._last = time.perf_counter()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)
        log.info("Окно создано; контур: %s", self.current.name)

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
        self.canvas.set_target(z)
        self.editor.replace(z, spec)             # сигнал replaced: панели, формула, подписи
        self._sync_part_spin()
        self.status.setText("")
        return True

    # ---------- UI ----------
    def _build_panel(self):
        st = self.settings
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
        self.cmb_proj.setCurrentIndex(max(0, self.cmb_proj.findData(self.opts.projection)))
        self.cmb_proj.currentIndexChanged.connect(self._on_proj)

        self.btn_play = QPushButton("▶ Пуск")
        self.btn_play.clicked.connect(self._on_toggle)
        btn_reset = QPushButton("⟲ Сброс")
        btn_reset.clicked.connect(self._on_reset)

        self.s_speed = QSlider(Qt.Horizontal, minimum=-200, maximum=200,
                               value=int(round(st.get("speed") * 100)))
        self.s_speed.valueChanged.connect(lambda v: setattr(self.ctrl, "speed", v / 100))
        self.s_phase = QSlider(Qt.Horizontal, minimum=0, maximum=1000)
        self.s_phase.valueChanged.connect(self._on_phase)
        self.sp_vec = QSpinBox(minimum=0, maximum=1024, value=st.get("max_epicycles"))
        self.sp_vec.setToolTip("Число вращающихся эпициклов, 0–1024.\n0 — только центр контура.\n"
                               "Значения выше числа гармоник контура включают все гармоники.")
        self.sp_vec.valueChanged.connect(self._on_vectors)
        self.s_trail = QSlider(Qt.Horizontal, minimum=1, maximum=100,
                               value=max(1, int(round(st.get("trail_fraction") * 100))))
        self.s_trail.valueChanged.connect(lambda v: setattr(self.ctrl, "trail_fraction", v / 100))

        def check(text, attr):
            c = QCheckBox(text)
            c.setChecked(getattr(self.canvas, attr))
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

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(lambda _=False: slot())
        if shortcut is not None:
            a.setShortcut(QKeySequence(shortcut))
        menu.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&Файл")
        self._act(m, "Открыть проект…", self._open_project, QKeySequence.Open)
        self._act(m, "Сохранить проект…", self._save_project, QKeySequence.Save)
        m.addSeparator()
        self._act(m, "Импорт геоданных…", self._on_import, "Ctrl+I")
        m.addSeparator()
        self._act(m, "Выход", self.close, QKeySequence.Quit)
        m = mb.addMenu("&Правка")
        self._act(m, "Отменить правку коэффициента  (Ctrl+Z)", self.editor.undo)
        self._act(m, "Повторить  (Ctrl+Y)", self.editor.redo)
        m = mb.addMenu("&Экспорт")
        for kind, title in (("png", "Изображение PNG…"), ("svg", "Векторное изображение SVG…"),
                            ("gif", "Анимация GIF…"), ("mp4", "Видео MP4…")):
            self._act(m, title, lambda k=kind: self._export_media(k))
        m.addSeparator()
        self._act(m, "Звук WAV: тембр контура…", lambda: self._export_media("wav", "timbre"))
        self._act(m, "Звук WAV: наращивание гармоник…", lambda: self._export_media("wav", "buildup"))
        self._act(m, "Звук WAV: мелодия…", lambda: self._export_media("wav", "melody"))
        self._act(m, "Мелодия MIDI…", lambda: self._export_media("mid", "melody"))
        m.addSeparator()
        self._act(m, "Коэффициенты (CSV)…", self._export_coeffs_csv)
        self._act(m, "Контуры: исходный и восстановленный (CSV)…", self._export_contour_csv)
        m = mb.addMenu("&Справка")
        self._act(m, "Где находится журнал…", self._show_log_location)

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

    # ---------- обработчики панели ----------
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
        d = self.settings.get("last_dir") or ""
        path, _ = QFileDialog.getOpenFileName(self, "Импорт геоданных", d, FILE_FILTER)
        if not path:
            return
        self.settings.set("last_dir", str(Path(path).parent))
        self.status.setText(f"Загрузка: {path} …")
        log.info("Импорт файла: %s", path)
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
        log.error("Сообщение пользователю: %s", msg)
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
        self.editor.set_max_epicycles(v)

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

    def _canvas_menu(self, k, pos):
        was_playing = self.ctrl.playing
        self.ctrl.pause()                              # пока меню открыто, кадр не двигается
        show_harmonic_menu(self, self.editor, [k], pos)
        if was_playing:
            self.ctrl.play()
            self._last = time.perf_counter()

    def _after_change(self):
        self.formula.refresh()
        self._update_info()
        self.canvas.update()

    def _update_info(self):
        c = self.ctrl.chain
        t = np.arange(len(self.z)) / len(self.z)
        err = rmse(self.z, reconstruct_at(c.freqs, c.coeffs, t))
        p, s = self.prepared, self.ctrl.spectrum
        rotating = int(np.count_nonzero(c.freqs != 0))
        self.info.setText(f"Эпициклов: {rotating}\nRMSE: {err:.5f}\n"
                          f"Часть {p.part_rank} из {p.n_parts}\n"
                          f"Точек исходного контура: {p.raw_points}\nПроекция: {p.projection}\n"
                          f"Изменено коэффициентов: {int(s.edited_mask().sum())}\n"
                          f"Отключено гармоник: {int((~s.enabled).sum())}")

    # ---------- файловые диалоги ----------
    def _pick_save(self, title, filt, default_name, dir_key, suffix):
        d = self.settings.get(dir_key) or self.settings.get("last_dir") or ""
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(d) / default_name) if d else default_name, filt)
        if not path:
            return None
        if not path.lower().endswith(suffix):
            path += suffix
        self.settings.set(dir_key, str(Path(path).parent))
        return path

    # ---------- проекты ----------
    def _save_project(self):
        path = self._pick_save("Сохранить проект", "Проект Fourier Atlas (*.fatlas.json)",
                               f"{_safe_name(self.current.name)}.fatlas.json", "project_dir", ".json")
        if not path:
            return
        c = self.ctrl
        state = build_state(self.current.name, self.z, c.spectrum, c.max_epicycles, c.speed,
                            c.trail_fraction, c.phase,
                            source={"region": self.current.name, "group": self.current.group,
                                    "part": self.prepared.part_rank, "projection": self.prepared.projection})
        try:
            save_project(path, state)
        except ProjectError as e:
            self._error(str(e))
            return
        log.info("Проект сохранён: %s", path)
        self.status.setText(f"Проект сохранён: {path}")

    def _open_project(self):
        d = self.settings.get("project_dir") or self.settings.get("last_dir") or ""
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", d, PROJECT_FILTER)
        if not path:
            return
        self.settings.set("project_dir", str(Path(path).parent))
        try:
            state = load_project(path)
            spec = spectrum_from_state(state)
        except ProjectError as e:
            self._error(str(e))
            return
        xy = from_complex(state.z)
        region = Region(state.name, [xy], kind="planar", group="Проект", source=path)
        key = self.catalog.add(region)
        self.current = region
        self.prepared = PreparedContour(xy, 1, 1, "none", len(state.z))
        self.z = state.z
        self.canvas.set_target(self.z)
        self.editor.replace(self.z, spec)
        # параметры анимации из проекта (виджеты — без повторных сигналов, значения — точные)
        self.sp_vec.blockSignals(True)
        self.sp_vec.setValue(1024 if state.max_epicycles is None else state.max_epicycles)
        self.sp_vec.blockSignals(False)
        self.editor.set_max_epicycles(state.max_epicycles)
        self.s_speed.blockSignals(True)
        self.s_speed.setValue(max(-200, min(200, int(round(state.speed * 100)))))
        self.s_speed.blockSignals(False)
        self.ctrl.speed = state.speed
        self.s_trail.blockSignals(True)
        self.s_trail.setValue(max(1, int(round(state.trail_fraction * 100))))
        self.s_trail.blockSignals(False)
        self.ctrl.trail_fraction = state.trail_fraction
        self.ctrl.seek(state.phase)
        self._sync_phase_slider()
        self._fill_combo(select=key)
        self._sync_part_spin()
        log.info("Проект открыт: %s", path)
        self.status.setText(f"Проект открыт: {Path(path).name}")

    # ---------- экспорт ----------
    def _export_media(self, kind: str, audio_mode: str | None = None):
        if self._export_worker is not None:
            self._error("Экспорт уже выполняется. Дождитесь завершения или отмените его.")
            return
        ex = self.settings.get("export")
        flags = {"circles": self.canvas.show_circles, "vectors": self.canvas.show_vectors,
                 "target": self.canvas.show_target}
        dlg = ExportDialog(kind, ex, flags, {**self.settings.get("audio"), **self.audio_panel.values()},
                           audio_mode, self)
        if dlg.exec() != QDialog.Accepted:
            return
        o = dlg.values()
        suffix_name = {"timbre": "_timbre", "buildup": "_buildup", "melody": "_melody"}.get(audio_mode or o.audio, "")
        if kind == "mp4" and o.audio == "none":
            suffix_name = ""
        path = self._pick_save(f"Экспорт {kind.upper()}", f"{kind.upper()} (*.{kind})",
                               f"{_safe_name(self.current.name)}{suffix_name}.{kind}", "export_dir", f".{kind}")
        if not path:
            return
        for key, val in (("width", o.width), ("height", o.height), ("fps", o.fps),
                         ("duration", o.duration), ("supersample", o.supersample)):
            self.settings.set(f"export.{key}", val)
        self.settings.set("audio.steps", o.steps)
        self.settings.set("audio.step_seconds", o.step_seconds)
        ap, mp = self.audio_panel.params(), self.audio_panel.melody_params()
        c = self.ctrl
        job = ExportJob(kind=kind, path=path, spectrum=c.spectrum.copy(), target=self.z.copy(),
                        max_epicycles=c.max_epicycles, trail_fraction=c.trail_fraction, phase=c.phase,
                        width=o.width, height=o.height, fps=o.fps, duration=o.duration,
                        supersample=o.supersample,
                        style=Style(show_circles=o.circles, show_vectors=o.vectors, show_target=o.target),
                        audio=o.audio, f0=ap.f0, audio_stereo=ap.stereo, steps=o.steps,
                        step_seconds=o.step_seconds, revolution_seconds=o.revolution_seconds,
                        voices=mp.voices, bar_seconds=mp.bar_seconds, scale=mp.scale,
                        root_midi=mp.root_midi, quantize=mp.quantize)
        try:
            validate_job(job)
        except ExportError as e:
            self._error(str(e))
            return
        self._run_job(job)

    def _run_job(self, job: ExportJob):
        w = ProgressWorker(run_export, job, cancel_exc=(ExportCancelled,))
        self._export_worker = w
        dlg = QProgressDialog(f"Экспорт {job.kind.upper()}: {Path(job.path).name}", "Отмена", 0,
                              job.n_frames, self)
        dlg.setWindowTitle("Экспорт")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(300)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.canceled.connect(w.cancel)
        w.signals.progress.connect(lambda i, n: dlg.setValue(i))
        w.signals.done.connect(lambda p: self._export_finished(dlg, f"Сохранено: {p}"))
        w.signals.cancelled.connect(lambda: self._export_finished(dlg, "Экспорт отменён"))
        w.signals.failed.connect(lambda msg: self._export_finished(dlg, None, msg))
        log.info("Экспорт %s → %s (%d кадров)", job.kind, job.path, job.n_frames)
        start(w)

    def _export_finished(self, dlg, message, error=None):
        self._export_worker = None
        dlg.close()
        dlg.deleteLater()
        if error:
            self._error(error)
        else:
            self.status.setText(message)

    def _export_coeffs_csv(self):
        path = self._pick_save("Коэффициенты (CSV)", "CSV (*.csv)",
                               f"{_safe_name(self.current.name)}_coefficients.csv", "export_dir", ".csv")
        if path:
            self._write_csv(lambda: export_coefficients_csv(path, self.ctrl.spectrum), path)

    def _export_contour_csv(self):
        path = self._pick_save("Контуры (CSV)", "CSV (*.csv)",
                               f"{_safe_name(self.current.name)}_contour.csv", "export_dir", ".csv")
        if path:
            self._write_csv(lambda: export_contour_csv(path, self.z, self.ctrl.spectrum), path)

    def _write_csv(self, fn, path):
        try:
            fn()
        except OSError as e:
            log.exception("Ошибка записи CSV")
            self._error(f"Не удалось записать файл: {e}")
            return
        log.info("CSV сохранён: %s", path)
        self.status.setText(f"Сохранено: {path}")

    def _show_log_location(self):
        QMessageBox.information(self, "Журнал", f"Файл журнала:\n{log_dir() / 'fourier_atlas.log'}")

    # ---------- настройки окна ----------
    def _restore_window(self):
        st = self.settings
        try:
            if st.get("window_geometry"):
                self.restoreGeometry(QByteArray.fromBase64(st.get("window_geometry").encode()))
            if st.get("window_state"):
                self.restoreState(QByteArray.fromBase64(st.get("window_state").encode()))
        except Exception:                                  # noqa: BLE001 — повреждённые данные не критичны
            log.warning("Не удалось восстановить геометрию окна", exc_info=True)

    def closeEvent(self, e):
        st = self.settings
        try:
            st.set("window_geometry", bytes(self.saveGeometry().toBase64()).decode())
            st.set("window_state", bytes(self.saveState().toBase64()).decode())
            st.set("country", self.cmb.currentText())
            st.set("projection", self.opts.projection)
            st.set("max_epicycles", int(self.sp_vec.value()))
            st.set("speed", float(self.ctrl.speed))
            st.set("trail_fraction", float(self.ctrl.trail_fraction))
            st.set("show_circles", self.canvas.show_circles)
            st.set("show_vectors", self.canvas.show_vectors)
            st.set("show_target", self.canvas.show_target)
            for key, val in self.audio_panel.values().items():
                st.set(f"audio.{key}", val)
        except (ValueError, KeyError):
            log.warning("Часть настроек не сохранена", exc_info=True)
        st.save()
        if self._export_worker is not None:
            self._export_worker.cancel()
        self.audio_panel.shutdown()
        QThreadPool.globalInstance().waitForDone(3000)
        log.info("Окно закрыто")
        super().closeEvent(e)
