"""Диалог параметров экспорта (PNG, SVG, GIF, MP4 [со звуком], WAV)."""
from __future__ import annotations
from dataclasses import dataclass
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QLabel, QSpinBox)

PRESETS = [("1280 × 720 (HD)", 1280, 720), ("1920 × 1080 (Full HD)", 1920, 1080),
           ("1080 × 1080 (квадрат)", 1080, 1080), ("720 × 720 (квадрат)", 720, 720),
           ("640 × 480", 640, 480), ("3840 × 2160 (4K)", 3840, 2160)]
SUPERSAMPLE = [("Без сглаживания (быстро)", 1), ("Среднее (2×)", 2), ("Высокое (3×, медленно)", 3)]
AUDIO_MODES = [("Без звука", "none"), ("Со звуком: тембр контура", "timbre"),
               ("Со звуком: наращивание гармоник", "buildup"), ("Со звуком: мелодия", "melody")]


@dataclass
class ExportOptions:
    width: int
    height: int
    fps: int
    duration: float
    supersample: int
    circles: bool
    vectors: bool
    target: bool
    audio: str = "none"
    steps: int = 24
    step_seconds: float = 0.6
    revolution_seconds: float = 4.0


class ExportDialog(QDialog):
    """kind: png | svg | gif | mp4 | wav | mid. Для wav audio_mode ('timbre'/'buildup'/'melody') задан заранее."""

    def __init__(self, kind: str, defaults: dict, flags: dict, audio_defaults: dict | None = None,
                 audio_mode: str | None = None, parent=None):
        super().__init__(parent)
        self.kind, self.fixed_audio = kind, audio_mode
        ad = audio_defaults or {"steps": 24, "step_seconds": 0.6}
        names = {"timbre": "тембр", "buildup": "наращивание", "melody": "мелодия"}
        self.bar_seconds = float(ad.get("bar_seconds", 4.0))
        self.setWindowTitle("Экспорт " + (f"WAV ({names.get(audio_mode, '')})" if kind == "wav"
                                           else "MIDI (мелодия)" if kind == "mid" else kind.upper()))
        side_max = 1920 if kind == "gif" else 8192
        self.form = QFormLayout(self)
        form = self.form
        self._visual_rows = []

        def row(label, widget, visual=False):
            form.addRow(label, widget)
            if visual:
                self._visual_rows.append(widget)
            return widget

        self.preset = QComboBox()
        self.preset.addItem("Свой размер", None)
        for title, w, h in PRESETS:
            if max(w, h) <= side_max:
                self.preset.addItem(title, (w, h))
        self.w = QSpinBox(minimum=16, maximum=side_max, value=min(defaults["width"], side_max))
        self.h = QSpinBox(minimum=16, maximum=side_max, value=min(defaults["height"], side_max))
        self.preset.currentIndexChanged.connect(self._on_preset)
        row("Размер", self.preset, True)
        row("Ширина, px", self.w, True)
        row("Высота, px", self.h, True)

        self.fps = QSpinBox(minimum=1, maximum=120, value=defaults["fps"])
        self.dur = QDoubleSpinBox(minimum=0.2, maximum=600.0, singleStep=0.5, decimals=1,
                                  value=defaults["duration"], suffix=" с")
        self.audio = QComboBox()
        for title, v in AUDIO_MODES:
            self.audio.addItem(title, v)
        self.steps = QSpinBox(minimum=1, maximum=200, value=ad["steps"])
        self.step_s = QDoubleSpinBox(minimum=0.1, maximum=10.0, singleStep=0.1, decimals=1,
                                     value=ad["step_seconds"], suffix=" с")
        self.rev = QDoubleSpinBox(minimum=0.5, maximum=120.0, singleStep=0.5, decimals=1, value=4.0, suffix=" с")
        self.info = QLabel()

        if kind in ("gif", "mp4"):
            row("Кадров в секунду", self.fps, True)
        if kind == "mp4":
            row("Звук", self.audio)
        row("Длительность оборота / звука", self.dur)
        row("Шагов наращивания", self.steps)
        row("Длительность шага", self.step_s)
        row("Время оборота контура в видео", self.rev)
        row("", self.info)

        self.ss = QComboBox()
        for title, v in SUPERSAMPLE:
            self.ss.addItem(title, v)
        self.ss.setCurrentIndex(max(0, self.ss.findData(defaults["supersample"])))
        row("Сглаживание", self.ss, True)
        self.cb_circles, self.cb_vectors, self.cb_target = (QCheckBox("Окружности"), QCheckBox("Векторы"),
                                                            QCheckBox("Исходный контур"))
        for cb, key in ((self.cb_circles, "circles"), (self.cb_vectors, "vectors"), (self.cb_target, "target")):
            cb.setChecked(flags[key])
            row("", cb, True)
        if kind in ("png", "svg"):
            form.addRow("", QLabel("Экспортируется текущий кадр (текущая фаза)."))

        if kind in ("wav", "mid"):
            for w in self._visual_rows:
                form.setRowVisible(w, False)
        for w in (self.fps, self.dur, self.audio, self.steps, self.step_s):
            sig = getattr(w, "currentIndexChanged", None) or getattr(w, "valueChanged")
            sig.connect(self._update_state)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._update_state()

    def _mode(self) -> str:
        if self.kind == "mid":
            return "melody"
        return self.fixed_audio or (self.audio.currentData() if self.kind == "mp4" else "none")

    def _update_state(self, *_):
        mode = self._mode()
        build = mode == "buildup"
        self.form.setRowVisible(self.dur, self.kind in ("gif", "mp4", "wav", "mid") and not build)
        for w in (self.steps, self.step_s):
            self.form.setRowVisible(w, build)
        self.form.setRowVisible(self.rev, build and self.kind == "mp4")
        bars = max(1, int(round(self.dur.value() / self.bar_seconds)))
        if mode == "melody":
            self.info.setText(f"Тактов: {bars} (по {self.bar_seconds:g} с, итого {bars * self.bar_seconds:g} с)"
                              + (f"; кадров: {int(round(self.fps.value() * bars * self.bar_seconds))}"
                                 if self.kind == "mp4" else ""))
        elif self.kind in ("gif", "mp4"):
            secs = self.steps.value() * self.step_s.value() if build else self.dur.value()
            n = int(round(self.fps.value() * secs))
            extra = "  (GIF большого размера может весить десятки МБ)" if self.kind == "gif" else ""
            self.info.setText(f"Кадров: {n}{extra}")
        elif self.kind == "wav" and build:
            self.info.setText(f"Длительность: до {self.steps.value() * self.step_s.value():.1f} с")

    def _on_preset(self, _):
        size = self.preset.currentData()
        if size:
            self.w.setValue(size[0])
            self.h.setValue(size[1])

    def values(self) -> ExportOptions:
        return ExportOptions(self.w.value(), self.h.value(), self.fps.value(), float(self.dur.value()),
                             int(self.ss.currentData()), self.cb_circles.isChecked(),
                             self.cb_vectors.isChecked(), self.cb_target.isChecked(),
                             audio=self._mode(), steps=self.steps.value(),
                             step_seconds=float(self.step_s.value()), revolution_seconds=float(self.rev.value()))
