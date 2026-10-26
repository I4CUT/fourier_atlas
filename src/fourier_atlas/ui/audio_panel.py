"""Вкладка «Звук»: тембр контура, «наращивание» и «мелодия» (QSoundEffect)."""
from __future__ import annotations
import logging
import os
import shutil
import tempfile
from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ..audio import (SCALES, AudioError, AudioParams, MelodyParams, audible_chain, buildup_buffer,
                     buildup_schedule, loop_buffer, loop_melody, max_harmonic, midi_to_freq, note_name,
                     select_voices, write_wav)

log = logging.getLogger(__name__)
QUANT_TITLES = {0: "Без квантования", 4: "Сетка 1/4", 8: "Сетка 1/8", 16: "Сетка 1/16", 32: "Сетка 1/32"}


class AudioPlayer:
    """Обёртка над QSoundEffect: бесконечный цикл или одно проигрывание. Файлы чередуются (a/b)."""

    def __init__(self):
        from PySide6.QtMultimedia import QSoundEffect          # ImportError → панель отключит проигрывание
        inf = getattr(QSoundEffect, "Infinite", None)
        if inf is None:
            inf = QSoundEffect.Loop.Infinite
        self._inf = getattr(inf, "value", inf)
        self.dir = tempfile.mkdtemp(prefix="fatlas_snd_")
        self.effect = QSoundEffect()
        self._n = 0

    def play(self, wave, sample_rate: int, loop: bool, volume: float) -> None:
        path = os.path.join(self.dir, f"s{self._n % 2}.wav")
        self._n += 1
        write_wav(path, wave, sample_rate)
        self.effect.stop()
        self.effect.setSource(QUrl.fromLocalFile(path))
        self.effect.setLoopCount(self._inf if loop else 1)
        self.effect.setVolume(volume)
        self.effect.play()

    def set_volume(self, v: float) -> None:
        self.effect.setVolume(v)

    def stop(self) -> None:
        self.effect.stop()

    @property
    def playing(self) -> bool:
        return bool(self.effect.isPlaying())

    def cleanup(self) -> None:
        self.effect.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


class AudioPanel(QWidget):
    MODES = ("timbre", "buildup", "melody")
    save_requested = Signal(str, str)          # (вид файла: 'wav' | 'mid', режим звука)

    def __init__(self, hub, controller, settings, parent=None):
        super().__init__(parent)
        self.hub, self.ctrl = hub, controller
        a = settings.get("audio")
        try:
            self.player = AudioPlayer()
            self.player_error = ""
        except Exception as e:                                  # noqa: BLE001
            self.player, self.player_error = None, str(e)
            log.warning("QtMultimedia недоступен: %s", e)
        self._mode = None
        self._steps, self._step_seconds = a["steps"], a["step_seconds"]

        # --- тембр ---
        self.f0 = QDoubleSpinBox(minimum=20.0, maximum=2000.0, decimals=1, singleStep=1.0, value=a["f0"], suffix=" Гц")
        self.note = QLabel()
        self.stereo = QComboBox()
        self.stereo.addItem("Стерео XY: левый = x(t), правый = y(t)", True)
        self.stereo.addItem("Моно: (x + y) / 2", False)
        self.stereo.setCurrentIndex(0 if a["stereo"] else 1)
        self.volume = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=int(round(a["volume"] * 100)))

        # --- мелодия ---
        self.scale = QComboBox()
        for key, (title, _) in SCALES.items():
            self.scale.addItem(title, key)
        self.scale.setCurrentIndex(max(0, self.scale.findData(a["scale"])))
        self.root = QSpinBox(minimum=24, maximum=84, value=a["root"])
        self.root_note = QLabel()
        self.voices = QSpinBox(minimum=1, maximum=32, value=a["voices"])
        self.bar = QDoubleSpinBox(minimum=1.0, maximum=30.0, decimals=1, singleStep=0.5, value=a["bar_seconds"],
                                  suffix=" с")
        self.quant = QComboBox()
        for q, title in QUANT_TITLES.items():
            self.quant.addItem(title, q)
        self.quant.setCurrentIndex(max(0, self.quant.findData(a["quantize"])))

        self.btn = {m: QPushButton() for m in self.MODES}
        for m, b in self.btn.items():
            b.clicked.connect(lambda _=False, m=m: self._toggle(m))
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.melody_info = QLabel()
        self.melody_info.setWordWrap(True)
        hint = QLabel(
            "Тембр: контур — периодическая волна, гармоника k звучит на k·f0 (на осциллографе XY рисуется контур). "
            "Наращивание: гармоники подключаются по убыванию амплитуды. Мелодия: каждый из крупнейших эпициклов играет "
            "ноту, когда его вектор проходит направление «вверх» (гармоника k — |k| нот за такт, один оборот контура = "
            "такт); высота растёт с |k|, громкость — |c_k|, вращение против часовой — вправо, по часовой — влево.")
        hint.setWordWrap(True)

        def row(*widgets):
            h = QHBoxLayout()
            for w in widgets:
                h.addWidget(w)
            h.addStretch(1)
            return h

        lay = QVBoxLayout(self)
        lay.addLayout(row(QLabel("Основная частота f0:"), self.f0, self.note, self.stereo, QLabel("Громкость"), self.volume))
        lay.addLayout(row(QLabel("Лад:"), self.scale, QLabel("Тоника (MIDI):"), self.root, self.root_note,
                          QLabel("Голосов:"), self.voices, QLabel("Такт:"), self.bar, self.quant))
        lay.addLayout(row(*self.btn.values()))
        save_row = [QLabel("Сохранить в файл:")]
        for title, kind, mode in (("💾 Мелодия → WAV…", "wav", "melody"), ("💾 Мелодия → MIDI…", "mid", "melody"),
                                  ("💾 Тембр → WAV…", "wav", "timbre"), ("💾 Наращивание → WAV…", "wav", "buildup")):
            b = QPushButton(title)
            b.clicked.connect(lambda _=False, k=kind, m=mode: self.save_requested.emit(k, m))
            save_row.append(b)
        lay.addLayout(row(*save_row))
        lay.addWidget(self.info)
        lay.addWidget(self.melody_info)
        lay.addWidget(hint)
        lay.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._apply_change)
        hub.changed.connect(lambda: self._timer.start(200))
        hub.replaced.connect(self._on_replaced)
        for w in (self.f0, self.root, self.voices, self.bar):
            w.valueChanged.connect(lambda _: self._timer.start(200))
        for w in (self.stereo, self.scale, self.quant):
            w.currentIndexChanged.connect(lambda _: self._timer.start(200))
        self.volume.valueChanged.connect(lambda v: self.player and self.player.set_volume(v / 100))
        if self.player is not None:
            self.player.effect.playingChanged.connect(self._on_playing_changed)
        else:
            for b in self.btn.values():
                b.setEnabled(False)
        self._update_buttons()
        self._update_info()

    # ---------- параметры ----------
    def params(self) -> AudioParams:
        return AudioParams(f0=float(self.f0.value()), stereo=bool(self.stereo.currentData()))

    def melody_params(self) -> MelodyParams:
        return MelodyParams(voices=self.voices.value(), bar_seconds=float(self.bar.value()),
                            scale=self.scale.currentData(), root_midi=self.root.value(),
                            quantize=int(self.quant.currentData()))

    def values(self) -> dict:
        return {"f0": float(self.f0.value()), "stereo": bool(self.stereo.currentData()),
                "volume": self.volume.value() / 100, "voices": self.voices.value(),
                "bar_seconds": float(self.bar.value()), "scale": self.scale.currentData(),
                "root": self.root.value(), "quantize": int(self.quant.currentData())}

    def _spec(self):
        return self.hub.editor.spectrum

    # ---------- проигрывание ----------
    def _toggle(self, mode: str):
        if self._mode == mode:
            self.stop()
        else:
            self._start(mode)

    def _start(self, mode: str):
        if self.player is None:
            return
        p, spec = self.params(), self._spec()
        try:
            if mode == "timbre":
                buf, loop = loop_buffer(spec, p, 1.5, self.ctrl.max_epicycles), True
            elif mode == "melody":
                mp = self.melody_params()
                if not select_voices(spec, mp):
                    raise AudioError("нет подходящих гармоник (все отключены или слишком быстрые для такта)")
                buf, loop = loop_melody(spec, mp, p), True
            else:
                sched = buildup_schedule(len(audible_chain(spec, p)[0]), self._steps)
                buf, loop = buildup_buffer(spec, p, sched, self._step_seconds), False
            self.player.play(buf, p.sample_rate, loop=loop, volume=self.volume.value() / 100)
        except AudioError as e:
            self.melody_info.setText(f"Звук не сформирован: {e}")
            self._mode = None
            self._update_buttons()
            return
        self._mode = mode
        self._update_buttons()

    def stop(self):
        if self.player is not None:
            self.player.stop()
        self._mode = None
        self._update_buttons()

    def _on_playing_changed(self):
        if self.player is not None and not self.player.playing and self._mode == "buildup":
            self._mode = None                                   # наращивание доиграло
            self._update_buttons()

    def _update_buttons(self):
        titles = {"timbre": "тембр контура", "buildup": "наращивание", "melody": "мелодия"}
        icons = {"timbre": "▶", "buildup": "▶", "melody": "♫"}
        for m, b in self.btn.items():
            b.setText("■ Стоп" if self._mode == m else f"{icons[m]} {titles[m].capitalize()}")

    def _on_replaced(self):
        self.stop()
        self._update_info()

    def _apply_change(self):
        self._update_info()
        if self._mode in ("timbre", "melody"):                  # звук следует за правками «вживую»
            self._start(self._mode)

    def _update_info(self):
        self.root_note.setText(f"≈ {note_name(midi_to_freq(self.root.value()))}")
        if self.player is None:
            self.info.setText(f"Проигрывание недоступно (QtMultimedia): {self.player_error}. "
                              "Экспорт WAV, MIDI и MP4 работает.")
            return
        p, spec = self.params(), self._spec()
        n = len(audible_chain(spec, p, self.ctrl.max_epicycles)[0])
        total = int((spec.enabled & (spec.freqs != 0)).sum())
        self.note.setText(f"≈ {note_name(p.f0)}")
        self.info.setText(f"Тембр: звучит гармоник {n} из {total} включённых (|k| ≤ {max_harmonic(p)}, выше "
                          f"{p.max_freq / 1000:.0f} кГц отбрасываются); число гармоник = число эпициклов.")
        try:
            voices = select_voices(spec, self.melody_params())
        except AudioError as e:
            self.melody_info.setText(str(e))
            return
        parts = [f"k={v.k:+d} {note_name(v.freq)} ×{abs(v.k)}" for v in voices[:10]]
        self.melody_info.setText("Мелодия (голос: нота × нот за такт): " + ", ".join(parts)
                                 + (" …" if len(voices) > 10 else "") if voices else
                                 "Мелодия: нет подходящих гармоник (отключены или слишком быстрые для такта).")

    def shutdown(self):
        if self.player is not None:
            self.player.cleanup()
