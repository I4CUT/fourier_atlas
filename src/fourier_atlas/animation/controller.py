"""Контроллер анимации: фаза, скорость, пауза, след — без Qt и без FFT на кадре.

При изменении спектра вызывается refresh(): цепочка эпициклов и кэш траектории
(path) пересчитываются ОДИН раз. Каждый кадр — это лишь cumsum по K векторам
и срез кэша для следа.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core import Spectrum, build_chain, reconstruct_at


@dataclass
class Frame:
    phase: float            # [0, 1)
    joints: np.ndarray      # (K+1,) complex: центры окружностей и конец цепочки
    radii: np.ndarray       # (K,) радиусы окружностей
    trail: np.ndarray       # (m,) complex: след конечной точки, последний = joints[-1]


class AnimationController:
    def __init__(self, spectrum: Spectrum, max_epicycles: int | None = None,
                 path_samples: int = 1024, trail_fraction: float = 1.0,
                 speed: float = 0.1):
        if path_samples < 8:
            raise ValueError("path_samples должно быть >= 8")
        self.spectrum = spectrum
        self.path_samples = int(path_samples)
        self._max_epicycles = None
        self._max_epicycles = self._check_n(max_epicycles)
        self._trail_fraction = 1.0
        self.trail_fraction = trail_fraction
        self.speed = float(speed)          # периодов в секунду, может быть < 0
        self.playing = False
        self._phase = 0.0
        self.chain = None
        self.path = None
        self.refresh()

    # ---------- параметры ----------
    @property
    def phase(self) -> float:
        return self._phase

    @staticmethod
    def _check_n(value):
        if value is not None and value < 0:
            raise ValueError("max_epicycles >= 0")
        return None if value is None else int(value)

    @property
    def max_epicycles(self):
        """Число вращающихся эпициклов (None — все). 0 — только центр (k=0)."""
        return self._max_epicycles

    @max_epicycles.setter
    def max_epicycles(self, value):
        self._max_epicycles = self._check_n(value)
        self.refresh()

    @property
    def trail_fraction(self) -> float:
        return self._trail_fraction

    @trail_fraction.setter
    def trail_fraction(self, value: float):
        if not (0.0 < value <= 1.0):
            raise ValueError("trail_fraction должен быть в (0, 1]")
        self._trail_fraction = float(value)

    # ---------- синхронизация со спектром ----------
    def set_spectrum(self, spectrum: Spectrum) -> None:
        self.spectrum = spectrum
        self.refresh()

    def refresh(self) -> None:
        """Пересчитать цепочку и кэш траектории (вызывать после любых правок спектра)."""
        self.chain = build_chain(self.spectrum, max_rotating=self._max_epicycles)
        ts = np.arange(self.path_samples) / self.path_samples
        self.path = reconstruct_at(self.chain.freqs, self.chain.coeffs, ts)

    # ---------- управление ----------
    def play(self): self.playing = True
    def pause(self): self.playing = False

    def toggle(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def reset(self) -> None:
        self.playing = False
        self._phase = 0.0

    def seek(self, phase: float) -> None:
        self._phase = float(phase) % 1.0

    def advance(self, dt: float) -> bool:
        """Сдвинуть фазу на dt секунд. True, если фаза изменилась."""
        if not self.playing or dt <= 0 or self.speed == 0:
            return False
        self._phase = (self._phase + self.speed * dt) % 1.0
        return True

    # ---------- кадр ----------
    def frame(self) -> Frame:
        joints = self.chain.joints(self._phase)
        m = max(2, int(round(self._trail_fraction * self.path_samples)))
        end = int(np.floor(self._phase * self.path_samples)) % self.path_samples
        sign = 1 if self.speed >= 0 else -1
        idx = (end - sign * np.arange(m - 1, -1, -1)) % self.path_samples
        trail = np.append(self.path[idx], joints[-1])
        return Frame(self._phase, joints, self.chain.radii, trail)
