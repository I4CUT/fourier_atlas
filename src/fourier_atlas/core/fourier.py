"""Комплексный ряд Фурье замкнутого контура.

НОРМИРОВКА (документировано):
  Контур — N точек z_n, n = 0..N-1, равномерно по длине дуги, параметр
  t_n = n/N ∈ [0, 1).  Коэффициенты:

      c_k = (1/N) * Σ_n z_n * exp(-2πi k n / N)  =  fft(z)[k] / N

  Частоты k — целые, порядок как у numpy.fft.fftfreq (0, 1, ..., -2, -1).
  Ряд:
      z(t) = Σ_k c_k * exp(2πi k t)

  При полном наборе частот z(t_n) = z_n точно (обратное ДПФ). k = 0 — центр
  масс контура, |c_k| — радиусы эпициклов, arg c_k — начальные фазы.
  Для чётного N частота Nyquist N/2 хранится как k = -N/2 (как в numpy).
  Реконструкция в промежуточных t — тригонометрическая интерполяция.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Spectrum:
    """Спектр: исходные коэффициенты неизменны, правки хранятся отдельно."""
    freqs: np.ndarray          # (N,) целые k
    original: np.ndarray       # (N,) complex — исходный спектр, не меняется
    edited: np.ndarray = field(default=None)   # рабочая копия
    enabled: np.ndarray = field(default=None)  # (N,) bool

    def __post_init__(self):
        self.freqs = np.asarray(self.freqs, dtype=int)
        self.original = np.asarray(self.original, dtype=complex)
        if self.edited is None:
            self.edited = self.original.copy()
        if self.enabled is None:
            self.enabled = np.ones(len(self.freqs), dtype=bool)
        self._index = {int(k): i for i, k in enumerate(self.freqs)}

    @property
    def n(self) -> int:
        return len(self.freqs)

    def index_of(self, k: int) -> int:
        return self._index[int(k)]

    # --- редактирование без потери исходного спектра ---
    def set_coeff(self, k: int, value: complex) -> None:
        self.edited[self.index_of(k)] = complex(value)

    def reset_coeff(self, k: int) -> None:
        i = self.index_of(k)
        self.edited[i] = self.original[i]

    def reset_all(self) -> None:
        self.edited = self.original.copy()
        self.enabled[:] = True

    def set_enabled(self, k: int, flag: bool) -> None:
        self.enabled[self.index_of(k)] = bool(flag)

    def select_band(self, max_harmonics: int) -> None:
        """Оставляет k=0 и все |k| <= max_harmonics."""
        self.enabled = np.abs(self.freqs) <= int(max_harmonics)

    def active(self):
        """(freqs, coeffs) только включённых гармоник с учётом правок."""
        m = self.enabled
        return self.freqs[m], self.edited[m]

    def amplitudes(self) -> np.ndarray:
        return np.abs(self.edited)

    def phases(self) -> np.ndarray:
        return np.angle(self.edited)

    def copy(self) -> "Spectrum":
        """Независимая копия (для фонового экспорта: поток не трогает живой спектр)."""
        return Spectrum(self.freqs.copy(), self.original.copy(),
                        self.edited.copy(), self.enabled.copy())

    # --- история правок ---
    def snapshot(self):
        return self.edited.copy(), self.enabled.copy()

    def restore(self, snap) -> None:
        self.edited, self.enabled = snap[0].copy(), snap[1].copy()

    def edited_mask(self) -> np.ndarray:
        """True там, где коэффициент изменён вручную относительно исходного."""
        return self.edited != self.original

    def coeff(self, k: int) -> complex:
        return complex(self.edited[self.index_of(k)])

    def sorted_by_frequency(self):
        order = np.argsort(self.freqs)
        return self.freqs[order], self.edited[order]


def compute_spectrum(z: np.ndarray) -> Spectrum:
    """Один раз считает FFT (не вызывать на каждом кадре!)."""
    z = np.asarray(z, dtype=complex)
    if z.ndim != 1 or len(z) < 4:
        raise ValueError("Ожидался 1D комплексный массив длиной >= 4")
    n = len(z)
    c = np.fft.fft(z) / n
    k = np.rint(np.fft.fftfreq(n, d=1.0 / n)).astype(int)
    return Spectrum(freqs=k, original=c)


def reconstruct_at(freqs: np.ndarray, coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """z(t) = Σ c_k exp(2πi k t) для произвольных t (в долях периода).

    Память: обработка блоками, чтобы не создавать матрицу len(t) × K целиком.
    """
    freqs = np.asarray(freqs, dtype=float)
    coeffs = np.asarray(coeffs, dtype=complex)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(t.shape, dtype=complex)
    block = max(1, int(4_000_000 // max(1, len(freqs))))
    for a in range(0, len(t), block):
        tb = t[a:a + block]
        out[a:a + block] = np.exp(2j * np.pi * np.outer(tb, freqs)) @ coeffs
    return out


def reconstruct(spec: Spectrum, n_points: int | None = None,
                use_edits: bool = True, respect_mask: bool = True) -> np.ndarray:
    """Реконструкция n_points точек t_j = j/n_points по выбранным частотам."""
    n_points = n_points or spec.n
    coeffs = spec.edited if use_edits else spec.original
    freqs = spec.freqs
    if respect_mask:
        freqs, coeffs = freqs[spec.enabled], coeffs[spec.enabled]
    t = np.arange(n_points) / n_points
    return reconstruct_at(freqs, coeffs, t)
