"""Встроенные демонстрационные контуры (до подключения базы стран, этап 3)."""
from __future__ import annotations
import numpy as np
from ..core import (resample_arclength, to_complex, normalize_contour,
                    compute_spectrum, Spectrum)


def _circle():
    a = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    return np.column_stack([np.cos(a), np.sin(a)])


def _ellipse():
    a = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    return np.column_stack([3 * np.cos(a), np.sin(a)])


def _square():
    return np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)


def _star():
    pts = []
    for i in range(10):
        r = 1.0 if i % 2 == 0 else 0.4
        a = np.pi / 2 + i * np.pi / 5
        pts.append([r * np.cos(a), r * np.sin(a)])
    return np.array(pts)


def _heart():
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.column_stack([x, y])


SAMPLES = {"Окружность": _circle, "Эллипс": _ellipse, "Квадрат": _square,
           "Звезда": _star, "Сердце": _heart}


def contour_to_spectrum(xy: np.ndarray, n: int = 1024):
    """xy -> (z_нормированный, Spectrum). Единый конвейер подготовки геометрии."""
    z = to_complex(resample_arclength(xy, n))
    z, _, _ = normalize_contour(z)
    return z, compute_spectrum(z)


def make_sample(name: str, n: int = 1024):
    return contour_to_spectrum(SAMPLES[name](), n)
