"""Подготовка геометрии: замыкание, равномерная дискретизация, нормализация.

Контур хранится как массив float (M, 2) либо как комплексный массив z = x + i*y.
"""
from __future__ import annotations
import numpy as np


def to_complex(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Ожидался массив формы (M, 2)")
    return xy[:, 0] + 1j * xy[:, 1]


def from_complex(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    return np.column_stack([z.real, z.imag])


def close_contour(xy: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Замыкает контур (последняя точка == первая) и убирает подряд идущие дубликаты."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Ожидался массив формы (M, 2)")
    if not np.all(np.isfinite(xy)):
        raise ValueError("Контур содержит NaN/inf")
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.hypot(*np.diff(xy, axis=0).T) > tol
    xy = xy[keep]
    if len(xy) > 1 and np.hypot(*(xy[0] - xy[-1])) <= tol:
        xy = xy[:-1]                      # уже замкнут — временно снимаем дубль
    if len(xy) < 3:
        raise ValueError("Для контура нужно минимум 3 различные точки")
    return np.vstack([xy, xy[0]])


def contour_length(xy_closed: np.ndarray) -> float:
    return float(np.hypot(*np.diff(xy_closed, axis=0).T).sum())


def resample_arclength(xy: np.ndarray, n: int) -> np.ndarray:
    """Равномерная по длине дуги выборка n точек замкнутого контура.

    Возвращает (n, 2) БЕЗ повторной замыкающей точки: s_j = j*L/n, j=0..n-1.
    Старт — первая точка исходного контура.
    """
    if n < 4:
        raise ValueError("n должно быть >= 4")
    c = close_contour(xy)
    seg = np.hypot(*np.diff(c, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        raise ValueError("Нулевая длина контура")
    targets = np.arange(n) * (total / n)
    x = np.interp(targets, s, c[:, 0])
    y = np.interp(targets, s, c[:, 1])
    return np.column_stack([x, y])


def normalize_contour(z: np.ndarray, flip_y: bool = False):
    """Центрирует по среднему и масштабирует так, чтобы max|z| = 1.

    Возвращает (z_norm, center, scale), где z = z_norm*scale + center
    (если flip_y=False). flip_y зеркалит ось Y (SVG/экранные координаты).
    """
    z = np.asarray(z, dtype=complex)
    if flip_y:
        z = np.conj(z)
    center = z.mean()
    zc = z - center
    scale = float(np.max(np.abs(zc)))
    if scale == 0:
        raise ValueError("Вырожденный контур")
    return zc / scale, center, scale
