"""Сравнение исходного и восстановленного контуров, ошибки."""
from __future__ import annotations
import numpy as np

from ..core import ReconstructionError, convergence_by_amplitude, evaluate, reconstruct


def contour_diagonal(z: np.ndarray) -> float:
    z = np.asarray(z, complex)
    return float(np.hypot(np.ptp(z.real), np.ptp(z.imag))) or 1.0


def compare_reconstruction(z: np.ndarray, spec) -> tuple[np.ndarray, ReconstructionError]:
    """Реконструкция по ТЕКУЩЕМУ состоянию спектра (правки + маска) в точках исходного контура."""
    recon = reconstruct(spec, len(z))
    return recon, evaluate(z, recon)


def relative_error_percent(rmse_value: float, z: np.ndarray) -> float:
    """RMSE в процентах диагонали ограничивающего прямоугольника исходного контура."""
    return 100.0 * rmse_value / contour_diagonal(z)


def epicycles_for_relative_error(spec, z: np.ndarray, percent: float):
    """Минимальное число вращающихся эпициклов (по убыванию |c|) для относительной ошибки <= percent %.
    None, если даже все гармоники не дают такой точности (ошибка вычислений)."""
    target = percent / 100.0 * contour_diagonal(z)
    ns, err = convergence_by_amplitude(spec)
    idx = np.where(err <= target)[0]
    return int(ns[idx[0]]) if len(idx) else None
