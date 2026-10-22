"""Оценка точности реконструкции и кривая сходимости."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .fourier import Spectrum


def _check(a, b):
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    if a.shape != b.shape:
        raise ValueError("Контуры должны иметь одинаковое число точек")
    return a, b


def rmse(a, b) -> float:
    a, b = _check(a, b)
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)))


def max_error(a, b) -> float:
    a, b = _check(a, b)
    return float(np.max(np.abs(a - b)))


def hausdorff(a, b, block: int = 512) -> float:
    """Симметричное дискретное расстояние Хаусдорфа (не зависит от сдвига фазы)."""
    a, b = np.asarray(a, complex), np.asarray(b, complex)

    def directed(p, q):
        worst = 0.0
        for i in range(0, len(p), block):
            d = np.abs(p[i:i + block, None] - q[None, :]).min(axis=1)
            worst = max(worst, float(d.max()))
        return worst
    return max(directed(a, b), directed(b, a))


@dataclass
class ReconstructionError:
    rmse: float
    max_error: float
    hausdorff: float
    relative_rmse: float   # rmse / диагональ bbox оригинала


def evaluate(original: np.ndarray, recon: np.ndarray) -> ReconstructionError:
    o, r = _check(original, recon)
    diag = float(np.hypot(np.ptp(o.real), np.ptp(o.imag))) or 1.0
    e = rmse(o, r)
    return ReconstructionError(e, max_error(o, r), hausdorff(o, r), e / diag)


def convergence_curve(spec: Spectrum, max_m: int | None = None):
    """RMSE(M) для полосы |k| <= M по ИСХОДНОМУ спектру, M = 0..max_m.

    Без реконструкций: по теореме Парсеваля для ДПФ
        RMSE(M)² = Σ_{|k|>M} |c_k|²    (на узлах выборки).
    """
    m_all = int(np.max(np.abs(spec.freqs)))
    max_m = m_all if max_m is None else min(int(max_m), m_all)
    p = np.abs(spec.original) ** 2
    energy = np.bincount(np.abs(spec.freqs), weights=p, minlength=m_all + 1)
    tail = np.append(np.cumsum(energy[::-1])[::-1], 0.0)   # tail[m] = Σ_{|k|>=m}
    ms = np.arange(0, max_m + 1)
    return ms, np.sqrt(np.maximum(tail[ms + 1], 0.0))


def harmonics_for_rmse(spec: Spectrum, target: float) -> int:
    """Минимальное M, при котором RMSE <= target."""
    ms, err = convergence_curve(spec)
    idx = np.where(err <= target)[0]
    return int(ms[idx[0]]) if len(idx) else int(ms[-1])
