"""Цепочка эпициклов: позиции векторов для заданной фазы без FFT."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .fourier import Spectrum


@dataclass
class EpicycleChain:
    freqs: np.ndarray    # (K,) частоты в порядке цепочки
    coeffs: np.ndarray   # (K,) комплексные коэффициенты
    radii: np.ndarray    # (K,) |c|

    def joints(self, t: float) -> np.ndarray:
        """K+1 точек: начало (0) и конец каждого вектора при фазе t ∈ [0,1)."""
        vecs = self.coeffs * np.exp(2j * np.pi * self.freqs * t)
        return np.concatenate([[0j], np.cumsum(vecs)])

    def tip(self, t: float) -> complex:
        return complex(self.joints(t)[-1])

    def tips(self, ts: np.ndarray) -> np.ndarray:
        """Концевые точки для массива фаз (для следа/кэша кадров)."""
        ph = np.exp(2j * np.pi * np.outer(np.asarray(ts, float), self.freqs))
        return ph @ self.coeffs


def build_chain(spec: Spectrum, max_vectors: int | None = None,
                max_rotating: int | None = None) -> EpicycleChain:
    """Включённые гармоники; k=0 (центр) первым, далее по убыванию амплитуды
    (крупные круги первыми; сумма от порядка не зависит).

    max_rotating — число ВРАЩАЮЩИХСЯ эпициклов (k != 0); k=0 не вращается и
    в это число не входит (0 → остаётся только центр). max_vectors — общий
    предел длины цепочки вместе с k=0 (для совместимости)."""
    f, c = spec.active()
    order = np.lexsort((np.abs(f), -np.abs(c)))   # по убыванию |c|, при равенстве меньший |k|
    is_zero = f[order] == 0
    rot = order[~is_zero]
    if max_rotating is not None:
        if max_rotating < 0:
            raise ValueError("max_rotating >= 0")
        rot = rot[:max_rotating]
    order = np.concatenate([order[is_zero], rot])
    if max_vectors is not None:
        order = order[:max_vectors]
    return EpicycleChain(freqs=f[order].astype(float), coeffs=c[order], radii=np.abs(c[order]))
