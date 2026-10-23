"""Определение гармоники под курсором на холсте анимации (без Qt)."""
from __future__ import annotations
import numpy as np


def pick_harmonic(frame, chain_freqs, viewport, px: float, py: float,
                  tol: float = 7.0, min_radius_px: float = 1.0):
    """Частота k эпицикла, окружность или вектор которого ближе всего к точке (px, py) экрана.

    Учитываются только видимые окружности (радиус >= min_radius_px). None, если
    ближайший элемент дальше tol пикселей.
    """
    r = np.asarray(frame.radii, float) * viewport.scale
    if len(r) == 0:
        return None
    jx, jy = viewport.to_screen(frame.joints)
    ax, ay, bx, by = jx[:-1], jy[:-1], jx[1:], jy[1:]
    ring = np.abs(np.hypot(ax - px, ay - py) - r)
    abx, aby = bx - ax, by - ay
    l2 = abx**2 + aby**2
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / np.where(l2 > 0, l2, 1.0), 0.0, 1.0)
    seg = np.hypot(px - (ax + t * abx), py - (ay + t * aby))
    score = np.where(r >= min_radius_px, np.minimum(ring, seg), np.inf)
    i = int(np.argmin(score))
    return int(chain_freqs[i]) if score[i] <= tol else None
