"""Преобразование мировых координат (комплексных, Y вверх) в экранные (Y вниз)."""
from __future__ import annotations
import numpy as np


class Viewport:
    def __init__(self, width: int = 800, height: int = 600):
        self.width, self.height = max(1, width), max(1, height)
        self.center = 0j
        self.scale = 100.0

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = max(1, width), max(1, height)

    def fit(self, points: np.ndarray, margin: float = 0.08) -> None:
        pts = np.asarray(points, complex)
        lo = complex(pts.real.min(), pts.imag.min())
        hi = complex(pts.real.max(), pts.imag.max())
        w = max(hi.real - lo.real, 1e-12)
        h = max(hi.imag - lo.imag, 1e-12)
        self.center = (lo + hi) / 2
        usable = 1.0 - 2 * margin
        self.scale = usable * min(self.width / w, self.height / h)

    def to_screen(self, z):
        """Возвращает (sx, sy) — массивы/скаляры пикселей."""
        d = (np.asarray(z, complex) - self.center) * self.scale
        return self.width / 2 + d.real, self.height / 2 - d.imag

    def to_world(self, sx, sy) -> complex:
        return self.center + complex((sx - self.width / 2) / self.scale,
                                     -(sy - self.height / 2) / self.scale)

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Масштабирование с неподвижной точкой под курсором."""
        if factor <= 0:
            raise ValueError("factor > 0")
        anchor = self.to_world(sx, sy)
        self.scale = float(np.clip(self.scale * factor, 1e-6, 1e9))
        after = self.to_world(sx, sy)
        self.center += anchor - after

    def pan(self, dx_px: float, dy_px: float) -> None:
        self.center -= complex(dx_px, -dy_px) / self.scale
