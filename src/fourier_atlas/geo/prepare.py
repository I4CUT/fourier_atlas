"""Подготовка контура региона: выбор части, проекция, ориентация.

Результат — плоский контур (M, 2), ось Y вверх, обход против часовой стрелки,
готовый для resample_arclength → FFT.

Проекции для географических данных (градусы):
  equirectangular — x = λ·cos φ0, y = φ, φ0 — широта центра bbox выбранной части
                    (локально сохраняет пропорции, без искажений Меркатора);
  plate_carree    — x = λ, y = φ (как есть).
Долготы разворачиваются (unwrap), поэтому контуры через антимеридиан (Россия, Фиджи)
не разрываются. Кольцо, охватывающее полюс (Антарктида), замыкается по полюсу.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .model import GeoLoadError, Region

PROJECTIONS = ("auto", "equirectangular", "plate_carree")


@dataclass(frozen=True)
class PrepareOptions:
    part: int = 1                 # ранг части по площади: 1 — самая большая
    projection: str = "auto"


@dataclass
class PreparedContour:
    xy: np.ndarray
    part_rank: int
    n_parts: int
    projection: str               # фактически применённая: equirectangular/plate_carree/none
    raw_points: int


def _shoelace(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _unwrap_lon(lon: np.ndarray) -> np.ndarray:
    return np.unwrap(lon, period=360.0)


def _est_area(part: np.ndarray, kind: str) -> float:
    if kind != "geographic":
        return abs(_shoelace(part))
    lon = _unwrap_lon(part[:, 0])
    lat = part[:, 1]
    c = np.cos(np.deg2rad((lat.min() + lat.max()) / 2))
    return abs(_shoelace(np.column_stack([lon * c, lat])))


def ranked_parts(region: Region) -> list[int]:
    """Индексы пригодных частей по убыванию площади."""
    scored = []
    for i, p in enumerate(region.parts):
        if len(p) >= 3 and np.all(np.isfinite(p)):
            a = _est_area(p, region.kind)
            if a > 0:
                scored.append((a, i))
    return [i for _, i in sorted(scored, key=lambda t: -t[0])]


def part_count(region: Region) -> int:
    return len(ranked_parts(region))


def project_geographic(ring: np.ndarray, projection: str) -> np.ndarray:
    lon = _unwrap_lon(ring[:, 0])
    lat = ring[:, 1].copy()
    if abs(lon[-1] - lon[0]) > 180:                       # кольцо охватывает полюс
        pole = 90.0 if lat.mean() > 0 else -90.0
        lon = np.concatenate([lon, [lon[-1], lon[0]]])
        lat = np.concatenate([lat, [pole, pole]])
    k = np.cos(np.deg2rad((lat.min() + lat.max()) / 2)) if projection == "equirectangular" else 1.0
    return np.column_stack([lon * max(k, 1e-6), lat])


def prepare_region(region: Region, options: PrepareOptions | None = None) -> PreparedContour:
    opt = options or PrepareOptions()
    if opt.projection not in PROJECTIONS:
        raise ValueError(f"Неизвестная проекция: {opt.projection}")
    ranks = ranked_parts(region)
    if not ranks:
        raise GeoLoadError(f"У региона «{region.name}» нет пригодных контуров")
    if not 1 <= opt.part <= len(ranks):
        raise ValueError(f"Часть {opt.part} вне диапазона 1..{len(ranks)}")
    ring = np.asarray(region.parts[ranks[opt.part - 1]], float)

    if region.kind == "geographic" and opt.projection != "none":
        proj = "equirectangular" if opt.projection == "auto" else opt.projection
        xy = project_geographic(ring, proj)
    else:
        proj, xy = "none", ring.copy()
    if region.y_down:
        xy[:, 1] = -xy[:, 1]
    if _shoelace(xy) < 0:                                 # единый обход: против часовой
        xy = xy[::-1].copy()
    return PreparedContour(xy, opt.part, len(ranks), proj, len(ring))
