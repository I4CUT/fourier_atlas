"""Локальная база контуров стран (Natural Earth, GeoJSON/Shapefile).

Файл ищется в порядке: переменная FOURIER_ATLAS_COUNTRIES → каталог данных пакета
(data/countries) → ~/.fourier_atlas/countries. Предпочтение: 50m > 110m > остальное.
Скачать данные: python scripts/fetch_countries.py
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

from .loader import load_file
from .model import Region

log = logging.getLogger(__name__)
ENV_VAR = "FOURIER_ATLAS_COUNTRIES"
GROUP = "Страны"


def package_countries_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "countries"


def user_countries_dir() -> Path:
    return Path.home() / ".fourier_atlas" / "countries"


def _rank(p: Path) -> tuple:
    n = p.name.lower()
    return (0 if "50m" in n else 1 if "110m" in n else 2, n)


def find_countries_file() -> Path | None:
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env)
        if p.is_file():
            return p
        log.warning("%s указывает на несуществующий файл: %s", ENV_VAR, env)
    for d in (package_countries_dir(), user_countries_dir()):
        if d.is_dir():
            files = [f for f in d.iterdir() if f.suffix.lower() in (".geojson", ".json", ".shp")]
            if files:
                return sorted(files, key=_rank)[0]
    return None


def load_countries(path=None) -> list[Region]:
    p = Path(path) if path else find_countries_file()
    if p is None:
        return []
    regions = load_file(p, group=GROUP)
    log.info("База стран: %s (%d объектов)", p, len(regions))
    return regions
