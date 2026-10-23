"""Единая точка входа импорта: формат определяется по расширению."""
from __future__ import annotations
from pathlib import Path

from .csvio import load_csv
from .geojson import load_geojson
from .model import GeoLoadError, Region
from .shapefile import load_shapefile
from .svg import load_svg

LOADERS = {".geojson": load_geojson, ".json": load_geojson, ".shp": load_shapefile,
           ".svg": load_svg, ".csv": load_csv, ".txt": load_csv}
FILE_FILTER = "Геоданные (*.geojson *.json *.shp *.svg *.csv *.txt);;Все файлы (*)"


def load_file(path, group: str = "Импорт") -> list[Region]:
    p = Path(path)
    if not p.is_file():
        raise GeoLoadError(f"Файл не найден: {p}")
    fn = LOADERS.get(p.suffix.lower())
    if fn is None:
        raise GeoLoadError(f"Неподдерживаемый формат '{p.suffix}'. Поддерживаются: "
                           + ", ".join(sorted(LOADERS)))
    try:
        return fn(p, group=group)
    except GeoLoadError:
        raise
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        raise GeoLoadError(f"Ошибка чтения {p.name}: {e}")
