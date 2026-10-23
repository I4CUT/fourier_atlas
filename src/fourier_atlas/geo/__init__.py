"""Загрузка и подготовка геометрии. Не зависит от Qt."""
from .model import Region, GeoLoadError
from .loader import load_file, LOADERS, FILE_FILTER
from .prepare import PrepareOptions, PreparedContour, prepare_region, part_count, ranked_parts
from .database import find_countries_file, load_countries

__all__ = [n for n in dir() if not n.startswith("_")]
