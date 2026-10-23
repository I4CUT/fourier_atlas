"""Каталог источников контуров: демо-фигуры + база стран + импортированные файлы."""
from __future__ import annotations
import logging

from ..geo.database import load_countries
from ..geo.model import GeoLoadError, Region
from .samples import SAMPLES

log = logging.getLogger(__name__)
DEMO_GROUP, COUNTRY_GROUP, IMPORT_GROUP = "Демо-фигуры", "Страны", "Импорт"
_ORDER = {COUNTRY_GROUP: 0, DEMO_GROUP: 1, IMPORT_GROUP: 2}


class Catalog:
    def __init__(self):
        self._items: dict[str, Region] = {}
        self.warning: str = ""

    def add(self, region: Region) -> str:
        """Добавляет регион; при совпадении имени добавляет суффикс (2), (3)… Возвращает ключ."""
        key, n = region.name, 2
        while key in self._items:
            key = f"{region.name} ({n})"
            n += 1
        self._items[key] = region
        return key

    def get(self, key: str) -> Region:
        return self._items[key]

    def names(self) -> list[str]:
        def k(item):
            name, r = item
            return (_ORDER.get(r.group, 9), name.casefold() if r.group == COUNTRY_GROUP else 0)
        return [n for n, _ in sorted(self._items.items(), key=k)]

    def search(self, text: str) -> list[str]:
        t = text.strip().casefold()
        if not t:
            return self.names()
        hits = []
        for name in self.names():
            r = self._items[name]
            if any(t in s.casefold() for s in (name, r.iso, *r.aliases)):
                hits.append(name)
        return hits

    def default_name(self) -> str:
        for name in self.names():
            r = self._items[name]
            if r.group == COUNTRY_GROUP and "italy" in {s.casefold() for s in (name, *r.aliases)}:
                return name
        return self.names()[0]

    @classmethod
    def from_defaults(cls, countries_path=None) -> "Catalog":
        cat = cls()
        for name, fn in SAMPLES.items():
            cat.add(Region(name, [fn()], kind="planar", group=DEMO_GROUP))
        try:
            for r in load_countries(countries_path):
                r.group = COUNTRY_GROUP
                cat.add(r)
        except GeoLoadError as e:
            log.error("База стран не загружена: %s", e)
            cat.warning = f"База стран не загружена: {e}"
        if not any(r.group == COUNTRY_GROUP for r in cat._items.values()) and not cat.warning:
            cat.warning = ("База стран не найдена. Выполните scripts/fetch_countries.py "
                           "или импортируйте GeoJSON/Shapefile/SVG/CSV.")
        return cat
