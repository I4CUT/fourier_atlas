"""Модель географических данных: регион = имя + набор внешних контуров (частей)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


class GeoLoadError(Exception):
    """Ошибка чтения/подготовки геоданных (текст предназначен для пользователя)."""


@dataclass
class Region:
    name: str
    parts: list                      # list[np.ndarray (M, 2)] — внешние контуры
    kind: str = "geographic"         # "geographic" (lon/lat, градусы) | "planar"
    y_down: bool = False             # True для SVG (ось Y вниз)
    iso: str = ""
    aliases: tuple = ()
    group: str = "Импорт"
    source: str = ""


NAME_KEYS = ("NAME_RU", "name_ru", "NAME_EN", "NAME", "ADMIN", "name", "Name", "NAME_LONG",
             "COUNTRY", "country", "NAME_0", "admin")
ALIAS_KEYS = ("NAME", "NAME_EN", "NAME_RU", "NAME_LONG", "ADMIN", "SOVEREIGNT", "FORMAL_EN",
              "ABBREV", "name", "Name", "country", "COUNTRY", "NAME_0", "admin")
ISO_KEYS = ("ISO_A3_EH", "ADM0_A3", "ISO_A3", "iso_a3", "ISO3", "GID_0", "ISO", "iso")


# Единые названия для интерфейса: источники данных (Natural Earth и др.) могут давать
# устаревшие или иные формы. Прежнее название остаётся в aliases, чтобы поиск его находил.
NAME_OVERRIDES = {"Белоруссия": "Беларусь"}


def _display(name: str) -> str:
    return NAME_OVERRIDES.get(name, name)


def identity_from_props(props: dict | None):
    """(name, iso, aliases) из атрибутов объекта; '-99' (Natural Earth) считается пустым."""
    props = props or {}

    def val(k):
        v = props.get(k)
        return v.strip() if isinstance(v, str) and v.strip() and v.strip() != "-99" else ""

    name = next((val(k) for k in NAME_KEYS if val(k)), "")
    iso = next((val(k) for k in ISO_KEYS if val(k)), "")
    raw_name = name
    name = _display(name)
    aliases = [a for a in (val(k) for k in ALIAS_KEYS) if a]
    if raw_name != name:
        aliases.append(raw_name)                          # старое название — только для поиска
    aliases = tuple(dict.fromkeys(a for a in aliases if a != name))
    return name, iso, aliases


def clean_ring(arr) -> np.ndarray | None:
    """Приводит к (M, 2) float, отбрасывает Z/M, нефинитные значения; None, если < 3 точек."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        return None
    a = a[:, :2]
    a = a[np.all(np.isfinite(a), axis=1)]
    return a if len(a) >= 3 else None


def looks_geographic(parts) -> bool:
    return bool(parts) and all(np.all(np.abs(p[:, 0]) <= 180.0001) and np.all(np.abs(p[:, 1]) <= 90.0001)
                               for p in parts)
