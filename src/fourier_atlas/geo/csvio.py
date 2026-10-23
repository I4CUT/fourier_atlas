"""Импорт контура из CSV/TXT с координатами.

Правила: разделитель определяется автоматически (',', ';', табуляция, пробелы);
десятичная запятая допускается при разделителе ';'/табуляции/пробеле;
заголовок необязателен. Колонки: по именам (lon/lng/longitude/x и lat/latitude/y),
иначе первые две. Пустая строка начинает новую часть контура.
Данные считаются географическими ТОЛЬКО если в заголовке явно есть lon/lat.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

from .model import GeoLoadError, Region, clean_ring, looks_geographic

_LON = ("lon", "lng", "long", "longitude")
_LAT = ("lat", "latitude")
_X, _Y = ("x",), ("y",)


def _read_text(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise GeoLoadError("Не удалось определить кодировку CSV")


def _split(line: str, delim: str | None) -> list[str]:
    return [f.strip().strip('"') for f in (line.split(delim) if delim else line.split())]


def _num(s: str, decimal_comma: bool):
    try:
        return float(s.replace(",", ".") if decimal_comma else s)
    except ValueError:
        return None


def load_csv(path, group: str = "Импорт") -> list[Region]:
    p = Path(path)
    lines = _read_text(p).splitlines()
    first = next((l for l in lines if l.strip()), "")
    if not first:
        raise GeoLoadError("Пустой файл")
    counts = {d: first.count(d) for d in (";", "\t", ",")}
    delim = max(counts, key=counts.get) if max(counts.values()) > 0 else None
    decimal_comma = delim != ","

    head = _split(first, delim)
    has_header = any(_num(h, decimal_comma) is None for h in head[:2])
    names = [h.lower() for h in head]
    ix = iy = 0
    iy = 1
    geographic = False
    if has_header:
        lon = next((i for i, n in enumerate(names) if n in _LON), None)
        lat = next((i for i, n in enumerate(names) if n in _LAT), None)
        if lon is not None and lat is not None:
            ix, iy, geographic = lon, lat, True
        else:
            xi = next((i for i, n in enumerate(names) if n in _X), None)
            yi = next((i for i, n in enumerate(names) if n in _Y), None)
            if xi is not None and yi is not None:
                ix, iy = xi, yi

    parts, cur, bad = [], [], 0
    for line in lines[1 if has_header else 0:]:
        if not line.strip():
            if cur:
                parts.append(cur)
                cur = []
            continue
        f = _split(line, delim)
        if max(ix, iy) >= len(f):
            bad += 1
            continue
        x, y = _num(f[ix], decimal_comma), _num(f[iy], decimal_comma)
        if x is None or y is None:
            bad += 1
            continue
        cur.append((x, y))
    if cur:
        parts.append(cur)
    arrays = [r for r in (clean_ring(np.array(c)) for c in parts) if r is not None]
    if not arrays:
        raise GeoLoadError("В CSV не найдено ни одного контура (нужно >= 3 числовых точек)"
                           + (f"; пропущено некорректных строк: {bad}" if bad else ""))
    kind = "geographic" if geographic and looks_geographic(arrays) else "planar"
    return [Region(p.stem, arrays, kind=kind, group=group, source=str(p))]
