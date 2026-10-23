"""Чтение ESRI Shapefile (.shp + .dbf [+ .prj, .cpg]) без внешних библиотек.

Поддержаны Polygon/PolyLine (в т.ч. Z/M-варианты: Z/M-значения игнорируются).
Внешними считаются кольца, обходимые по часовой стрелке (спецификация ESRI);
дыры (против часовой) отбрасываются. Если по часовой нет ни одного кольца —
берутся все (нестандартные файлы).
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np

from .model import GeoLoadError, Region, clean_ring, identity_from_props, looks_geographic

_POLYGON = {5, 15, 25}
_LINE = {3, 13, 23}


def _signed_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_shp(path) -> list:
    """Возвращает по записи: список колец (np.ndarray) либо None для пустых/неподдерживаемых."""
    data = Path(path).read_bytes()
    if len(data) < 100 or struct.unpack(">i", data[:4])[0] != 9994:
        raise GeoLoadError("Не похоже на .shp (неверная сигнатура)")
    out, pos = [], 100
    while pos + 8 <= len(data):
        _, clen = struct.unpack(">2i", data[pos:pos + 8])
        pos += 8
        end = pos + clen * 2
        if clen < 0 or end > len(data):
            raise GeoLoadError("Файл .shp повреждён или обрезан")
        c, pos = data[pos:end], end
        stype = struct.unpack("<i", c[:4])[0] if len(c) >= 4 else 0
        if stype not in _POLYGON | _LINE:
            out.append(None)
            continue
        if len(c) < 44:
            raise GeoLoadError("Повреждена запись .shp")
        nparts, npts = struct.unpack("<2i", c[36:44])
        need = 44 + 4 * nparts + 16 * npts
        if nparts < 0 or npts < 0 or need > len(c):
            raise GeoLoadError("Повреждена запись .shp (размеры частей)")
        idx = np.frombuffer(c, "<i4", nparts, 44)
        pts = np.frombuffer(c, "<f8", 2 * npts, 44 + 4 * nparts).reshape(-1, 2).copy()
        bounds = list(idx) + [npts]
        rings = [pts[bounds[i]:bounds[i + 1]] for i in range(nparts)]
        if stype in _POLYGON:
            ext = [r for r in rings if len(r) >= 3 and _signed_area(r) < 0]
            rings = ext or [r for r in rings if len(r) >= 3 and _signed_area(r) != 0]
        out.append(rings)
    return out


def read_dbf(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    data = p.read_bytes()
    if len(data) < 32:
        raise GeoLoadError("Повреждён .dbf")
    nrec, hlen, rlen = struct.unpack("<IHH", data[4:12])
    cpg = p.with_suffix(".cpg")
    encs = [cpg.read_text().strip()] if cpg.exists() else []
    encs += ["utf-8", "cp1252"]
    fields, off = [], 32
    while off < hlen - 1 and data[off] != 0x0D:
        name = data[off:off + 11].split(b"\0")[0].decode("ascii", "ignore")
        fields.append((name, chr(data[off + 11]), data[off + 16]))
        off += 32
    rows = []
    for i in range(nrec):
        rec = data[hlen + i * rlen: hlen + (i + 1) * rlen]
        if len(rec) < rlen:
            break
        if rec[:1] == b"*":                                 # помечена как удалённая
            rows.append(None)
            continue
        row, o = {}, 1
        for name, ftype, flen in fields:
            raw = rec[o:o + flen]
            o += flen
            text = None
            for enc in encs:
                try:
                    text = raw.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            text = (text if text is not None else raw.decode("latin-1")).strip().strip("\0")
            if ftype in "NF":
                try:
                    text = float(text) if text else None
                except ValueError:
                    text = None
            row[name] = text
        rows.append(row)
    return rows


def load_shapefile(path, group: str = "Импорт") -> list[Region]:
    p = Path(path)
    shapes = read_shp(p)
    props = read_dbf(p.with_suffix(".dbf"))
    prj = p.with_suffix(".prj")
    projected = prj.exists() and "PROJCS" in prj.read_text(errors="ignore").upper()
    regions = []
    for n, rings in enumerate(shapes):
        row = props[n] if n < len(props) else {}
        if rings is None or row is None:                    # пустая или удалённая запись
            continue
        parts = [r for r in (clean_ring(x) for x in rings) if r is not None]
        if not parts:
            continue
        name, iso, aliases = identity_from_props(row)
        kind = "planar" if projected or not looks_geographic(parts) else "geographic"
        regions.append(Region(name or f"{p.stem} #{n + 1}", parts, kind=kind, iso=iso,
                              aliases=aliases, group=group, source=str(p)))
    if not regions:
        raise GeoLoadError("В shapefile нет полигонов или линий")
    return regions
