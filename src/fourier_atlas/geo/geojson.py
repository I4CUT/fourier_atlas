"""Чтение GeoJSON (RFC 7946) на чистом Python/NumPy. Используются внешние кольца полигонов."""
from __future__ import annotations
import json
from pathlib import Path

from .model import GeoLoadError, Region, clean_ring, identity_from_props, looks_geographic

MAX_BYTES = 400 * 1024 * 1024


def _geometry_parts(geom, out: list) -> None:
    if not isinstance(geom, dict):
        return
    t, c = geom.get("type"), geom.get("coordinates")
    try:
        if t == "Polygon" and c:
            out.append(c[0])                                  # только внешнее кольцо
        elif t == "MultiPolygon":
            out.extend(poly[0] for poly in c if poly)
        elif t == "LineString":
            out.append(c)
        elif t == "MultiLineString":
            out.extend(c)
        elif t == "GeometryCollection":
            for g in geom.get("geometries", []):
                _geometry_parts(g, out)
    except (TypeError, IndexError):
        raise GeoLoadError("Некорректная структура координат в GeoJSON")


def parse_geojson(obj, source: str = "", group: str = "Импорт", default_name: str = "Контур") -> list[Region]:
    if not isinstance(obj, dict) or "type" not in obj:
        raise GeoLoadError("Это не GeoJSON: нет поля 'type'")
    t = obj["type"]
    if t == "FeatureCollection":
        feats = obj.get("features") or []
    elif t == "Feature":
        feats = [obj]
    else:
        feats = [{"type": "Feature", "geometry": obj, "properties": {}}]
    regions = []
    for n, f in enumerate(feats, 1):
        if not isinstance(f, dict):
            continue
        raw: list = []
        _geometry_parts(f.get("geometry"), raw)
        parts = [r for r in (clean_ring(x) for x in raw) if r is not None]
        if not parts:
            continue
        name, iso, aliases = identity_from_props(f.get("properties"))
        regions.append(Region(name or f"{default_name} {n}", parts,
                              kind="geographic" if looks_geographic(parts) else "planar",
                              iso=iso, aliases=aliases, group=group, source=source))
    if not regions:
        raise GeoLoadError("В GeoJSON нет полигонов или линий")
    return regions


def load_geojson(path, group: str = "Импорт") -> list[Region]:
    p = Path(path)
    if p.stat().st_size > MAX_BYTES:
        raise GeoLoadError("Файл слишком большой")
    try:
        obj = json.loads(p.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise GeoLoadError(f"Не удалось разобрать GeoJSON: {e}")
    return parse_geojson(obj, source=str(p), group=group, default_name=p.stem)
