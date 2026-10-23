"""Тесты загрузки геоданных: GeoJSON, Shapefile, SVG, CSV, подготовка контура, каталог."""
import json
import os
import struct
import tempfile
from pathlib import Path
import numpy as np
import pytest

from fourier_atlas.core import compute_spectrum, reconstruct, resample_arclength, to_complex, rmse
from fourier_atlas.data.catalog import Catalog
from fourier_atlas.data.samples import contour_to_spectrum
from fourier_atlas.geo import (GeoLoadError, PrepareOptions, load_file, part_count,
                               prepare_region, find_countries_file)
from fourier_atlas.geo.model import Region
from fourier_atlas.geo.prepare import _shoelace
from fourier_atlas.geo.svg import parse_path, parse_transform


def tmp():
    return tempfile.TemporaryDirectory()


def write(d, name, text, enc="utf-8"):
    p = Path(d) / name
    p.write_bytes(text.encode(enc) if isinstance(text, str) else text)
    return p


def sq(x0, y0, s):                      # квадрат против часовой, замкнутый
    return [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0]]


# ======================= GeoJSON =======================
def test_geojson_countries_multipolygon_names_iso():
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"NAME": "Italy", "NAME_RU": "Италия", "ISO_A3": "-99",
                                           "ISO_A3_EH": "ITA", "ADMIN": "Italy"},
         "geometry": {"type": "MultiPolygon", "coordinates": [
             [sq(10, 40, 5), sq(11, 41, 1)],                     # материк (с дырой)
             [sq(8, 39, 1)]]}},                                  # остров
        {"type": "Feature", "properties": {"NAME": "Point"}, "geometry": {"type": "Point", "coordinates": [1, 2]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [sq(0, 0, 1)]}}]}
    with tmp() as d:
        regs = load_file(write(d, "c.geojson", json.dumps(fc)))
    assert len(regs) == 2                                        # точка пропущена
    it = regs[0]
    assert it.name == "Италия" and it.iso == "ITA" and "Italy" in it.aliases
    assert len(it.parts) == 2 and it.kind == "geographic"        # дыра отброшена
    assert regs[1].name.startswith("c ")                         # имя по умолчанию


def test_geojson_bare_geometry_z_coords_and_planar_detection():
    g = {"type": "Polygon", "coordinates": [[[0, 0, 5], [1000, 0, 5], [1000, 1000, 5], [0, 1000, 5]]]}
    with tmp() as d:
        r, = load_file(write(d, "g.json", json.dumps(g)))
    assert r.parts[0].shape == (4, 2) and r.kind == "planar"


def test_geojson_errors():
    with tmp() as d:
        with pytest.raises(GeoLoadError):
            load_file(write(d, "bad.geojson", "{not json"))
        with pytest.raises(GeoLoadError):
            load_file(write(d, "x.geojson", json.dumps({"hello": 1})))
        with pytest.raises(GeoLoadError):
            load_file(write(d, "e.geojson", json.dumps({"type": "FeatureCollection", "features": []})))
        with pytest.raises(GeoLoadError):
            load_file(write(d, "t.xyz", "1"))
        with pytest.raises(GeoLoadError):
            load_file(Path(d) / "missing.geojson")


# ======================= Shapefile =======================
def make_shp(records):
    """records: список списков колец; пишет корректный Polygon .shp (type 5)."""
    body = b""
    for n, rings in enumerate(records, 1):
        pts = np.vstack(rings)
        starts, k = [], 0
        for r in rings:
            starts.append(k); k += len(r)
        content = struct.pack("<i4d2i", 5, *pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(),
                              len(rings), len(pts)) if False else (
            struct.pack("<i", 5) + struct.pack("<4d", pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            + struct.pack("<2i", len(rings), len(pts)) + struct.pack(f"<{len(rings)}i", *starts)
            + pts.astype("<f8").tobytes())
        body += struct.pack(">2i", n, len(content) // 2) + content
    header = struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(body)) // 2) \
        + struct.pack("<2i", 1000, 5) + b"\0" * 64
    return header + body


def make_dbf(rows, fields, enc="utf-8"):
    hlen = 32 + 32 * len(fields) + 1
    rlen = 1 + sum(f[2] for f in fields)
    h = struct.pack("<BBBBIHH", 3, 24, 1, 1, len(rows), hlen, rlen) + b"\0" * 20
    for name, t, ln in fields:
        h += name.encode().ljust(11, b"\0") + t.encode() + b"\0" * 4 + bytes([ln, 0]) + b"\0" * 14
    h += b"\r"
    body = b""
    for row in rows:
        body += b" " if row is not None else b"*"
        row = row or [""] * len(fields)
        for v, (_, t, ln) in zip(row, fields):
            body += str(v).encode(enc).ljust(ln, b" ")
    return h + body


def cw(r):          # по часовой (внешнее кольцо shapefile)
    return np.array(r[::-1], float)


def test_shapefile_polygons_holes_attrs_cyrillic():
    outer = cw(sq(10, 40, 5))
    hole = np.array(sq(11, 41, 1), float)                       # против часовой — дыра
    island = cw(sq(0, 0, 2))
    shp = make_shp([[outer, hole, island], [cw(sq(50, 50, 3))]])
    dbf = make_dbf([["Италия", "ITA", 5.0], ["Две", "TWO", 7.0]],
                   [("NAME", "C", 20), ("ISO_A3", "C", 3), ("AREA", "N", 8)], enc="cp1251")
    with tmp() as d:
        write(d, "w.shp", shp); write(d, "w.dbf", dbf); write(d, "w.cpg", "cp1251")
        regs = load_file(Path(d) / "w.shp")
    assert [r.name for r in regs] == ["Италия", "Две"]
    assert len(regs[0].parts) == 2 and regs[0].iso == "ITA"      # дыра отброшена
    assert regs[0].kind == "geographic"


def test_shapefile_without_dbf_deleted_records_prj_and_errors():
    shp = make_shp([[cw(sq(0, 0, 1))], [cw(sq(5, 5, 1))]])
    dbf = make_dbf([None, ["B"]], [("NAME", "C", 5)])
    with tmp() as d:
        write(d, "a.shp", shp)
        regs = load_file(Path(d) / "a.shp")                     # без .dbf
        assert [r.name for r in regs] == ["a #1", "a #2"]
        write(d, "a.dbf", dbf); write(d, "a.prj", 'PROJCS["WGS 84 / UTM"]')
        regs = load_file(Path(d) / "a.shp")
        assert [r.name for r in regs] == ["B"] and regs[0].kind == "planar"   # удалённая пропущена
        write(d, "t.shp", shp[:150])
        with pytest.raises(GeoLoadError):
            load_file(Path(d) / "t.shp")                         # обрезан
        write(d, "n.shp", b"x" * 200)
        with pytest.raises(GeoLoadError):
            load_file(Path(d) / "n.shp")                         # не shp


# ======================= SVG =======================
def test_svg_path_commands_relative_implicit_close():
    subs = parse_path("M10 10 h20 v20 h-20 z m100 0 l10 0 10 10 z")
    assert subs[0] == [(10, 10), (30, 10), (30, 30), (10, 30)]
    assert subs[1][:3] == [(110, 10), (120, 10), (130, 20)]     # неявный l; z вернул в начало подпути


def test_svg_arc_circle_and_compact_flags():
    # круг радиуса 10 двумя дугами; флаги слитно (01)
    pts = parse_path("M 20 10 a10 10 0 1 1 -20 0 a10 10 0 1 1 20 0")[0]
    r = np.hypot(np.array([p[0] for p in pts]) - 10, np.array([p[1] for p in pts]) - 10)
    assert np.allclose(r, 10, atol=1e-6)
    pts2 = parse_path("M0 0 a10 10 0 01 20 0")[0]               # флаги "0","1" без пробела
    assert pts2[-1] == (20.0, 0.0) and len(pts2) > 3


def test_svg_beziers_endpoint_and_smooth_reflection():
    c = parse_path("M0 0 C 0 10 10 10 10 0 S 20 -10 20 0", steps=10)[0]
    assert c[10] == pytest.approx((10, 0)) and c[-1] == pytest.approx((20, 0))
    q = parse_path("M0 0 Q 5 10 10 0 T 20 0", steps=10)[0]
    assert q[-1] == pytest.approx((20, 0)) and max(p[1] for p in q) > 0 > min(p[1] for p in q)


def test_svg_transforms_shapes_y_down_and_security():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg"><g transform="translate(100 0) scale(2)">'
           '<rect x="0" y="0" width="10" height="5"/><circle cx="50" cy="50" r="5"/></g>'
           '<defs><rect width="999" height="999"/></defs>'
           '<polygon points="0,0 4,0 4,4 0,4"/></svg>')
    with tmp() as d:
        r, = load_file(write(d, "s.svg", svg))
        assert r.y_down and r.kind == "planar" and len(r.parts) == 3   # defs пропущен
        rect = r.parts[0]
        assert rect[:, 0].min() == 100 and rect[:, 0].max() == 120 and rect[:, 1].max() == 10
        with pytest.raises(GeoLoadError):
            load_file(write(d, "e.svg", '<!DOCTYPE x [<!ENTITY a "b">]><svg/>'))
        with pytest.raises(GeoLoadError):
            load_file(write(d, "b.svg", "<svg><"))
        with pytest.raises(GeoLoadError):
            load_file(write(d, "n.svg", '<svg xmlns="http://www.w3.org/2000/svg"><text>hi</text></svg>'))
    m = parse_transform("translate(5,6) rotate(90)")
    assert np.allclose(m @ [1, 0, 1], [5, 7, 1])


# ======================= CSV =======================
def test_csv_variants():
    with tmp() as d:
        r, = load_file(write(d, "a.csv", "lon,lat\n10,40\n12,40\n12,42\n10,42\n"))
        assert r.kind == "geographic" and r.parts[0].shape == (4, 2)
        r, = load_file(write(d, "b.csv", "x;y\n0,5;0\n1,5;0\n1,5;2,5\n0;2\n"))      # десятичная запятая
        assert r.kind == "planar" and r.parts[0][1, 0] == 1.5
        r, = load_file(write(d, "c.txt", "0 0\n1 0\n1 1\n\n5 5\n6 5\n6 6\n"))         # пробелы, 2 части
        assert len(r.parts) == 2
        r, = load_file(write(d, "d.csv", "lat,lon\n40,10\n40,12\n42,12\n"))          # порядок колонок по именам
        assert r.parts[0][0].tolist() == [10, 40]
        r, = load_file(write(d, "e.csv", "0,0\n1,0\nabc,def\n1,1\nnan,1\n"))          # мусор пропущен
        assert len(r.parts[0]) == 3
        with pytest.raises(GeoLoadError):
            load_file(write(d, "f.csv", "x,y\n1,2\n3,4\n"))
        with pytest.raises(GeoLoadError):
            load_file(write(d, "g.csv", ""))


# ======================= подготовка =======================
def test_prepare_antimeridian_equirect_scaling_and_orientation():
    # квадрат 10°×10° на широте 60, пересекает 180°, задан по часовой
    ring = np.array([[175, 55], [175, 65], [-175, 65], [-175, 55], [175, 55]], float)
    reg = Region("X", [ring])
    p = prepare_region(reg)
    w = np.ptp(p.xy[:, 0]); h = np.ptp(p.xy[:, 1])
    assert abs(h - 10) < 1e-9 and abs(w - 10 * np.cos(np.deg2rad(60))) < 1e-9   # разрыва нет
    assert _shoelace(p.xy) > 0                                                  # против часовой
    plate = prepare_region(reg, PrepareOptions(projection="plate_carree"))
    assert abs(np.ptp(plate.xy[:, 0]) - 10) < 1e-9 and plate.projection == "plate_carree"


def test_prepare_pole_ring_closed_through_pole():
    lon = np.linspace(-180, 180, 37)
    ring = np.column_stack([lon, np.full(37, -70.0)])
    p = prepare_region(Region("Ant", [ring]))
    assert p.xy[:, 1].min() == -90 and _shoelace(p.xy) > 0


def test_prepare_part_ranking_planar_y_down_and_errors():
    big, small = np.array(sq(0, 0, 10), float), np.array(sq(20, 20, 1), float)
    reg = Region("R", [small, big], kind="planar")
    assert part_count(reg) == 2
    assert np.ptp(prepare_region(reg).xy[:, 0]) == 10
    assert np.ptp(prepare_region(reg, PrepareOptions(part=2)).xy[:, 0]) == 1
    with pytest.raises(ValueError):
        prepare_region(reg, PrepareOptions(part=3))
    with pytest.raises(ValueError):
        prepare_region(reg, PrepareOptions(projection="mercator"))
    up = Region("U", [np.array([[0, 0], [4, 0], [4, 2], [0, 2]], float)], kind="planar", y_down=True)
    xy = prepare_region(up).xy
    assert xy[:, 1].max() == 0 and xy[:, 1].min() == -2 and _shoelace(xy) > 0
    with pytest.raises(GeoLoadError):
        prepare_region(Region("Z", [np.zeros((3, 2))], kind="planar"))


def test_end_to_end_geojson_to_spectrum_reconstruction():
    # гексагон на широте 45° → контур → спектр → при полном наборе совпадает
    a = np.linspace(0, 2 * np.pi, 7)[:-1]
    ring = np.column_stack([20 + 5 * np.cos(a), 45 + 5 * np.sin(a)])
    gj = {"type": "Feature", "properties": {"NAME": "Hex"},
          "geometry": {"type": "Polygon", "coordinates": [ring.tolist() + [ring[0].tolist()]]}}
    with tmp() as d:
        r, = load_file(write(d, "h.geojson", json.dumps(gj)))
    prep = prepare_region(r)
    z, spec = contour_to_spectrum(prep.xy, 512)
    assert np.allclose(reconstruct(spec), z, atol=1e-10)
    assert abs(np.abs(z).max() - 1) < 1e-12
    spec.select_band(6)
    assert rmse(z, reconstruct(spec)) < 0.2                      # шестиугольник: низкие гармоники дают форму


# ======================= каталог и база =======================
def test_catalog_defaults_search_default_name_and_unique_keys():
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"NAME": "Italy", "NAME_RU": "Италия", "ISO_A3_EH": "ITA"},
         "geometry": {"type": "Polygon", "coordinates": [sq(10, 40, 5)]}},
        {"type": "Feature", "properties": {"NAME": "France", "ISO_A3_EH": "FRA"},
         "geometry": {"type": "Polygon", "coordinates": [sq(0, 45, 5)]}}]}
    with tmp() as d:
        f = write(d, "ne_110m_admin_0_countries.geojson", json.dumps(fc))
        cat = Catalog.from_defaults(f)
        assert cat.warning == ""
        names = cat.names()
        assert names[:2] == ["France", "Италия"] or names[:2] == ["Италия", "France"]
        assert cat.names()[-1] in ("Эллипс", "Сердце", "Звезда", "Квадрат", "Окружность")
        assert cat.default_name() == "Италия"
        assert cat.search("ita") == ["Италия"] and cat.search("fra") == ["France"]
        k1 = cat.add(Region("France", [np.array(sq(0, 0, 1), float)]))
        assert k1 == "France (2)"
    cat2 = Catalog.from_defaults("/nonexistent/path.geojson") if False else None


def test_catalog_without_database_has_demos_and_warning():
    old = os.environ.get("FOURIER_ATLAS_COUNTRIES")
    os.environ["FOURIER_ATLAS_COUNTRIES"] = ""
    try:
        cat = Catalog.from_defaults()
    finally:
        if old is None:
            os.environ.pop("FOURIER_ATLAS_COUNTRIES", None)
        else:
            os.environ["FOURIER_ATLAS_COUNTRIES"] = old
    if find_countries_file() is None:
        assert cat.warning and len(cat.names()) == 5
    assert cat.default_name() in cat.names()


def test_geo_package_has_no_qt():
    import subprocess, sys
    code = ("import sys, fourier_atlas.geo, fourier_atlas.data.catalog; "
            "sys.exit(1 if any(m.startswith(('PySide','PyQt')) for m in sys.modules) else 0)")
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    assert subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": src}).returncode == 0


def test_belarus_display_name_override():
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"NAME": "Belarus", "NAME_RU": "Белоруссия", "ISO_A3_EH": "BLR"},
         "geometry": {"type": "Polygon", "coordinates": [sq(24, 51, 5)]}}]}
    with tmp() as d:
        f = write(d, "ne_110m_admin_0_countries.geojson", json.dumps(fc))
        cat = Catalog.from_defaults(f)
    assert "Беларусь" in cat.names() and "Белоруссия" not in cat.names()
    r = cat.get("Беларусь")
    assert r.name == "Беларусь" and r.iso == "BLR"
    assert cat.search("беларусь") == ["Беларусь"]
    assert cat.search("белоруссия") == ["Беларусь"]          # старое название находит новое
    assert cat.search("belarus") == ["Беларусь"]
    assert r.name != "Белоруссия"
