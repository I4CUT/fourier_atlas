"""Тесты экспорта: растр/SVG/GIF/MP4, отмена, ошибки, CSV."""
import csv
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest

from fourier_atlas.animation import AnimationController
from fourier_atlas.core import reconstruct
from fourier_atlas.data.samples import make_sample
from fourier_atlas.export import (ExportCancelled, ExportError, ExportJob, Style, build_primitives,
                                  export_coefficients_csv, export_contour_csv, fit_viewport, render_raster,
                                  render_svg, run_export, validate_job)


def tmp():
    return tempfile.TemporaryDirectory()


def job(kind, d, name=None, **kw):
    z, spec = make_sample("Сердце", 512)
    base = dict(kind=kind, path=str(Path(d) / (name or f"out.{kind}")), spectrum=spec.copy(), target=z,
                max_epicycles=30, width=320, height=240, fps=10, duration=1.0, supersample=2)
    base.update(kw)
    return ExportJob(**base)


def scene(style=None):
    z, spec = make_sample("Звезда", 512)
    c = AnimationController(spec, max_epicycles=12, path_samples=512)
    c.seek(0.3)
    st = style or Style()
    v = fit_viewport(400, 300, z, c.path, st.margin)
    return build_primitives(c.frame(), v, z, st), st, c


# ---------- примитивы ----------
def test_primitives_structure_and_flags():
    prims, st, c = scene()
    kinds = [p.kind for p in prims]
    assert kinds.count("circle") >= 5 and kinds.count("dot") == 1 and kinds[-1] == "dot"
    assert sum(1 for p in prims if p.kind == "polyline") >= 1 + 1 + 6      # контур + векторы + след
    none, _, _ = scene(Style(show_circles=False, show_vectors=False, show_target=False))
    assert [p.kind for p in none].count("circle") == 0
    assert all(p.kind != "polyline" or p.color[:3] == (255, 190, 60) for p in none)   # только след


def test_tip_dot_matches_frame_tip():
    prims, st, c = scene()
    z, _ = make_sample("Звезда", 512)
    view = fit_viewport(400, 300, z, c.path, st.margin)
    sx, sy = view.to_screen(c.frame().joints[-1])
    dot = prims[-1]
    assert abs(dot.center[0] - sx) < 1e-9 and abs(dot.center[1] - sy) < 1e-9


# ---------- растр ----------
def test_raster_size_background_and_tip_pixel():
    prims, st, c = scene()
    img = render_raster(prims, (400, 300), st.background, supersample=2)
    assert img.size == (400, 300) and img.mode == "RGB"
    assert img.getpixel((2, 2)) == st.background
    cx, cy = (int(round(v)) for v in prims[-1].center)
    r, g, b = img.getpixel((cx, cy))
    assert r > 200 and g > 150 and b < 120                      # оранжевая точка на конце цепочки
    arr = np.asarray(img)
    assert (arr != np.array(st.background)).any(axis=2).mean() > 0.005


def test_raster_supersample_changes_antialiasing_only():
    prims, st, _ = scene()
    a = np.asarray(render_raster(prims, (200, 150), st.background, 1)).astype(int)
    b = np.asarray(render_raster(prims, (200, 150), st.background, 3)).astype(int)
    assert a.shape == b.shape and np.abs(a - b).mean() < 12


# ---------- SVG ----------
def test_svg_is_valid_xml_with_expected_elements():
    prims, st, _ = scene()
    svg = render_svg(prims, (400, 300), st.background)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert root.tag == ns + "svg" and root.get("viewBox") == "0 0 400 300"
    assert len(root.findall(ns + "circle")) == sum(1 for p in prims if p.kind in ("circle", "dot"))
    assert len(root.findall(ns + "polyline")) == sum(1 for p in prims if p.kind == "polyline")
    assert root.find(ns + "rect").get("fill") == "rgb(18,20,28)"
    dot = root.findall(ns + "circle")[-1]
    assert abs(float(dot.get("cx")) - prims[-1].center[0]) < 0.01


# ---------- задания ----------
def test_png_job_and_atomic_no_leftovers():
    from PIL import Image
    with tmp() as d:
        seen = []
        p = run_export(job("png", d, phase=0.4), progress=lambda i, n: seen.append((i, n)))
        im = Image.open(p)
        assert im.size == (320, 240) and im.format == "PNG" and seen == [(1, 1)]
        assert sorted(os.listdir(d)) == ["out.png"]


def test_svg_job_writes_parseable_file():
    with tmp() as d:
        p = run_export(job("svg", d, phase=0.1, width=500, height=500))
        root = ET.parse(p).getroot()
        assert root.get("width") == "500"


def test_gif_job_frames_loop_and_progress():
    from PIL import Image
    with tmp() as d:
        calls = []
        p = run_export(job("gif", d, fps=8, duration=1.0, width=160, height=120, supersample=1),
                       progress=lambda i, n: calls.append((i, n)))
        im = Image.open(p)
        assert im.format == "GIF" and im.n_frames == 8 and im.size == (160, 120)
        assert im.info.get("loop") == 0
        assert calls[0] == (1, 8) and calls[-1] == (8, 8)


def test_gif_frames_differ_animation_not_static():
    from PIL import Image
    with tmp() as d:
        p = run_export(job("gif", d, fps=6, duration=1.0, width=160, height=120, supersample=1))
        im = Image.open(p)
        im.seek(0); a = np.asarray(im.convert("RGB")).astype(int)
        im.seek(3); b = np.asarray(im.convert("RGB")).astype(int)
        assert np.abs(a - b).sum() > 0


def test_mp4_job_valid_video_even_size():
    import imageio.v2 as imageio
    with tmp() as d:
        p = run_export(job("mp4", d, fps=10, duration=1.0, width=321, height=241, supersample=1))
        assert p.stat().st_size > 1000
        r = imageio.get_reader(str(p), format="FFMPEG")
        meta = r.get_meta_data()
        assert meta["size"] == (320, 240) and abs(meta["fps"] - 10) < 0.5
        frame = r.get_data(0)
        assert frame.shape == (240, 320, 3)
        r.close()
        assert sorted(os.listdir(d)) == ["out.mp4"]


def test_cancel_removes_partial_file_and_raises():
    with tmp() as d:
        n = {"i": 0}

        def prog(i, total):
            n["i"] = i

        with pytest.raises(ExportCancelled):
            run_export(job("gif", d, fps=10, duration=2.0, width=120, height=90, supersample=1),
                       progress=prog, cancel=lambda: n["i"] >= 3)
        assert os.listdir(d) == []


def test_job_uses_isolated_spectrum_copy():
    z, spec = make_sample("Сердце", 512)
    snap = spec.copy()
    spec.set_coeff(1, 5.0)                        # правка «живого» спектра после копирования
    assert abs(snap.coeff(1) - 5.0) > 0.01
    snap.set_enabled(2, False)
    assert spec.enabled[spec.index_of(2)]


def test_validation_errors():
    with tmp() as d:
        for kw in (dict(width=8), dict(height=99999), dict(supersample=9)):
            with pytest.raises(ExportError):
                validate_job(job("png", d, **kw))
        bad = job("png", d)
        bad.kind = "bmp"
        with pytest.raises(ExportError):
            validate_job(bad)
        with pytest.raises(ExportError):
            validate_job(job("gif", d, fps=0))
        with pytest.raises(ExportError):
            validate_job(job("mp4", d, duration=1000))
        with pytest.raises(ExportError):
            validate_job(job("gif", d, width=3000, height=100))
        with pytest.raises(ExportError):
            validate_job(job("mp4", d, fps=120, duration=600))      # слишком много кадров


def test_unwritable_path_gives_export_error():
    with tmp() as d:
        blocker = Path(d) / "file"
        blocker.write_text("x")
        with pytest.raises(ExportError):
            run_export(job("png", d, name="file/inner.png"))        # «каталог» — это файл


def test_memory_streaming_many_frames_constant():
    import tracemalloc
    with tmp() as d:
        tracemalloc.start()
        run_export(job("mp4", d, fps=20, duration=3.0, width=240, height=160, supersample=1))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 40 * 1024 * 1024                              # 60 кадров не копятся в памяти


# ---------- CSV ----------
def test_coefficients_csv_roundtrip():
    z, spec = make_sample("Сердце", 256)
    spec.set_coeff(1, 0.25 - 0.5j)
    spec.set_enabled(-2, False)
    with tmp() as d:
        p = export_coefficients_csv(Path(d) / "c.csv", spec)
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        assert len(rows) == 256 and [int(r["k"]) for r in rows] == sorted(spec.freqs.tolist())
        r1 = next(r for r in rows if r["k"] == "1")
        assert complex(float(r1["re"]), float(r1["im"])) == 0.25 - 0.5j and r1["edited"] == "1"
        assert abs(float(r1["amplitude"]) - abs(0.25 - 0.5j)) < 1e-15
        o = spec.original[spec.index_of(1)]
        assert float(r1["original_re"]) == o.real and float(r1["original_im"]) == o.imag
        assert next(r for r in rows if r["k"] == "-2")["enabled"] == "0"
        assert os.listdir(d) == ["c.csv"]


def test_contour_csv_values_and_reimport_as_geodata():
    from fourier_atlas.geo import load_file
    z, spec = make_sample("Сердце", 128)
    spec.select_band(5)
    with tmp() as d:
        p = export_contour_csv(Path(d) / "contour.csv", z, spec)
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        rec = reconstruct(spec, 128)
        assert len(rows) == 128
        assert abs(float(rows[7]["x_reconstructed"]) - rec[7].real) < 1e-15
        assert abs(float(rows[7]["error"]) - abs(z[7] - rec[7])) < 1e-15
        # наш же CSV-импортёр читает первые две колонки (index,t) — проверяем, что экспорт не ломает загрузчик
        regs = load_file(p)
        assert len(regs) == 1
