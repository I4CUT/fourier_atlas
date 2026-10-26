"""Тесты проекта JSON, настроек и журналирования."""
import json
import logging
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

from fourier_atlas.core import reconstruct
from fourier_atlas.data.samples import make_sample
from fourier_atlas.export import (ProjectError, build_state, load_project, save_project, spectrum_from_state,
                                  FORMAT)
from fourier_atlas.logging_setup import install_excepthook, setup_logging
from fourier_atlas.settings import DEFAULTS, Settings


def tmp():
    return tempfile.TemporaryDirectory()


def make_state():
    z, spec = make_sample("Звезда", 256)
    spec.set_coeff(1, 0.3 + 0.1j)
    spec.set_coeff(-3, 0.02j)
    spec.set_enabled(7, False)
    spec.set_enabled(-9, False)
    st = build_state("Тест", z, spec, 37, -0.25, 0.4, 0.125, source={"region": "Звезда", "part": 1})
    return z, spec, st


# ---------- проект ----------
def test_project_roundtrip_exact_spectrum_and_view():
    z, spec, st = make_state()
    with tmp() as d:
        p = save_project(Path(d) / "p.fatlas.json", st)
        assert os.listdir(d) == ["p.fatlas.json"]
        back = load_project(p)
    assert back.name == "Тест" and back.max_epicycles == 37 and back.speed == -0.25
    assert back.trail_fraction == 0.4 and back.phase == 0.125 and back.source["region"] == "Звезда"
    assert np.array_equal(back.z, z)                                  # контур без потерь
    s2 = spectrum_from_state(back)
    assert np.array_equal(s2.original, spec.original)                 # исходный спектр восстановлен
    assert np.array_equal(s2.edited, spec.edited) and np.array_equal(s2.enabled, spec.enabled)
    assert np.allclose(reconstruct(s2), reconstruct(spec))


def test_project_stores_only_sparse_edits():
    z, spec, st = make_state()
    assert sorted(st.edits) == [-3, 1] and sorted(st.disabled) == [-9, 7]
    with tmp() as d:
        p = save_project(Path(d) / "p.json", st)
        obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["format"] == FORMAT and obj["version"] == 1
    assert len(obj["spectrum"]["edited"]) == 2 and len(obj["contour"]) == 256


def test_project_none_epicycles_and_defaults():
    z, spec = make_sample("Квадрат", 64)
    st = build_state("Q", z, spec, None, 0.1, 1.0, 0.0)
    with tmp() as d:
        back = load_project(save_project(Path(d) / "q.json", st))
    assert back.max_epicycles is None and back.edits == {} and back.disabled == []


def good_obj():
    z, spec, st = make_state()
    with tmp() as d:
        p = save_project(Path(d) / "p.json", st)
        return json.loads(p.read_text(encoding="utf-8"))


def write_obj(d, obj, name="x.json"):
    p = Path(d) / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_project_errors_are_friendly():
    obj = good_obj()
    cases = []
    cases.append(({**obj, "format": "other"}, "не проект"))
    cases.append(({**obj, "version": 99}, "новой версией"))
    cases.append(({**{k: v for k, v in obj.items() if k != "contour"}}, "повреждён"))
    cases.append(({**obj, "contour": [[0, 0]] * 3}, "повреждён"))
    cases.append(({**obj, "contour": [[0, float("nan")]] * 20}, "повреждён"))
    cases.append(({**obj, "spectrum": {"edited": [{"k": 9999, "re": 1, "im": 0}], "disabled": []}}, "несуществующей"))
    cases.append(({**obj, "spectrum": {"edited": [], "disabled": [5000]}}, "несуществующая"))
    cases.append(({**obj, "view": {"max_epicycles": 99999}}, "пределов"))
    cases.append(({**obj, "view": {"speed": "fast"}}, "Некорректное"))
    cases.append(({**obj, "view": {"trail_fraction": 0}}, "пределов"))
    with tmp() as d:
        for bad, word in cases:
            with pytest.raises(ProjectError) as e:
                load_project(write_obj(d, bad))
            assert word in str(e.value), (word, str(e.value))
        (Path(d) / "junk.json").write_text("{not json")
        with pytest.raises(ProjectError):
            load_project(Path(d) / "junk.json")
        with pytest.raises(ProjectError):
            load_project(Path(d) / "missing.json")


def test_save_to_unwritable_location_raises_project_error():
    z, spec, st = make_state()
    with tmp() as d:
        (Path(d) / "f").write_text("x")
        with pytest.raises(ProjectError):
            save_project(Path(d) / "f" / "p.json", st)


# ---------- настройки ----------
def test_settings_defaults_roundtrip_and_get_set():
    with tmp() as d:
        p = Path(d) / "s.json"
        s = Settings.load(p)
        assert s.data == DEFAULTS and s.get("export.fps") == 30 and s.get("nope", 5) == 5
        s.set("max_epicycles", 250)
        s.set("export.width", 1920)
        s.set("country", "Италия")
        s.set("show_circles", False)
        assert s.save() and not (Path(d) / "s.json.part").exists()
        t = Settings.load(p)
        assert t.get("max_epicycles") == 250 and t.get("export.width") == 1920
        assert t.get("country") == "Италия" and t.get("show_circles") is False
        assert t.get("export.height") == 720                              # остальное — по умолчанию


def test_settings_validation_on_set_and_load():
    with tmp() as d:
        s = Settings.load(Path(d) / "s.json")
        for key, bad in (("max_epicycles", 5000), ("max_epicycles", -1), ("speed", 99),
                         ("projection", "mercator"), ("export.fps", 0), ("export.supersample", 9)):
            with pytest.raises(ValueError):
                s.set(key, bad)
        with pytest.raises(KeyError):
            s.set("unknown", 1)
        p = Path(d) / "bad.json"
        p.write_text(json.dumps({"max_epicycles": 99999, "speed": "x", "projection": "weird",
                                 "show_circles": "yes", "country": 5, "trail_fraction": 0.5,
                                 "export": {"fps": 500, "width": 800, "height": "tall"}}))
        t = Settings.load(p)
        assert t.get("max_epicycles") == 100 and t.get("speed") == 0.08 and t.get("projection") == "auto"
        assert t.get("show_circles") is True and t.get("country") == ""
        assert t.get("trail_fraction") == 0.5 and t.get("export.width") == 800     # валидные берутся
        assert t.get("export.fps") == 30 and t.get("export.height") == 720


def test_settings_corrupt_file_does_not_crash():
    with tmp() as d:
        p = Path(d) / "s.json"
        p.write_text("{{{ broken")
        assert Settings.load(p).data == DEFAULTS
        p.write_text("[1,2,3]")
        assert Settings.load(p).data == DEFAULTS


def test_settings_save_failure_returns_false():
    with tmp() as d:
        (Path(d) / "f").write_text("x")
        s = Settings(Path(d) / "f" / "s.json")
        assert s.save() is False


def test_config_dir_env_override():
    from fourier_atlas.settings import config_dir
    old = os.environ.get("FOURIER_ATLAS_HOME")
    os.environ["FOURIER_ATLAS_HOME"] = "/tmp/fa_home_test"
    try:
        assert config_dir() == Path("/tmp/fa_home_test")
    finally:
        os.environ.pop("FOURIER_ATLAS_HOME") if old is None else os.environ.__setitem__("FOURIER_ATLAS_HOME", old)


# ---------- журналирование ----------
def test_logging_file_rotation_idempotent_and_excepthook():
    import sys
    with tmp() as d:
        path = setup_logging(d, console=False)
        setup_logging(d, console=False)                                   # повторный вызов не дублирует
        logging.getLogger("fourier_atlas.test").warning("проверка журнала «привет»")
        handlers = [h for h in logging.getLogger().handlers if getattr(h, "_fourier_atlas_handler", False)]
        assert len(handlers) == 1
        got = []
        old_hook = sys.excepthook
        install_excepthook(got.append)
        try:
            try:
                raise RuntimeError("взрыв")
            except RuntimeError:
                sys.excepthook(*sys.exc_info())
        finally:
            sys.excepthook = old_hook
        for h in handlers:
            h.flush()
        text = Path(path).read_text(encoding="utf-8")
        assert "проверка журнала «привет»" in text and "Необработанное исключение" in text
        assert "RuntimeError: взрыв" in text and got == ["RuntimeError: взрыв"]
        for h in handlers:
            logging.getLogger().removeHandler(h); h.close()


def test_logging_unavailable_directory_is_not_fatal():
    with tmp() as d:
        (Path(d) / "f").write_text("x")
        assert setup_logging(Path(d) / "f" / "logs", console=False) is None


def test_pipeline_saved_project_to_png_and_csv():
    """Сохранить проект с правками → открыть → экспортировать: результат отражает правки."""
    from PIL import Image
    from fourier_atlas.export import (ExportJob, export_coefficients_csv, run_export)
    z, spec, st = make_state()
    with tmp() as d:
        back = load_project(save_project(Path(d) / "p.json", st))
        s2 = spectrum_from_state(back)
        png = run_export(ExportJob(kind="png", path=str(Path(d) / "a.png"), spectrum=s2.copy(), target=back.z,
                                   max_epicycles=back.max_epicycles, trail_fraction=back.trail_fraction,
                                   phase=back.phase, width=200, height=150, supersample=1))
        assert Image.open(png).size == (200, 150)
        csv_path = export_coefficients_csv(Path(d) / "c.csv", s2)
        text = Path(csv_path).read_text(encoding="utf-8-sig")
        assert text.count("\n") == 257 and ",1," in text            # 256 строк + заголовок


def test_spectrum_copy_is_independent():
    z, spec = make_sample("Квадрат", 64)
    c = spec.copy()
    c.set_coeff(1, 9.0); c.set_enabled(2, False)
    assert spec.coeff(1) != 9.0 and spec.enabled.all()
    assert np.array_equal(c.original, spec.original) and c.original is not spec.original
