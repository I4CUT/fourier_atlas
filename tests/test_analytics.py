"""Тесты редактора спектра, таблицы, сходимости и сравнения (без Qt)."""
import numpy as np
import pytest

from fourier_atlas.analytics import (SpectrumEditor, parse_complex, table_order, spectrum_plot_data,
                                     compare_reconstruction, relative_error_percent,
                                     epicycles_for_relative_error)
from fourier_atlas.animation import AnimationController
from fourier_atlas.core import (build_chain, convergence_by_amplitude, epicycles_for_rmse, reconstruct,
                                rmse)
from fourier_atlas.data.samples import make_sample


def setup(name="Сердце", n=512, **kw):
    z, spec = make_sample(name, n)
    ctrl = AnimationController(spec, path_samples=n, **kw)
    ed = SpectrumEditor(ctrl, z)
    events = []
    ed.subscribe(events.append)
    return z, spec, ctrl, ed, events


# ---------- сходимость ----------
def test_convergence_by_amplitude_matches_direct_reconstruction():
    z, spec = make_sample("Звезда", 512)
    ns, err = convergence_by_amplitude(spec)
    assert np.all(np.diff(err) <= 1e-15) and err[-1] < 1e-12 and len(ns) == 512
    for n in (0, 1, 5, 20, 100):
        spec.enabled[:] = True
        chain = build_chain(spec, max_rotating=n)
        recon = np.array([chain.tips(np.arange(512) / 512)]).ravel()
        assert abs(rmse(z, recon) - err[n]) < 1e-9, n
    assert epicycles_for_rmse(spec, err[30]) <= 30


def test_epicycles_for_relative_error():
    z, spec = make_sample("Сердце", 512)
    n1 = epicycles_for_relative_error(spec, z, 1.0)
    n01 = epicycles_for_relative_error(spec, z, 0.1)
    assert 0 < n1 < n01 <= 511
    chain = build_chain(spec, max_rotating=n01)
    recon = chain.tips(np.arange(512) / 512)
    assert relative_error_percent(rmse(z, recon), z) <= 0.1 + 1e-9
    assert epicycles_for_relative_error(spec, z, 1e-30) in (None, 511)


# ---------- редактор ----------
def test_edit_updates_reconstruction_and_keeps_original():
    z, spec, ctrl, ed, events = setup(max_epicycles=None)
    orig = spec.original.copy()
    before = ctrl.path.copy()
    assert ed.set_coeff(1, 0.5 + 0.5j) is True
    assert events == ["changed"]
    assert np.array_equal(spec.original, orig)                       # исходный спектр цел
    assert not np.allclose(ctrl.path, before)                        # реконструкция обновилась
    assert np.allclose(ctrl.path, reconstruct(spec, ctrl.path_samples), atol=1e-9)
    assert ed.set_coeff(1, 0.5 + 0.5j) is False and events == ["changed"]   # без изменения — без события
    assert spec.edited_mask()[spec.index_of(1)] and spec.edited_mask().sum() == 1
    ed.reset_coeff(1)
    assert np.allclose(ctrl.path, before, atol=1e-9) and not spec.edited_mask().any()


def test_toggle_harmonics_and_band_sync_with_chain():
    z, spec, ctrl, ed, events = setup(max_epicycles=None)
    ed.set_enabled(1, False)
    assert 1 not in ctrl.chain.freqs
    ed.set_enabled_many([2, 3, -1], False)
    assert not {1, 2, 3, -1} & set(ctrl.chain.freqs.tolist())
    ed.enable_band(5)
    assert set(ctrl.chain.freqs.tolist()) == set(range(-5, 6))
    ed.enable_all(True)
    assert len(ctrl.chain.freqs) == spec.n
    assert ed.enable_all(True) is False
    assert np.allclose(ctrl.path, reconstruct(spec, ctrl.path_samples), atol=1e-9)


def test_undo_redo_and_history_rules():
    z, spec, ctrl, ed, events = setup(max_epicycles=None)
    base = ctrl.path.copy()
    ed.set_coeff(1, 2.0)
    ed.set_enabled(2, False)
    state2 = ctrl.path.copy()
    assert ed.can_undo and not ed.can_redo
    ed.undo()
    assert spec.enabled[spec.index_of(2)] and abs(spec.coeff(1) - 2.0) < 1e-12
    ed.undo()
    assert np.allclose(ctrl.path, base, atol=1e-9) and not ed.can_undo and ed.can_redo
    ed.redo(); ed.redo()
    assert np.allclose(ctrl.path, state2, atol=1e-9) and not ed.can_redo
    ed.undo()
    ed.set_coeff(3, 1j)                                               # новая правка очищает redo
    assert not ed.can_redo
    assert ed.undo() and ed.undo() and not ed.undo()


def test_history_limit_and_reset_all_and_replace():
    z, spec, ctrl, ed, events = setup(max_epicycles=None)
    ed.history_limit = 3
    for v in (1, 2, 3, 4, 5):
        ed.set_coeff(1, v)
    n = 0
    while ed.undo():
        n += 1
    assert n == 3
    ed.set_coeff(4, 9); ed.set_enabled(5, False)
    assert ed.reset_all() and not spec.edited_mask().any() and spec.enabled.all()
    assert ed.reset_all() is False
    z2, spec2 = make_sample("Квадрат", 512)
    events.clear()
    ed.replace(z2, spec2)
    assert events == ["replaced"] and ctrl.spectrum is spec2 and not ed.can_undo and ed.z is z2


def test_editor_validation_and_epicycle_count_event():
    z, spec, ctrl, ed, events = setup()
    with pytest.raises(ValueError):
        ed.set_coeff(10_000, 1)
    with pytest.raises(ValueError):
        ed.set_coeff(1, complex("nan"))
    with pytest.raises(ValueError):
        ed.set_coeff(1, complex(float("inf"), 0))
    ed.set_max_epicycles(7)
    assert events == ["changed"] and np.count_nonzero(ctrl.chain.freqs) == 7
    assert not ed.can_undo                                             # число эпициклов — не в истории


def test_edit_changes_epicycle_chain_order_by_amplitude():
    z, spec, ctrl, ed, events = setup(max_epicycles=3)
    big = 100.0
    ed.set_coeff(17, big)                                              # ничтожная гармоника стала главной
    assert ctrl.chain.freqs[1] == 17 and ctrl.chain.radii[1] == big


# ---------- разбор ввода ----------
def test_parse_complex():
    assert parse_complex("1.5-2i") == 1.5 - 2j
    assert parse_complex("3+0,5j") == 3 + 0.5j
    assert parse_complex(" -4 ") == -4 and parse_complex("2i") == 2j and parse_complex("-i") == -1j
    assert parse_complex("1e-1+2E1i") == 0.1 + 20j
    assert parse_complex("−3") == -3                                    # типографский минус
    assert abs(parse_complex("2@1.5707963267948966") - 2j) < 1e-12
    assert abs(parse_complex("2@90°") - 2j) < 1e-12 and abs(parse_complex("1∠180°") + 1) < 1e-12
    for bad in ("", "abc", "inf", "nan", "1+", "1@x", "-1@0", "1++2i"):
        with pytest.raises(ValueError):
            parse_complex(bad)


# ---------- таблица и графики ----------
def test_table_order_and_plot_data():
    z, spec, ctrl, ed, events = setup()
    o = table_order(spec, "k")
    assert np.all(np.diff(spec.freqs[o]) > 0)
    a = table_order(spec, "amp")
    assert np.all(np.diff(np.abs(spec.original[a])) <= 1e-15)
    with pytest.raises(ValueError):
        table_order(spec, "x")
    ed.set_coeff(2, 3 + 4j); ed.set_enabled(-1, False)
    d = spectrum_plot_data(spec)
    i2, im1 = list(d.k).index(2), list(d.k).index(-1)
    assert abs(d.amp[i2] - 5) < 1e-12 and d.edited[i2] and not d.edited[im1]
    assert not d.enabled[im1] and d.enabled[i2]
    assert np.isnan(d.phase[list(d.k).index(0)]) or abs(d.amp[list(d.k).index(0)]) > 0   # c0≈0 → фаза не определена
    assert len(d.k) == spec.n


def test_compare_reflects_current_state():
    z, spec, ctrl, ed, events = setup()
    recon, e = compare_reconstruction(z, spec)
    assert e.rmse < 1e-12 and e.max_error < 1e-12 and len(recon) == len(z)
    ed.enable_band(8)
    recon, e2 = compare_reconstruction(z, spec)
    assert e2.rmse > 1e-3 and e2.hausdorff <= e2.max_error + 1e-12
    ed.set_coeff(1, spec.coeff(1) * 1.2)
    _, e3 = compare_reconstruction(z, spec)
    assert e3.rmse != e2.rmse
    assert abs(relative_error_percent(e2.rmse, z) - 100 * e2.relative_rmse) < 1e-12


def test_analytics_has_no_qt():
    import os, subprocess, sys
    code = ("import sys, fourier_atlas.analytics; "
            "sys.exit(1 if any(m.startswith(('PySide','PyQt','pyqtgraph')) for m in sys.modules) else 0)")
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    assert subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": src}).returncode == 0
