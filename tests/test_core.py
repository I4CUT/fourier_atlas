"""Тесты математического ядра: окружность, эллипс, квадрат, синтетическая кривая."""
import subprocess
import sys
import numpy as np
import pytest

from fourier_atlas.core import (
    close_contour, resample_arclength, normalize_contour, to_complex,
    compute_spectrum, reconstruct, reconstruct_at, build_chain,
    rmse, max_error, hausdorff, evaluate, convergence_curve, harmonics_for_rmse,
)

N = 1024
T = np.arange(N) / N


def coeff(spec, k):
    return spec.edited[spec.index_of(k)]


# ---------------- окружность ----------------
def test_circle_spectrum():
    s = compute_spectrum(np.exp(2j * np.pi * T))
    assert abs(coeff(s, 1) - 1) < 1e-12
    others = [abs(coeff(s, k)) for k in s.freqs if k != 1]
    assert max(others) < 1e-12


def test_circle_reconstruction_two_harmonics():
    z = 5 * np.exp(2j * np.pi * T) + (2 - 3j)
    s = compute_spectrum(z)
    s.select_band(1)
    s.set_enabled(-1, False)                      # оставить только k=0 и k=1
    assert np.allclose(reconstruct(s), z, atol=1e-10)
    chain = build_chain(s)
    assert len(chain.freqs) == 2 and chain.freqs[0] == 0
    assert np.isclose(chain.radii[1], 5)


def test_circle_from_polygon_via_arclength():
    ang = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    xy = np.column_stack([np.cos(ang), np.sin(ang)])
    z = to_complex(resample_arclength(xy, N))
    s = compute_spectrum(z)
    assert abs(abs(coeff(s, 1)) - 1) < 1e-4
    assert abs(coeff(s, 0)) < 1e-4


# ---------------- эллипс ----------------
def test_ellipse_coefficients():
    a, b = 3.0, 1.0
    z = a * np.cos(2 * np.pi * T) + 1j * b * np.sin(2 * np.pi * T)
    s = compute_spectrum(z)
    assert abs(coeff(s, 1) - (a + b) / 2) < 1e-12
    assert abs(coeff(s, -1) - (a - b) / 2) < 1e-12   # отрицательная частота
    s.select_band(1)
    assert np.allclose(reconstruct(s), z, atol=1e-12)


def test_ellipse_needs_negative_frequency():
    z = 3 * np.cos(2 * np.pi * T) + 1j * np.sin(2 * np.pi * T)
    s = compute_spectrum(z)
    s.select_band(1)
    s.set_enabled(-1, False)
    assert rmse(z, reconstruct(s)) > 0.5


# ---------------- квадрат ----------------
def square_xy():
    return np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)   # против часовой


def test_square_symmetry_selection_rule():
    z = to_complex(resample_arclength(square_xy(), 4096))
    s = compute_spectrum(z)
    for k in s.freqs:
        if k % 4 != 1:
            assert abs(coeff(s, k)) < 1e-9, k
    ratio = abs(coeff(s, -3)) / abs(coeff(s, 1))
    assert abs(ratio - 1 / 9) < 0.005


def test_square_convergence_monotone():
    z = to_complex(resample_arclength(square_xy(), 2048))
    s = compute_spectrum(z)
    ms, err = convergence_curve(s)
    assert np.all(np.diff(err) <= 1e-15)
    assert err[-1] < 1e-12
    assert err[20] < err[5] < err[1]


def test_resample_uniform_spacing_and_closure():
    xy = resample_arclength(square_xy(), 400)
    assert len(xy) == 400
    closed = np.vstack([xy, xy[0]])
    d = np.hypot(*np.diff(closed, axis=0).T)
    assert np.allclose(d, 8 / 400, atol=1e-12)    # периметр 8


# ---------------- синтетическая кривая ----------------
def synthetic():
    t = T
    z = (1 + 1j) + 2 * np.exp(2j * np.pi * t) + 0.5 * np.exp(-4j * np.pi * t) \
        + 0.25j * np.exp(10j * np.pi * t)
    return z


def test_synthetic_exact_coefficients():
    s = compute_spectrum(synthetic())
    assert abs(coeff(s, 0) - (1 + 1j)) < 1e-12
    assert abs(coeff(s, 1) - 2) < 1e-12
    assert abs(coeff(s, -2) - 0.5) < 1e-12
    assert abs(coeff(s, 5) - 0.25j) < 1e-12
    assert np.sum(np.abs(s.original) > 1e-10) == 4


def test_roundtrip_full_and_interpolation():
    z = synthetic()
    s = compute_spectrum(z)
    assert np.allclose(reconstruct(s), z, atol=1e-12)
    t_fine = np.linspace(0, 1, 777, endpoint=False)
    zt = reconstruct_at(s.freqs, s.edited, t_fine)
    analytic = (1 + 1j) + 2 * np.exp(2j * np.pi * t_fine) + 0.5 * np.exp(-4j * np.pi * t_fine) \
        + 0.25j * np.exp(10j * np.pi * t_fine)
    assert np.allclose(zt, analytic, atol=1e-10)


def test_parseval_matches_direct_reconstruction():
    rng = np.random.default_rng(1)
    z = np.cumsum(rng.normal(size=256) + 1j * rng.normal(size=256))
    z -= np.linspace(0, 1, 256, endpoint=False) * (z[-1] - z[0])   # почти замкнутая
    s = compute_spectrum(z)
    ms, err = convergence_curve(s)
    for m in (0, 3, 10, 50):
        s.select_band(m)
        assert abs(rmse(z, reconstruct(s)) - err[m]) < 1e-9
    assert harmonics_for_rmse(s, err[10]) <= 10


# ---------------- спектр: правки, маска ----------------
def test_edit_does_not_touch_original():
    z = synthetic()
    s = compute_spectrum(z)
    orig = s.original.copy()
    s.set_coeff(1, 7 + 1j)
    assert np.array_equal(s.original, orig)
    assert not np.allclose(reconstruct(s), z)
    assert np.allclose(reconstruct(s, use_edits=False), z, atol=1e-12)
    s.reset_coeff(1)
    assert np.allclose(reconstruct(s), z, atol=1e-12)


def test_toggle_harmonic_removes_its_contribution():
    z = synthetic()
    s = compute_spectrum(z)
    s.set_enabled(5, False)
    expected = z - 0.25j * np.exp(10j * np.pi * T)
    assert np.allclose(reconstruct(s), expected, atol=1e-12)
    s.reset_all()
    assert np.allclose(reconstruct(s), z, atol=1e-12)


# ---------------- эпициклы ----------------
def test_epicycle_tips_match_reconstruction_any_order():
    z = synthetic()
    s = compute_spectrum(z)
    chain = build_chain(s)
    assert np.allclose(chain.tips(T), reconstruct(s), atol=1e-12)
    assert np.isclose(chain.tip(0.3), chain.joints(0.3)[-1])
    assert np.all(np.diff(chain.radii[1:]) <= 1e-15)    # после k=0 — по убыванию
    short = build_chain(s, max_vectors=2)
    assert len(short.freqs) == 2
    rot = build_chain(s, max_rotating=2)
    assert len(rot.freqs) == 3 and rot.freqs[0] == 0 and np.count_nonzero(rot.freqs) == 2
    assert list(build_chain(s, max_rotating=0).freqs) == [0]
    with pytest.raises(ValueError):
        build_chain(s, max_rotating=-1)


def test_joints_chain_is_cumulative():
    s = compute_spectrum(synthetic())
    ch = build_chain(s)
    j = ch.joints(0.1)
    assert np.allclose(np.diff(j), ch.coeffs * np.exp(2j * np.pi * ch.freqs * 0.1))


# ---------------- метрики ----------------
def test_metrics_basic():
    z = synthetic()
    assert rmse(z, z) == 0 and max_error(z, z) == 0
    shifted = np.roll(z, 37)                  # тот же контур, иная фаза
    assert hausdorff(z, shifted) < 0.1
    assert rmse(z, shifted) > 0.1
    e = evaluate(z, z + 0.01)
    assert abs(e.rmse - 0.01) < 1e-12 and e.relative_rmse < 0.01


# ---------------- геометрия / ошибки ----------------
def test_close_contour_and_validation():
    c = close_contour(np.array([[0, 0], [1, 0], [1, 1], [1, 1]], float))
    assert np.array_equal(c[0], c[-1]) and len(c) == 4
    already = close_contour(np.array([[0, 0], [1, 0], [1, 1], [0, 0]], float))
    assert len(already) == 4
    with pytest.raises(ValueError):
        close_contour(np.array([[0, 0], [1, 1]], float))
    with pytest.raises(ValueError):
        close_contour(np.array([[0, 0], [np.nan, 1], [1, 1]], float))
    with pytest.raises(ValueError):
        compute_spectrum(np.array([1 + 0j, 2]))


def test_normalize_roundtrip():
    z = synthetic() * 50 + 1000
    zn, c, sc = normalize_contour(z)
    assert abs(zn.mean()) < 1e-12 and np.isclose(np.abs(zn).max(), 1)
    assert np.allclose(zn * sc + c, z)


def test_core_has_no_qt_dependency():
    code = ("import sys, fourier_atlas.core; "
            "bad=[m for m in sys.modules if m.startswith(('PySide','PyQt'))]; "
            "sys.exit(1 if bad else 0)")
    import os
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": src})
    assert r.returncode == 0
