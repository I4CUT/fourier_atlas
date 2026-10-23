"""Тесты анимации (без Qt): фаза, скорость, след, синхронизация, отсутствие FFT на кадре."""
import subprocess
import sys
import numpy as np
import pytest

from fourier_atlas.animation import AnimationController, Viewport
from fourier_atlas.core import reconstruct
from fourier_atlas.data.samples import SAMPLES, make_sample


def make(name="Эллипс", **kw):
    z, spec = make_sample(name, 512)
    return z, spec, AnimationController(spec, path_samples=512, **kw)


def test_advance_wrap_pause_speed():
    _, _, c = make(speed=0.5)
    assert c.advance(1.0) is False            # на паузе фаза не меняется
    assert c.phase == 0
    c.play()
    assert c.advance(0.5) and np.isclose(c.phase, 0.25)
    c.advance(1.0)
    assert np.isclose(c.phase, 0.75)
    c.advance(0.6)                            # переход через 1.0
    assert np.isclose(c.phase, 0.05)
    c.speed = -0.5
    c.advance(0.2)
    assert np.isclose(c.phase, 0.95)          # обратное направление, оборачивание вниз
    c.pause()
    p = c.phase
    c.advance(5)
    assert c.phase == p


def test_seek_and_reset():
    _, _, c = make()
    c.seek(1.25)
    assert np.isclose(c.phase, 0.25)
    c.play(); c.reset()
    assert c.phase == 0 and not c.playing


def test_tip_matches_reconstruction_at_nodes():
    z, spec, c = make("Звезда")
    c.max_epicycles = None
    n = len(c.path)
    for j in (0, 1, 100, 255, 511):
        c.seek(j / n)
        assert abs(c.frame().joints[-1] - z[j]) < 1e-9
    assert np.allclose(c.path, reconstruct(spec, n), atol=1e-9)


def test_frame_structure_and_trail():
    _, _, c = make(max_epicycles=20)
    c.seek(0.3)
    f = c.frame()
    assert len(f.joints) == len(f.radii) + 1 == 22          # центр k=0 + 20 вращающихся
    assert f.trail[-1] == f.joints[-1]
    assert len(f.trail) == c.path_samples + 1       # trail_fraction = 1
    c.trail_fraction = 0.1
    assert len(c.frame().trail) == round(0.1 * c.path_samples) + 1
    with pytest.raises(ValueError):
        c.trail_fraction = 0


def test_trail_precedes_tip_forward_and_backward():
    _, _, c = make(max_epicycles=None, trail_fraction=0.05)
    c.seek(0.5)
    fwd = c.frame().trail
    assert np.isclose(fwd[-2], c.path[int(0.5 * c.path_samples)])
    c.speed = -0.1
    back = c.frame().trail
    assert np.isclose(back[0], c.path[int(0.5 * c.path_samples) + round(0.05 * c.path_samples) - 1])


def test_edit_sync_after_refresh():
    z, spec, c = make("Окружность", max_epicycles=None)
    before = c.path.copy()
    spec.set_coeff(1, spec.edited[spec.index_of(1)] * 2)   # правка коэффициента
    assert np.allclose(c.path, before)                      # кэш пока старый
    c.refresh()
    assert not np.allclose(c.path, before)
    assert np.allclose(c.path, reconstruct(spec, c.path_samples), atol=1e-9)
    spec.set_enabled(1, False)
    c.refresh()
    assert 1 not in c.chain.freqs


def test_max_epicycles_limits_rotating_vectors():
    _, _, c = make("Сердце", max_epicycles=5)
    assert len(c.chain.freqs) == 6 and c.chain.freqs[0] == 0        # центр + 5 вращающихся
    assert np.count_nonzero(c.chain.freqs) == 5
    c.max_epicycles = 40
    assert np.count_nonzero(c.chain.freqs) == 40
    with pytest.raises(ValueError):
        c.max_epicycles = -1


def test_zero_epicycles_is_center_only_and_bounds():
    z, spec, c = make("Сердце", max_epicycles=0)
    assert list(c.chain.freqs) == [0]
    f = c.frame()
    c.seek(0.37)
    assert abs(c.frame().joints[-1] - spec.edited[spec.index_of(0)]) < 1e-12   # точка не движется
    assert np.allclose(c.path, c.path[0])
    c.max_epicycles = 1024                                  # верхний предел интерфейса
    assert np.count_nonzero(c.chain.freqs) == spec.n - 1    # ограничено числом гармоник (511)
    spec.set_enabled(0, False)                              # центр отключён + 0 эпициклов → пусто
    c.max_epicycles = 0
    assert len(c.chain.freqs) == 0
    fr = c.frame()
    assert len(fr.joints) == 1 and fr.joints[0] == 0 and np.all(c.path == 0)
    c.play(); c.advance(0.1); c.frame()                     # анимация пустой цепочки не падает


def test_no_fft_during_animation(monkeypatch=None):
    _, _, c = make(speed=0.3)
    calls = []
    real = {n: getattr(np.fft, n) for n in ("fft", "ifft", "fftn", "rfft")}
    for n, f in real.items():
        setattr(np.fft, n, lambda *a, _n=n, **k: calls.append(_n))
    try:
        c.play()
        for _ in range(200):
            c.advance(0.016)
            c.frame()
    finally:
        for n, f in real.items():
            setattr(np.fft, n, f)
    assert calls == []


def test_frame_is_fast_for_many_vectors():
    import time
    _, _, c = make("Сердце", max_epicycles=None)
    c.play()
    t = time.perf_counter()
    for _ in range(200):
        c.advance(0.016); c.frame()
    per_frame = (time.perf_counter() - t) / 200
    assert per_frame < 0.004, per_frame           # << 16.6 мс бюджета кадра


def test_viewport_fit_and_roundtrip():
    v = Viewport(800, 600)
    pts = np.array([-2 - 1j, 2 + 1j])
    v.fit(pts)
    sx, sy = v.to_screen(pts)
    assert 0 < sx.min() and sx.max() < 800 and 0 < sy.min() and sy.max() < 600
    assert sy[1] < sy[0]                                 # ось Y вверх -> меньше пикселей
    z = 0.7 - 0.2j
    assert abs(v.to_world(*v.to_screen(z)) - z) < 1e-12


def test_viewport_zoom_keeps_anchor_and_pan():
    v = Viewport(800, 600); v.fit(np.array([-1 - 1j, 1 + 1j]))
    px, py = 123.0, 456.0
    w0 = v.to_world(px, py)
    v.zoom_at(px, py, 2.0)
    assert abs(v.to_world(px, py) - w0) < 1e-12
    s0 = v.to_screen(0.5 + 0.5j)
    v.pan(30, -10)
    s1 = v.to_screen(0.5 + 0.5j)
    assert np.isclose(s1[0] - s0[0], 30) and np.isclose(s1[1] - s0[1], -10)
    with pytest.raises(ValueError):
        v.zoom_at(0, 0, 0)


def test_all_samples_build():
    for name in SAMPLES:
        z, spec = make_sample(name, 256)
        assert len(z) == 256 and np.isclose(np.abs(z).max(), 1)
        assert abs(spec.original[spec.index_of(0)]) < 1e-12


def test_animation_package_has_no_qt():
    import os
    code = ("import sys, fourier_atlas.animation, fourier_atlas.data.samples; "
            "sys.exit(1 if any(m.startswith(('PySide','PyQt')) for m in sys.modules) else 0)")
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    assert subprocess.run([sys.executable, "-c", code],
                          env={**os.environ, "PYTHONPATH": src}).returncode == 0
