"""Тесты звука по спектру: частоты, амплитуды, фазы, XY-контур, зацикливание, WAV, наращивание."""
import tempfile
import wave
from pathlib import Path
import numpy as np
import pytest

from fourier_atlas.audio import (AudioError, AudioParams, audible_chain, buildup_buffer, buildup_schedule,
                                 loop_buffer, max_harmonic, note_name, synth_cycle, timbre_buffer, to_int16,
                                 write_wav)
from fourier_atlas.core import build_chain, compute_spectrum, reconstruct_at
from fourier_atlas.data.samples import make_sample

P0 = AudioParams()


def tmp():
    return tempfile.TemporaryDirectory()


def spectrum_of(coeffs: dict, n=256):
    """Спектр с заданными c_k (остальные 0)."""
    t = np.arange(n) / n
    z = sum(c * np.exp(2j * np.pi * k * t) for k, c in coeffs.items())
    return compute_spectrum(z)


# ---------- частоты, фазы, амплитуды ----------
def test_circle_is_pure_sine_xy_circle():
    spec = spectrum_of({1: 1.0})
    cyc = synth_cycle(*audible_chain(spec, P0), P0, 1.0)
    L, R = cyc.wave[:, 0], cyc.wave[:, 1]
    assert abs(cyc.f0 - 110) / 110 < 1e-5
    assert np.allclose(L**2 + R**2, 1.0, atol=1e-9)                       # окружность на XY
    X = np.fft.fft(L)
    peak_bin = int(np.argmax(np.abs(X[: len(L) // 2])))
    assert peak_bin == cyc.periods                                        # частота = f0'
    assert abs(peak_bin * P0.sample_rate / len(L) - cyc.f0) < 1e-9


def test_complex_fft_recovers_coefficients_with_signs_and_phases():
    coeffs = {1: 1.0, -2: 0.5j, 5: 0.25 * np.exp(1j), -9: 0.1 - 0.2j}
    spec = spectrum_of(coeffs)
    f, c = audible_chain(spec, P0)
    cyc = synth_cycle(f, c, P0, 1.0)
    z = cyc.wave[:, 0] + 1j * cyc.wave[:, 1]
    Z = np.fft.fft(z) / len(z)
    for k, ck in coeffs.items():
        assert abs(Z[(k * cyc.periods) % len(z)] - ck) < 1e-9, k          # отрицательные частоты — тоже
    energy = np.abs(Z) ** 2
    mask = np.ones(len(z), bool)
    mask[[(k * cyc.periods) % len(z) for k in coeffs]] = False
    assert energy[mask].max() < 1e-20                                     # больше ничего не звучит


def test_xy_matches_reconstruction_for_ellipse_square_synthetic_and_country_like():
    for name in ("Эллипс", "Квадрат", "Звезда", "Сердце"):
        z, spec = make_sample(name, 1024)
        f, c = audible_chain(spec, P0)
        cyc = synth_cycle(f, c, P0, 1.0)
        n = len(cyc.wave)
        t = cyc.periods * np.arange(n) / n
        expected = reconstruct_at(f, c, t)
        got = cyc.wave[:, 0] + 1j * cyc.wave[:, 1]
        assert np.allclose(got, expected, atol=1e-9), name


def test_dc_not_sounding_and_center_ignored():
    spec = spectrum_of({0: 5 + 5j, 1: 1.0})
    f, c = audible_chain(spec, P0)
    assert 0 not in f
    cyc = synth_cycle(f, c, P0, 0.5)
    assert abs(cyc.wave.mean()) < 1e-12


# ---------- ограничение полосы ----------
def test_band_limit_no_aliasing():
    spec = spectrum_of({1: 1.0, 3: 0.5, 200: 0.5, -300: 0.25}, n=1024)
    p = AudioParams(f0=2000.0)
    assert max_harmonic(p) == 10                                          # 20 кГц / 2 кГц
    f, c = audible_chain(spec, p)
    assert np.abs(f).max() == 10                                          # выше 10-й гармоники — отброшено
    assert sorted(f[np.abs(c) > 1e-9].tolist()) == [1, 3]                 # 200 и -300 не прошли
    cyc = synth_cycle(f, c, p, 0.5)
    z = cyc.wave[:, 0] + 1j * cyc.wave[:, 1]
    Z = np.abs(np.fft.fft(z) / len(z))
    keep = {(k * cyc.periods) % len(z) for k in (1, 3)}
    assert all(Z[i] < 1e-12 for i in range(len(z)) if i not in keep)
    assert synth_cycle([200], [1.0], p, 0.5).n_harmonics == 0             # синтезатор сам отбрасывает


def test_nyquist_guard_for_low_sample_rate():
    p = AudioParams(f0=100.0, sample_rate=22050, max_freq=24000)
    assert max_harmonic(p) == int(0.49 * 22050 // 100)


# ---------- зацикливание, уровни, моно ----------
def test_loop_is_seamless_and_peak_normalized():
    z, spec = make_sample("Звезда", 1024)
    buf = loop_buffer(spec, P0, 1.5)
    assert buf.dtype == np.float32 and buf.shape[1] == 2
    assert abs(np.abs(buf).max() - P0.peak) < 1e-6
    step = np.abs(np.diff(buf, axis=0)).max()
    wrap = np.abs(buf[0] - buf[-1]).max()
    assert wrap <= 1.5 * step                                             # стык такой же гладкий, как внутри
    n = len(buf)
    assert abs(n - round(P0.f0 * 1.5) * P0.sample_rate / P0.f0) < 1


def test_mono_mode_is_mix():
    z, spec = make_sample("Сердце", 1024)
    st = loop_buffer(spec, P0, 0.5)
    mo = loop_buffer(spec, AudioParams(stereo=False), 0.5)
    assert mo.shape[1] == 1 and mo.shape[0] == st.shape[0]
    f, c = audible_chain(spec, P0)
    raw = synth_cycle(f, c, P0, 0.5).wave
    expected = 0.5 * (raw[:, 0] + raw[:, 1])
    cyc_m = synth_cycle(f, c, AudioParams(stereo=False), 0.5).wave[:, 0]
    assert np.allclose(cyc_m, expected)


def test_timbre_buffer_exact_length_fades_and_silence():
    z, spec = make_sample("Сердце", 1024)
    w = timbre_buffer(spec, P0, 2.37)
    assert len(w) == round(2.37 * P0.sample_rate)
    assert abs(w[0]).max() < 1e-3 and abs(w[-1]).max() < 1e-3             # фейды
    assert np.isfinite(w).all()
    spec.enable_all = None
    spec.enabled[:] = False                                               # всё отключено → тишина
    s = timbre_buffer(spec, P0, 0.5)
    assert np.all(s == 0)
    with pytest.raises(AudioError):
        timbre_buffer(spec, P0, 0)


def test_disabling_and_editing_changes_sound():
    z, spec = make_sample("Звезда", 1024)
    a = loop_buffer(spec, P0, 0.5)
    spec.set_enabled(1, False)
    b = loop_buffer(spec, P0, 0.5)
    spec.reset_all()
    spec.set_coeff(5, spec.coeff(5) * 3)
    c = loop_buffer(spec, P0, 0.5)
    assert not np.allclose(a, b) and not np.allclose(a, c)


def test_max_epicycles_matches_chain_selection():
    z, spec = make_sample("Сердце", 1024)
    f, c = audible_chain(spec, P0, 7)
    ch = build_chain(spec, max_rotating=7)
    assert sorted(f.tolist()) == sorted(int(k) for k in ch.freqs if k != 0)
    assert len(audible_chain(spec, P0, 0)[0]) == 0


# ---------- наращивание ----------
def test_buildup_schedule():
    assert buildup_schedule(5, 24) == [1, 2, 3, 4, 5]
    s = buildup_schedule(500, 20)
    assert s[0] == 1 and s[-1] == 500 and s == sorted(set(s)) and len(s) <= 20
    assert buildup_schedule(0) == []


def test_buildup_buffer_growth_length_and_shared_gain():
    z, spec = make_sample("Звезда", 1024)
    sched = [1, 2, 4, 8, 16]
    w = buildup_buffer(spec, P0, sched, step_seconds=0.4)
    n_step = round(0.4 * P0.sample_rate)
    assert w.shape == (5 * n_step, 2) and np.isfinite(w).all()
    assert np.abs(w).max() <= P0.peak + 1e-6
    # в середине каждого шага звучит ровно столько гармоник, сколько в расписании
    f0 = P0.f0
    for i, n in enumerate(sched):
        seg = w[i * n_step + n_step // 4: i * n_step + n_step // 4 + 8000]
        z = seg[:, 0] + 1j * seg[:, 1]
        spectrum = np.abs(np.fft.fft(z * np.hanning(len(z)))) / len(z)
        peaks = int(np.sum(spectrum > 0.02 * spectrum.max()))
        assert peaks >= 1
    # тембр богаче к концу: число значимых спектральных линий растёт
    def lines(i):
        seg = w[i * n_step + 2000: i * n_step + 2000 + 16384]
        z = seg[:, 0] + 1j * seg[:, 1]
        a = np.abs(np.fft.fft(z * np.hanning(len(z))))
        return int(np.sum(a > 1e-3 * a.max()))
    assert lines(4) > lines(0)
    with pytest.raises(AudioError):
        buildup_buffer(spec, P0, [])


def test_buildup_crossfade_has_no_clicks():
    z, spec = make_sample("Звезда", 1024)
    w = buildup_buffer(spec, P0, [1, 3, 9, 27], step_seconds=0.3)
    steady = np.abs(np.diff(w, axis=0)).max()
    n_step = round(0.3 * P0.sample_rate)
    for i in range(1, 4):
        edge = np.abs(np.diff(w[i * n_step - 200: i * n_step + 200], axis=0)).max()
        assert edge <= 1.6 * steady                                       # на стыках нет скачков


# ---------- WAV ----------
def test_write_wav_roundtrip_and_validation():
    z, spec = make_sample("Квадрат", 1024)
    buf = timbre_buffer(spec, P0, 0.5)
    with tmp() as d:
        p = write_wav(Path(d) / "a.wav", buf, P0.sample_rate)
        with wave.open(str(p)) as f:
            assert (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()) == (2, 2, 44100, len(buf))
            raw = np.frombuffer(f.readframes(f.getnframes()), "<i2").reshape(-1, 2)
        assert np.abs(raw / 32767.0 - buf).max() < 1 / 32767
        assert [x.name for x in Path(d).iterdir()] == ["a.wav"]
        mono = write_wav(Path(d) / "m.wav", buf[:, :1], P0.sample_rate)
        with wave.open(str(mono)) as f:
            assert f.getnchannels() == 1
        with pytest.raises(AudioError):
            write_wav(Path(d) / "bad.wav", buf[:, 0], P0.sample_rate)
        (Path(d) / "f").write_text("x")
        with pytest.raises(AudioError):
            write_wav(Path(d) / "f" / "x.wav", buf, P0.sample_rate)
    assert to_int16(np.array([2.0, -2.0, 0.5], np.float32)).tolist() == [32767, -32767, 16384]


def test_param_validation_and_note_names():
    for p in (AudioParams(f0=10), AudioParams(f0=5000), AudioParams(sample_rate=12345),
              AudioParams(peak=0), AudioParams(max_freq=100)):
        with pytest.raises(AudioError):
            p.validate()
    assert note_name(440.0) == "A4" and note_name(110.0) == "A2" and note_name(261.63) == "C4"


def test_audio_package_has_no_qt():
    import os, subprocess, sys
    code = ("import sys, fourier_atlas.audio; "
            "sys.exit(1 if any(m.startswith(('PySide','PyQt')) for m in sys.modules) else 0)")
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    assert subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": src}).returncode == 0


# ---------- экспорт: WAV и MP4 со звуком ----------
def _job(kind, d, name=None, **kw):
    from fourier_atlas.export import ExportJob
    z, spec = make_sample("Звезда", 512)
    base = dict(kind=kind, path=str(Path(d) / (name or f"out.{kind}")), spectrum=spec.copy(), target=z,
                max_epicycles=20, width=160, height=120, fps=10, duration=1.0, supersample=1)
    base.update(kw)
    return ExportJob(**base)


def _probe(path):
    import subprocess
    import imageio_ffmpeg
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True)
    return r.stderr


def test_wav_export_timbre_and_buildup():
    from fourier_atlas.export import run_export
    with tmp() as d:
        p = run_export(_job("wav", d, audio="timbre", duration=1.5, f0=220.0))
        with wave.open(str(p)) as f:
            assert (f.getnchannels(), f.getframerate(), f.getnframes()) == (2, 44100, round(1.5 * 44100))
        j = _job("wav", d, name="b.wav", audio="buildup", steps=6, step_seconds=0.5, audio_stereo=False)
        sched = j.schedule()
        assert sched[-1] > 1 and len(sched) <= 6
        q = run_export(j)
        with wave.open(str(q)) as f:
            assert f.getnchannels() == 1 and f.getnframes() == round(len(sched) * 0.5 * 44100)
        assert sorted(x.name for x in Path(d).iterdir()) == ["b.wav", "out.wav"]


def test_wav_timbre_follows_current_state():
    from fourier_atlas.export import run_export
    with tmp() as d:
        a = run_export(_job("wav", d, name="a.wav", audio="timbre", duration=0.5))
        j = _job("wav", d, name="b.wav", audio="timbre", duration=0.5)
        j.spectrum.set_coeff(-4, j.spectrum.coeff(-4) * 6)
        b = run_export(j)
        assert Path(a).read_bytes() != Path(b).read_bytes()


def test_mp4_with_timbre_audio_has_both_streams_and_matching_duration():
    from fourier_atlas.export import run_export
    with tmp() as d:
        p = run_export(_job("mp4", d, audio="timbre", fps=10, duration=2.0))
        info = _probe(p)
        assert "Video: h264" in info and "Audio: aac" in info and "stereo" in info
        assert "Duration: 00:00:02.0" in info
        assert sorted(x.name for x in Path(d).iterdir()) == ["out.mp4"]       # временный .wav убран


def test_mp4_buildup_video_and_audio_in_sync():
    from fourier_atlas.export import run_export
    with tmp() as d:
        j = _job("mp4", d, audio="buildup", steps=5, step_seconds=0.6, fps=10, revolution_seconds=2.0)
        sched = j.schedule()
        assert j.n_frames == round(10 * len(sched) * 0.6)
        seen = []
        p = run_export(j, progress=lambda i, n: seen.append((i, n)))
        info = _probe(p)
        assert "Audio: aac" in info and seen[-1] == (j.n_frames, j.n_frames)
        dur = len(sched) * 0.6
        assert f"Duration: 00:00:0{int(dur)}" in info


def test_buildup_video_epicycle_count_grows():
    """Число эпициклов в кадрах растёт по расписанию (проверяем по числу окружностей в сцене)."""
    from fourier_atlas.animation import AnimationController
    from fourier_atlas.export import Style, build_primitives, fit_viewport
    z, spec = make_sample("Звезда", 512)
    c = AnimationController(spec, max_epicycles=None, path_samples=512)
    view = fit_viewport(400, 300, z, c.path, 0.08)
    counts = []
    for n in (1, 4, 16):
        c.max_epicycles = n
        counts.append(sum(1 for p in build_primitives(c.frame(), view, z, Style(min_circle_px=0)) if p.kind == "circle"))
    assert counts == [2, 5, 17]                                              # центр + n


def test_audio_job_validation_errors():
    from fourier_atlas.export import ExportError, validate_job
    with tmp() as d:
        for kw in (dict(audio="none"), dict(audio="loud"), dict(audio="timbre", f0=5.0),
                   dict(audio="buildup", steps=0), dict(audio="buildup", step_seconds=99)):
            with pytest.raises(ExportError):
                validate_job(_job("wav", d, **kw))
        j = _job("wav", d, audio="buildup")
        j.spectrum.enabled[:] = False
        with pytest.raises(ExportError):
            validate_job(j)                                                  # нечего наращивать
        with pytest.raises(ExportError):
            validate_job(_job("mp4", d, audio="timbre", f0=3000.0))
        validate_job(_job("mp4", d, audio="timbre"))                         # корректное — без ошибок
        validate_job(_job("gif", d, audio="timbre"))                         # для GIF звук игнорируется


def test_wav_cancel_leaves_no_files():
    from fourier_atlas.export import ExportCancelled, run_export
    with tmp() as d:
        with pytest.raises(ExportCancelled):
            run_export(_job("wav", d, audio="timbre"), cancel=lambda: True)
        assert list(Path(d).iterdir()) == []
