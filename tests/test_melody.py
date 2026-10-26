"""Тесты режима «мелодия»: момент нот, ритм 1:2:3, высоты, панорама, рендер, MIDI."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from fourier_atlas.animation import AnimationController
from fourier_atlas.audio import (AudioError, AudioParams, MelodyParams, SCALES, event_strength, event_times,
                                 loop_melody, midi_for_degree, midi_to_freq, render_loop, render_melody,
                                 select_voices, write_midi)
from fourier_atlas.core import build_chain, compute_spectrum
from fourier_atlas.data.samples import make_sample

AP = AudioParams()
MP = MelodyParams()


def tmp():
    return tempfile.TemporaryDirectory()


def spectrum_of(coeffs: dict, n=256):
    t = np.arange(n) / n
    return compute_spectrum(sum(c * np.exp(2j * np.pi * k * t) for k, c in coeffs.items()))


# ---------- ритм ----------
def test_event_times_vector_points_up_and_count_equals_abs_k():
    for k, c in ((1, 1.0), (3, 0.5 * np.exp(0.7j)), (-2, 0.3 - 0.4j), (-5, 1j), (7, -1.0)):
        t = event_times(k, c)
        assert len(t) == abs(k) and np.all((0 <= t) & (t < 1))
        ang = np.angle(c * np.exp(2j * np.pi * k * t))
        assert np.allclose(ang, np.pi / 2, atol=1e-9), (k, ang)          # вектор смотрит строго вверх


def test_polyrhythm_and_phase_shift():
    t1, t2, t3 = (event_times(k, 1.0) for k in (1, 2, 3))
    assert (len(t1), len(t2), len(t3)) == (1, 2, 3)
    assert np.allclose(np.diff(t3), 1 / 3)                               # равномерный ритм внутри голоса
    # смещение фазы сдвигает ритм
    a = event_times(2, 1.0)
    b = event_times(2, np.exp(0.5j))
    assert not np.allclose(a, b)
    with pytest.raises(ValueError):
        event_times(0, 1.0)


def test_event_times_quantize_snaps_to_grid():
    t = event_times(3, np.exp(0.37j), quantize=16)
    assert np.allclose(t * 16, np.round(t * 16))
    assert np.all((0 <= t) & (t < 1))


def test_video_sync_vector_vertical_at_event_time():
    """В анимации вектор голоса в момент ноты направлен вверх (экран/математика совпадают)."""
    z, spec = make_sample("Звезда", 512)
    voices = select_voices(spec, MP)
    v = voices[0]
    ctrl = AnimationController(spec, max_epicycles=None, path_samples=512)
    idx = int(np.where(ctrl.chain.freqs == v.k)[0][0])
    for t in v.times[:3]:
        ctrl.seek(t)
        j = ctrl.chain.joints(ctrl.phase)
        d = j[idx + 1] - j[idx]
        assert abs(d.real) < 1e-9 * max(1, abs(d)) and d.imag > 0


# ---------- выбор голосов ----------
def test_select_voices_top_by_amplitude_pitch_by_abs_k_velocity_pan():
    spec = spectrum_of({1: 1.0, -2: 0.5, 3: 0.25, 4: 0.1, 5: 0.05})
    mp = MelodyParams(voices=3)
    vs = select_voices(spec, mp)
    assert sorted(v.k for v in vs) == [-2, 1, 3]                         # три крупнейшие
    assert [v.k for v in vs] == [1, -2, 3]                               # высота растёт с |k|
    assert [v.midi for v in vs] == [57, 60, 62]                          # A3 C4 D4 (минорная пентатоника)
    assert vs[0].velocity == 1.0 and vs[1].velocity < vs[0].velocity
    assert vs[0].pan > 0 > vs[1].pan                                     # k>0 вправо, k<0 влево
    assert abs(vs[0].freq - 220.0) < 1e-9


def test_rate_cap_drops_fast_voices_and_disabled():
    spec = spectrum_of({1: 1.0, 40: 0.9, 3: 0.5})
    assert sorted(v.k for v in select_voices(spec, MelodyParams(bar_seconds=4.0))) == [1, 3]    # 40/4=10 > 8
    assert 40 in [v.k for v in select_voices(spec, MelodyParams(bar_seconds=6.0))]              # 40/6 < 8
    assert [v.k for v in select_voices(spec, MelodyParams(voices=8))] == [1, 3]   # шумовые k=0.0 не играют
    spec.set_enabled(1, False)
    assert 1 not in [v.k for v in select_voices(spec, MP)]
    spec.enabled[:] = False
    assert select_voices(spec, MP) == []


def test_scales_and_midi_helpers():
    assert [midi_for_degree(i, 57, "minor_pentatonic") for i in range(7)] == [57, 60, 62, 64, 67, 69, 72]
    assert [midi_for_degree(i, 60, "major") for i in range(8)] == [60, 62, 64, 65, 67, 69, 71, 72]
    assert midi_for_degree(200, 57, "chromatic") <= 108                  # сворачивание октав
    assert abs(midi_to_freq(69) - 440) < 1e-9 and abs(midi_to_freq(57) - 220) < 1e-9
    assert set(SCALES) == {"minor_pentatonic", "major_pentatonic", "natural_minor", "major", "chromatic"}


def test_settings_scale_choices_match_audio_scales():
    from fourier_atlas.settings import SCHEMA
    assert set(SCHEMA["audio.scale"][1]) == set(SCALES)


def test_params_validation():
    for kw in (dict(voices=0), dict(voices=40), dict(bar_seconds=0.5), dict(bar_seconds=60),
               dict(scale="dorian"), dict(root_midi=10), dict(quantize=7)):
        with pytest.raises(AudioError):
            MelodyParams(**kw).validate()


# ---------- рендер ----------
def onset_positions(sig, threshold, block=441):
    """Начала нот: пересечение порога огибающей (максимум по блокам ~10 мс), а не нулей синусоиды."""
    n = len(sig) // block * block
    env = np.abs(sig[:n]).reshape(-1, block).max(axis=1)
    above = env > threshold
    return np.where(above & ~np.roll(above, 1))[0] * block


def test_render_places_notes_at_event_times_for_single_voice():
    spec = spectrum_of({3: 1.0 * np.exp(0.9j)})
    mp = MelodyParams(voices=1)
    vs = select_voices(spec, mp)
    buf = render_loop(vs, mp, AudioParams(stereo=False))
    n = len(buf)
    assert buf.shape == (n, 1) and n == round(4.0 * 44100) and abs(np.abs(buf).max() - AP.peak) < 1e-6
    ons = onset_positions(buf[:, 0], 0.4 * AP.peak)
    expected = np.round(np.array(vs[0].times) * n).astype(int)
    assert len(ons) == 3
    assert np.all(np.abs(np.sort(ons) - np.sort(expected)) < 0.03 * 44100)      # в пределах 30 мс


def test_loop_is_circular_tail_wraps_and_seamless():
    spec = spectrum_of({1: 1.0})
    mp = MelodyParams(voices=1, bar_seconds=1.0)                        # хвост (1.2 с) длиннее такта
    vs = select_voices(spec, mp)
    buf = render_loop(vs, mp, AudioParams(stereo=False))
    step = np.abs(np.diff(buf[:, 0])).max()
    assert abs(buf[0, 0] - buf[-1, 0]) <= 2 * step                      # нет скачка на стыке
    assert np.isfinite(buf).all()


def test_stereo_pan_by_rotation_direction():
    spec = spectrum_of({1: 1.0, -2: 1.0})
    vs = select_voices(spec, MelodyParams(voices=2))
    right_only = render_loop([v for v in vs if v.k > 0], MP, AP)
    left_only = render_loop([v for v in vs if v.k < 0], MP, AP)
    assert np.abs(right_only[:, 1]).sum() > 1.5 * np.abs(right_only[:, 0]).sum()
    assert np.abs(left_only[:, 0]).sum() > 1.5 * np.abs(left_only[:, 1]).sum()


def test_render_melody_bars_fades_and_errors():
    z, spec = make_sample("Сердце", 1024)
    w = render_melody(spec, MelodyParams(bar_seconds=2.0), AP, bars=3)
    assert w.shape == (3 * round(2.0 * 44100), 2) and abs(w[0]).max() < 1e-2
    assert np.abs(w).max() <= AP.peak + 1e-6
    assert np.allclose(w[100000:100500], w[100000 + 88200:100500 + 88200], atol=1e-6)   # такты одинаковы
    spec.enabled[:] = False
    with pytest.raises(AudioError):
        render_melody(spec, MP, AP, 2)
    assert np.all(loop_melody(spec, MP, AP) == 0)
    z, spec = make_sample("Сердце", 1024)
    with pytest.raises(AudioError):
        render_melody(spec, MP, AP, 0)


def test_melody_follows_edits_and_differs_between_contours():
    za, a = make_sample("Звезда", 1024)
    zb, b = make_sample("Сердце", 1024)
    ra, rb = loop_melody(a, MP, AP), loop_melody(b, MP, AP)
    assert not np.allclose(ra, rb)
    a.set_coeff(-4, a.coeff(-4) * 0.1)
    assert not np.allclose(loop_melody(a, MP, AP), ra)


def test_event_strength_decays():
    spec = spectrum_of({2: 1.0})
    v = select_voices(spec, MelodyParams(voices=1))
    t0 = v[0].times[0]
    on = event_strength(v, t0, 4.0)[2]
    later = event_strength(v, t0 + 0.05, 4.0)[2]
    assert abs(on - 1.0) < 1e-9 and 0 < later < on


# ---------- MIDI ----------
def parse_midi(data: bytes):
    assert data[:4] == b"MThd" and int.from_bytes(data[4:8], "big") == 6
    fmt, ntr, div = (int.from_bytes(data[i:i + 2], "big") for i in (8, 10, 12))
    assert data[14:18] == b"MTrk"
    ln = int.from_bytes(data[18:22], "big")
    body = data[22:22 + ln]
    assert len(data) == 22 + ln
    i, tick, ev, tempo = 0, 0, [], None
    status = None
    while i < len(body):
        d = 0
        while True:
            b = body[i]; i += 1
            d = (d << 7) | (b & 0x7F)
            if not b & 0x80:
                break
        tick += d
        if body[i] == 0xFF:
            typ, ln2 = body[i + 1], body[i + 2]
            payload = body[i + 3:i + 3 + ln2]
            if typ == 0x51:
                tempo = int.from_bytes(payload, "big")
            i += 3 + ln2
            if typ == 0x2F:
                break
            continue
        status = body[i]
        n = 2
        if status & 0xF0 in (0xC0, 0xD0):
            n = 1
        ev.append((tick, status, tuple(body[i + 1:i + 1 + n])))
        i += 1 + n
    return fmt, ntr, div, tempo, ev


def test_midi_file_structure_notes_and_times():
    spec = spectrum_of({1: 1.0, -2: 0.6, 3: 0.3 * np.exp(0.4j)})
    mp = MelodyParams(voices=3, bar_seconds=3.0)
    vs = select_voices(spec, mp)
    with tmp() as d:
        p = write_midi(Path(d) / "m.mid", vs, mp, bars=2)
        assert [x.name for x in Path(d).iterdir()] == ["m.mid"]
        fmt, ntr, div, tempo, ev = parse_midi(p.read_bytes())
    assert (fmt, ntr, div) == (0, 1, 480) and tempo == 750000            # 3 с / 4 четверти
    ons = [(t, s & 0x0F, d[0], d[1]) for t, s, d in ev if s & 0xF0 == 0x90]
    offs = [(t, s & 0x0F, d[0]) for t, s, d in ev if s & 0xF0 == 0x80]
    assert len(ons) == len(offs) == 2 * (1 + 2 + 3)                      # 2 такта × Σ|k|
    for v in vs:
        got = sorted(t for t, ch, n, vel in ons if n == v.midi)
        want = sorted(b * 1920 + int(round(tt * 1920)) for b in range(2) for tt in v.times)
        assert got == want, v.k
        chans = {ch for t, ch, n, vel in ons if n == v.midi}
        assert chans == ({1} if v.k > 0 else {0})
    assert all(1 <= vel <= 127 for *_, vel in ons)
    assert ev[-1][0] >= max(t for t, *_ in ons)                          # конец после последней ноты


def test_midi_errors():
    with tmp() as d:
        with pytest.raises(AudioError):
            write_midi(Path(d) / "x.mid", [], MP, 1)
        (Path(d) / "f").write_text("x")
        spec = spectrum_of({1: 1.0})
        with pytest.raises(AudioError):
            write_midi(Path(d) / "f" / "x.mid", select_voices(spec, MP), MP, 1)


# ---------- экспорт: MIDI, WAV, MP4 ----------
def _job(kind, d, name=None, **kw):
    from fourier_atlas.export import ExportJob
    z, spec = make_sample("Звезда", 512)
    base = dict(kind=kind, path=str(Path(d) / (name or f"out.{kind}")), spectrum=spec.copy(), target=z,
                width=160, height=120, fps=10, duration=4.0, supersample=1, bar_seconds=2.0, voices=5)
    base.update(kw)
    return ExportJob(**base)


def test_job_mid_wav_mp4_melody_sizes_and_bars():
    import wave
    from fourier_atlas.export import run_export
    with tmp() as d:
        j = _job("mid", d)
        assert j.bars == 2 and j.seconds == 4.0 and j.n_frames == 1
        p = run_export(j)
        fmt, ntr, div, tempo, ev = parse_midi(Path(p).read_bytes())
        assert tempo == 500000                                           # 2 с на такт → 0,5 с на четверть
        w = run_export(_job("wav", d, name="m.wav", audio="melody"))
        with wave.open(str(w)) as f:
            assert f.getnframes() == round(4.0 * 44100) and f.getnchannels() == 2
        assert sorted(x.name for x in Path(d).iterdir()) == ["m.wav", "out.mid"]


def test_mp4_melody_video_has_audio_and_highlight_frames():
    import subprocess
    import imageio_ffmpeg
    from fourier_atlas.export import run_export
    import imageio.v2 as imageio
    with tmp() as d:
        j = _job("mp4", d, audio="melody", fps=10, duration=2.0, bar_seconds=2.0)
        assert j.n_frames == 20
        p = run_export(j)
        info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(p)],
                              capture_output=True, text=True).stderr
        assert "Video: h264" in info and "Audio: aac" in info and "Duration: 00:00:02.0" in info
        assert sorted(x.name for x in Path(d).iterdir()) == ["out.mp4"]
        r = imageio.get_reader(str(p), format="FFMPEG")
        frames = [r.get_data(i) for i in range(0, 20, 3)]
        r.close()
        assert len({f.tobytes() for f in frames}) > 3                    # кадры меняются (анимация)


def test_highlight_adds_primitives_only_for_active_voices():
    from fourier_atlas.export import Style, build_primitives, fit_viewport
    z, spec = make_sample("Звезда", 512)
    ctrl = AnimationController(spec, max_epicycles=5, path_samples=512)
    view = fit_viewport(400, 300, z, ctrl.path, 0.08)
    base = build_primitives(ctrl.frame(), view, z, Style())
    lit = build_primitives(ctrl.frame(), view, z, Style(), highlight={1: 1.0, 2: 0.01})
    assert len(lit) - len(base) in (1, 2)                                # слабая подсветка (<0.03) пропущена
    assert lit != base and sum(p.kind == "polyline" for p in lit) == sum(p.kind == "polyline" for p in base) + 1


def test_melody_job_validation_errors():
    from fourier_atlas.export import ExportError, validate_job
    with tmp() as d:
        for kw in (dict(voices=0), dict(bar_seconds=0.2), dict(scale="x"), dict(quantize=3), dict(root_midi=5)):
            with pytest.raises(ExportError):
                validate_job(_job("mid", d, **kw))
        j = _job("mid", d)
        j.spectrum.enabled[:] = False
        with pytest.raises(ExportError):
            validate_job(j)
        validate_job(_job("mp4", d, audio="melody"))
        validate_job(_job("wav", d, audio="melody"))
        with pytest.raises(ExportError):
            validate_job(_job("wav", d, audio="melody", f0=5.0))        # WAV использует параметры звука
        validate_job(_job("mid", d, f0=5.0))                             # MIDI от f0 не зависит


def test_cli_audio_end_to_end_from_geojson_file():
    import json, wave
    from fourier_atlas.cli_audio import main
    ring = [[10, 40], [14, 40], [15, 43], [12, 46], [9, 44], [10, 40]]
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"NAME": "Testland", "NAME_RU": "Тестландия", "ISO_A3_EH": "TST"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[ring], [[[0, 0], [1, 0], [1, 1], [0, 0]]]]}}]}
    with tmp() as d:
        f = Path(d) / "c.geojson"
        f.write_text(json.dumps(gj))
        out, mid = Path(d) / "o.wav", Path(d) / "o.mid"
        assert main(["--file", str(f), "--country", "Testland", "--mode", "melody", "--seconds", "8",
                     "--out", str(out), "--midi", str(mid)]) == 0
        with wave.open(str(out)) as w:
            assert w.getnframes() == 2 * 4 * 44100
        assert mid.read_bytes()[:4] == b"MThd"
        assert main(["--file", str(f), "--country", "TST", "--mode", "timbre", "--seconds", "1",
                     "--out", str(Path(d) / "t.wav")]) == 0
        assert main(["--file", str(f), "--mode", "melody", "--seconds", "4", "--out", str(Path(d) / "x.wav")]) == 0
        assert main(["--country", "Nowhere", "--out", str(out)]) == 1
