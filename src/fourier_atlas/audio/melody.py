"""Мелодия по спектру: каждый эпицикл играет ноту, когда его вектор проходит направление «вверх».

Правила (математические):
  • Эпицикл k делает k оборотов за один оборот контура (= один «такт» длиной bar_seconds), поэтому
    гармоника k играет |k| нот за такт — полиритмия 1:2:3:…
  • Момент ноты — когда направление вектора c_k·exp(2πi·k·t) совпадает с «вверх» (arg = π/2):
        t_m = ((π/2 − arg c_k)/(2π) + m) / k  (mod 1),   m = 0 … |k|−1.
    Фазы arg c_k задают рисунок ритма: у каждой страны он свой.
  • Громкость ноты — |c_k| (относительно крупнейшей гармоники).
  • Знак k (направление вращения) — в стереопанораму: против часовой стрелки (k>0) — вправо, по часовой — влево.
Музыкальные соглашения (выбор автора, настраиваются):
  • Высота — по порядку |k| среди выбранных голосов: чем больше |k|, тем выше нота (низкие гармоники играют
    редко и в басу, высокие — часто и выше). Ноты берутся из лада (по умолчанию минорная пентатоника).
  • Берутся top-N голосов по |c_k|; голоса, чей темп |k|/bar_seconds выше max_rate нот/с, отбрасываются.
  • Тембр — простая «маримба»: три затухающих обертона.
Буфер одного такта замкнут по кругу (хвосты нот переходят в следующий такт) — зацикливается без щелчка.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
from pathlib import Path
import numpy as np

from .synth import AudioError, AudioParams, _fade

SCALES = {
    "minor_pentatonic": ("Минорная пентатоника", (0, 3, 5, 7, 10)),
    "major_pentatonic": ("Мажорная пентатоника", (0, 2, 4, 7, 9)),
    "natural_minor": ("Натуральный минор", (0, 2, 3, 5, 7, 8, 10)),
    "major": ("Мажор", (0, 2, 4, 5, 7, 9, 11)),
    "chromatic": ("Хроматика", tuple(range(12))),
}
QUANTIZE = (0, 4, 8, 16, 32)
MAX_MIDI = 108


@dataclass(frozen=True)
class MelodyParams:
    voices: int = 8
    bar_seconds: float = 4.0
    scale: str = "minor_pentatonic"
    root_midi: int = 57                # A3
    quantize: int = 0                  # 0 — без квантования; иначе число долей такта (4, 8, 16, 32)
    max_rate: float = 8.0              # предельный темп голоса, нот/с
    decay: float = 0.35                # постоянная затухания ноты, с
    tail_seconds: float = 1.2
    velocity_curve: float = 0.7
    min_rel_amp: float = 1e-4          # голоса тише этой доли от крупнейшего не играют (шум/нули спектра)

    def validate(self) -> None:
        if not (1 <= self.voices <= 32):
            raise AudioError("Число голосов должно быть от 1 до 32")
        if not (1.0 <= self.bar_seconds <= 30.0):
            raise AudioError("Длительность такта: 1–30 секунд")
        if self.scale not in SCALES:
            raise AudioError(f"Неизвестный лад: {self.scale}")
        if not (24 <= self.root_midi <= 84):
            raise AudioError("Тоника (MIDI-номер): 24–84")
        if self.quantize not in QUANTIZE:
            raise AudioError(f"Квантование: одно из {QUANTIZE}")


@dataclass(frozen=True)
class Voice:
    k: int
    coeff: complex
    degree: int                 # порядковый номер голоса по высоте (0 — самый низкий)
    midi: int
    freq: float
    velocity: float             # 0..1
    pan: float                  # -1..1
    times: tuple                # моменты нот, доли такта [0, 1)


def midi_to_freq(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def midi_for_degree(i: int, root: int, scale: str) -> int:
    steps = SCALES[scale][1]
    m = root + 12 * (i // len(steps)) + steps[i % len(steps)]
    while m > MAX_MIDI:
        m -= 12                                           # слишком высокие ноты сворачиваем на октаву вниз
    return m


def event_times(k: int, coeff: complex, quantize: int = 0) -> np.ndarray:
    """Моменты (доли такта), когда вектор c_k·e^{2πikt} направлен вверх (arg = π/2)."""
    if k == 0:
        raise ValueError("k = 0 не вращается")
    u = (math.pi / 2 - np.angle(coeff)) / (2 * math.pi)
    t = ((u + np.arange(abs(k))) / k) % 1.0
    if quantize:
        t = np.round(t * quantize) / quantize % 1.0
    return np.unique(np.round(t, 12))


def select_voices(spec, mp: MelodyParams) -> list[Voice]:
    mp.validate()
    f, c = spec.active()
    keep = (f != 0) & (np.abs(f) <= mp.max_rate * mp.bar_seconds)
    f, c = f[keep], c[keep]
    if len(f) == 0:
        return []
    amax = float(np.abs(c).max())
    if amax <= 1e-12:
        return []
    audible = np.abs(c) >= mp.min_rel_amp * amax
    f, c = f[audible], c[audible]
    order = np.lexsort((np.abs(f), -np.abs(c)))[: mp.voices]       # top-N по |c|
    f, c = f[order], c[order]
    by_pitch = np.lexsort((-np.sign(f), np.abs(f)))                  # по |k|; при равенстве k>0 раньше
    voices = []
    for degree, i in enumerate(by_pitch):
        k, ck = int(f[i]), complex(c[i])
        midi = midi_for_degree(degree, mp.root_midi, mp.scale)
        vel = float(np.clip((abs(ck) / amax) ** mp.velocity_curve, 0.15, 1.0))
        voices.append(Voice(k, ck, degree, midi, midi_to_freq(midi), vel, 0.6 if k > 0 else -0.6,
                            tuple(float(x) for x in event_times(k, ck, mp.quantize))))
    return voices


# ---------------- рендер ----------------
def _note(freq: float, vel: float, mp: MelodyParams, sr: int) -> np.ndarray:
    n = int(mp.tail_seconds * sr)
    t = np.arange(n) / sr
    env = np.exp(-t / mp.decay) * np.minimum(1.0, t / 0.004)
    w = np.sin(2 * np.pi * freq * t)
    for mult, amp, tau in ((2, 0.35, 0.5), (3, 0.12, 0.3)):
        if freq * mult < 0.49 * sr:
            w = w + amp * np.sin(2 * np.pi * freq * mult * t) * np.exp(-t / (mp.decay * tau))
    out = w * env * vel
    out[-int(0.02 * sr):] *= np.linspace(1, 0, int(0.02 * sr))     # без щелчка в конце хвоста
    return out


def render_loop(voices: list[Voice], mp: MelodyParams, ap: AudioParams) -> np.ndarray:
    """Один такт, замкнутый по кругу, float32 (n, 2|1), пик = ap.peak."""
    ap.validate()
    n_bar = int(round(mp.bar_seconds * ap.sample_rate))
    ch = 2 if ap.stereo else 1
    mix = np.zeros((n_bar, ch))
    for v in voices:
        note = _note(v.freq, v.velocity, mp, ap.sample_rate)
        reps = -(-len(note) // n_bar)
        padded = np.zeros(reps * n_bar)
        padded[: len(note)] = note
        folded = padded.reshape(reps, n_bar).sum(axis=0)                # хвосты, перешедшие в следующие такты
        starts = np.round(np.array(v.times) * n_bar).astype(int) % n_bar
        imp = np.bincount(starts, minlength=n_bar).astype(float)
        sig = np.fft.irfft(np.fft.rfft(imp) * np.fft.rfft(folded), n_bar)   # круговая свёртка: ноты во все моменты
        if ap.stereo:
            ang = (v.pan + 1) * math.pi / 4
            mix += sig[:, None] * np.array([math.cos(ang), math.sin(ang)])
        else:
            mix += sig[:, None] * 0.5 * (math.cos((v.pan + 1) * math.pi / 4) + math.sin((v.pan + 1) * math.pi / 4))
    pk = float(np.abs(mix).max())
    return (mix * (ap.peak / pk)).astype(np.float32) if pk > 1e-12 else np.zeros_like(mix, np.float32)


def render_melody(spec, mp: MelodyParams, ap: AudioParams, bars: int) -> np.ndarray:
    voices = select_voices(spec, mp)
    if not voices:
        raise AudioError("Нет гармоник для мелодии (все отключены или слишком быстрые для такта)")
    if bars < 1:
        raise AudioError("Число тактов должно быть >= 1")
    out = np.tile(render_loop(voices, mp, ap), (bars, 1))
    return _fade(out, ap)


def loop_melody(spec, mp: MelodyParams, ap: AudioParams) -> np.ndarray:
    """Один такт для бесконечного проигрывания (без фейдов); тишина, если голосов нет."""
    voices = select_voices(spec, mp)
    n = int(round(mp.bar_seconds * ap.sample_rate))
    return render_loop(voices, mp, ap) if voices else np.zeros((n, 2 if ap.stereo else 1), np.float32)


def event_strength(voices: list[Voice], phase: float, bar_seconds: float, decay: float = 0.18) -> dict:
    """Подсветка для видео: k -> сила 0..1 = exp(−время_с_последней_ноты/decay)."""
    out = {}
    for v in voices:
        since = (phase - np.array(v.times)) % 1.0
        out[v.k] = float(np.exp(-since.min() * bar_seconds / decay))
    return out


# ---------------- MIDI ----------------
def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def write_midi(path, voices: list[Voice], mp: MelodyParams, bars: int, ticks: int = 480) -> Path:
    """Standard MIDI File (формат 0): такт 4/4, темп из bar_seconds, маримба (программа 12);
    канал 0 — вращение по часовой (панорама влево), канал 1 — против (вправо)."""
    if not voices:
        raise AudioError("Нет гармоник для мелодии")
    per_bar = ticks * 4
    ev = []                                                   # (тик, порядок, байты)
    ev.append((0, 0, bytes([0xFF, 0x51, 0x03]) + int(round(mp.bar_seconds / 4 * 1e6)).to_bytes(3, "big")))
    ev.append((0, 0, bytes([0xFF, 0x58, 0x04, 4, 2, 24, 8])))
    for chn, pan in ((0, 26), (1, 102)):
        ev.append((0, 1, bytes([0xC0 | chn, 12])))
        ev.append((0, 1, bytes([0xB0 | chn, 10, pan])))
    for b in range(bars):
        for v in voices:
            chn = 1 if v.k > 0 else 0
            t = np.array(v.times)
            gaps = np.diff(np.append(t, t[0] + 1.0)) * per_bar
            dur = int(max(1, min(per_bar // 8, gaps.min() - 1)))
            vel = int(20 + 107 * v.velocity)
            for tt in t:
                tick = b * per_bar + int(round(tt * per_bar))
                ev.append((tick, 3, bytes([0x90 | chn, v.midi, vel])))
                ev.append((tick + dur, 2, bytes([0x80 | chn, v.midi, 0])))      # off раньше on в тот же тик
    last = max(t for t, _, _ in ev) + ticks
    ev.append((last, 9, bytes([0xFF, 0x2F, 0x00])))
    ev.sort(key=lambda e: (e[0], e[1]))
    body, prev = b"", 0
    for tick, _, data in ev:
        body += _vlq(tick - prev) + data
        prev = tick
    data = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + \
        ticks.to_bytes(2, "big") + b"MTrk" + len(body).to_bytes(4, "big") + body
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise AudioError(f"Не удалось записать MIDI: {e}")
    return path
