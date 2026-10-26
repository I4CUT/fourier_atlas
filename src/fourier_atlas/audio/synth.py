"""Синтез звука из комплексного ряда Фурье контура.

Модель (стерео «XY»): контур z(t) = Σ c_k·exp(2πi·k·t) проигрывается как периодический
сигнал с основной частотой f0:   L(τ) = Re z(f0·τ),   R(τ) = Im z(f0·τ).
Гармоника k звучит на частоте |k|·f0 с громкостью |c_k| и фазой arg c_k. На осциллографе в
режиме XY (L по X, R по Y) рисуется исходный контур. Постоянная составляющая (k = 0) —
положение центра контура — не звучит. Моно: (Re z + Im z)/2.

Точность: подбирается целое число периодов P и длина L = round(P·sr/f0), так что гармоника k
попадает ровно в бин k·P обратного БПФ — один ifft даёт точный сигнал без накопления ошибок
фазы, а буфер зацикливается без щелчка. Фактическая основная частота f0' = P·sr/L отличается
от заданной менее чем на ~1e-5 относительно.
Гармоники с частотой выше max_freq (и выше 0.49·sr) отбрасываются — без алиасинга.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
from pathlib import Path
import wave
import numpy as np

from ..core import Spectrum, build_chain


class AudioError(Exception):
    """Недопустимые параметры звука (текст — для пользователя)."""


@dataclass(frozen=True)
class AudioParams:
    f0: float = 110.0
    sample_rate: int = 44100
    stereo: bool = True
    max_freq: float = 20000.0
    peak: float = 0.8              # пиковый уровень после нормировки (0..1)
    fade_ms: float = 10.0          # плавное нарастание/затухание (не для зацикленных буферов)

    def validate(self) -> None:
        if not (20.0 <= self.f0 <= 2000.0):
            raise AudioError("Основная частота должна быть от 20 до 2000 Гц")
        if self.sample_rate not in (22050, 32000, 44100, 48000):
            raise AudioError("Частота дискретизации: 22050, 32000, 44100 или 48000 Гц")
        if not (0.0 < self.peak <= 1.0):
            raise AudioError("Пиковый уровень должен быть в (0, 1]")
        if not (1000.0 <= self.max_freq <= 24000.0):
            raise AudioError("Верхняя граница частот: 1000–24000 Гц")


@dataclass
class Cycle:
    wave: np.ndarray        # (L, C) float32 — целое число периодов, не нормирован, C = 2 или 1
    f0: float               # фактическая основная частота
    periods: int
    n_harmonics: int        # сколько гармоник (k != 0) реально звучит


def max_harmonic(p: AudioParams) -> int:
    """Наибольший номер гармоники, который звучит (до max_freq и до 0.49·sr)."""
    return int(min(p.max_freq, 0.49 * p.sample_rate) // p.f0)


def audible_chain(spec: Spectrum, p: AudioParams, max_rotating: int | None = None):
    """(freqs, coeffs) звучащих гармоник: включённые, k != 0, |k| <= max_harmonic, затем top-N по |c|."""
    kmax = max_harmonic(p)
    f, c = spec.active()
    m = (f != 0) & (np.abs(f) <= kmax)
    f, c = f[m], c[m]
    if max_rotating is not None:
        if max_rotating < 0:
            raise AudioError("Число гармоник не может быть отрицательным")
        order = np.lexsort((np.abs(f), -np.abs(c)))[:max_rotating]
        f, c = f[order], c[order]
    return f, c


def synth_cycle(freqs, coeffs, p: AudioParams, min_seconds: float = 1.0) -> Cycle:
    """Целое число периодов (>= min_seconds), НЕ нормированное."""
    p.validate()
    P = max(1, int(round(p.f0 * min_seconds)))
    L = int(round(P * p.sample_rate / p.f0))
    f0 = P * p.sample_rate / L
    freqs = np.asarray(freqs, int)
    coeffs = np.asarray(coeffs, complex)
    keep = (freqs != 0) & (np.abs(freqs) * f0 <= min(p.max_freq, 0.49 * p.sample_rate))
    freqs, coeffs = freqs[keep], coeffs[keep]
    X = np.zeros(L, complex)
    np.add.at(X, (freqs * P) % L, coeffs * L)
    z = np.fft.ifft(X)                                   # z[n] = Σ c_k·exp(2πi·k·P·n/L)
    w = np.column_stack([z.real, z.imag]) if p.stereo else (0.5 * (z.real + z.imag))[:, None]
    return Cycle(w.astype(np.float64), f0, P, int(len(freqs)))


def _peak(w: np.ndarray) -> float:
    return float(np.max(np.abs(w))) if w.size else 0.0


def _normalized(w: np.ndarray, peak: float, reference: float | None = None) -> np.ndarray:
    ref = _peak(w) if reference is None else reference
    return (w * (peak / ref)).astype(np.float32) if ref > 1e-300 else np.zeros_like(w, np.float32)


def _tile(cycle: np.ndarray, start: int, n: int) -> np.ndarray:
    return cycle[(start + np.arange(n)) % len(cycle)]


def _fade(w: np.ndarray, p: AudioParams, fade_in=True, fade_out=True) -> np.ndarray:
    n = min(int(p.sample_rate * p.fade_ms / 1000), len(w) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        if fade_in:
            w[:n] *= ramp
        if fade_out:
            w[-n:] *= ramp[::-1]
    return w


def loop_buffer(spec: Spectrum, p: AudioParams, seconds: float = 1.5,
                max_epicycles: int | None = None) -> np.ndarray:
    """Буфер для бесконечного проигрывания: целое число периодов, без фейдов — зацикливается без щелчка."""
    f, c = audible_chain(spec, p, max_epicycles)
    return _normalized(synth_cycle(f, c, p, seconds).wave, p.peak)


def timbre_buffer(spec: Spectrum, p: AudioParams, seconds: float,
                  max_epicycles: int | None = None) -> np.ndarray:
    """Тембр контура точной длительности (round(seconds·sr) отсчётов) с короткими фейдами."""
    if seconds <= 0:
        raise AudioError("Длительность должна быть положительной")
    f, c = audible_chain(spec, p, max_epicycles)
    cyc = synth_cycle(f, c, p, min(seconds, 1.0))
    n = int(round(seconds * p.sample_rate))
    return _fade(_normalized(_tile(cyc.wave, 0, n), p.peak, _peak(cyc.wave)), p)


def buildup_schedule(total: int, steps: int = 24) -> list[int]:
    """Возрастающие числа эпициклов 1..total (примерно геометрически), последнее == total."""
    if total < 1:
        return []
    steps = max(1, steps)
    if total <= steps:
        return list(range(1, total + 1))
    vals = np.unique(np.round(np.geomspace(1, total, steps)).astype(int))
    return [int(v) for v in vals]


def buildup_buffer(spec: Spectrum, p: AudioParams, schedule: list[int], step_seconds: float = 0.6,
                   crossfade_ms: float = 25.0) -> np.ndarray:
    """«Сходимость на слух»: шаг i звучит с schedule[i] самыми крупными гармониками.

    Все шаги нормируются ОДНИМ множителем (по максимальному пику среди шагов), поэтому слышно
    реальное нарастание насыщенности. Фаза глобальная (непрерывная), стыки — перекрёстное затухание.
    """
    p.validate()
    if not schedule:
        raise AudioError("Нет гармоник для наращивания")
    n_step = int(round(step_seconds * p.sample_rate))
    n_x = min(int(p.sample_rate * crossfade_ms / 1000), n_step // 2)
    cycles = []
    for n in schedule:
        f, c = audible_chain(spec, p, n)
        cycles.append(synth_cycle(f, c, p, 0.2).wave)
    ref = max(_peak(w) for w in cycles)
    out = np.zeros((len(schedule) * n_step + n_x, cycles[0].shape[1]), np.float64)
    for i, cyc in enumerate(cycles):
        seg = _tile(cyc, i * n_step, n_step + n_x).copy()
        if i > 0 and n_x:
            seg[:n_x] *= np.linspace(0, 1, n_x)[:, None]
        if i < len(cycles) - 1 and n_x:
            seg[-n_x:] *= np.linspace(1, 0, n_x)[:, None]
        out[i * n_step: i * n_step + n_step + n_x] += seg
    out = out[: len(schedule) * n_step]
    return _fade(_normalized(out, p.peak, ref), p, fade_in=True, fade_out=True)


# ---------------- вспомогательное ----------------
def note_name(freq: float) -> str:
    midi = int(round(69 + 12 * math.log2(freq / 440.0)))
    return f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][midi % 12]}{midi // 12 - 1}"


def to_int16(w: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(w, -1.0, 1.0) * 32767).astype("<i2")


def write_wav(path, w: np.ndarray, sample_rate: int) -> Path:
    """PCM 16 бит, моно/стерео; запись атомарная."""
    path = Path(path)
    w = np.asarray(w)
    if w.ndim != 2 or w.shape[1] not in (1, 2):
        raise AudioError("Ожидался массив (отсчёты, каналы) с 1 или 2 каналами")
    tmp = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(tmp), "wb") as f:
            f.setnchannels(w.shape[1])
            f.setsampwidth(2)
            f.setframerate(int(sample_rate))
            f.writeframes(to_int16(w).tobytes())
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise AudioError(f"Не удалось записать WAV: {e}")
    return path
