"""Задания экспорта (PNG, SVG, GIF, MP4): выполняются в фоновом потоке, потоково и с отменой.

Поток получает ИЗОЛИРОВАННУЮ копию данных (копия спектра, свой контроллер анимации),
кадры пишутся по одному (память не растёт с числом кадров), результат сначала
пишется во временный файл и атомарно переименовывается — недописанных файлов не остаётся.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import numpy as np

from ..animation import AnimationController
from ..audio import (AudioError, AudioParams, MelodyParams, audible_chain, buildup_buffer, buildup_schedule,
                     event_strength, render_melody, select_voices, timbre_buffer, write_midi, write_wav)
from .scene import Style, build_primitives, fit_viewport, render_raster, render_svg

log = logging.getLogger(__name__)
KINDS = ("png", "svg", "gif", "mp4", "wav", "mid")
AUDIO_MODES = ("none", "timbre", "buildup", "melody")
MAX_FRAMES = 20000
MAX_GIF_SIDE = 1920


class ExportError(Exception):
    """Ошибка экспорта (текст предназначен для пользователя)."""


class ExportCancelled(Exception):
    pass


@dataclass
class ExportJob:
    kind: str
    path: str
    spectrum: object                      # независимая копия Spectrum
    target: np.ndarray                    # исходный (нормированный) контур
    max_epicycles: int | None = None
    trail_fraction: float = 1.0
    phase: float = 0.0                    # для png/svg
    width: int = 1280
    height: int = 720
    fps: int = 30
    duration: float = 10.0                # секунд на один полный оборот (gif/mp4)
    supersample: int = 2
    style: Style = field(default_factory=Style)
    path_samples: int = 1024
    # звук: для mp4 — дорожка к видео; для wav — обязателен 'timbre' или 'buildup'
    audio: str = "none"
    f0: float = 110.0
    audio_stereo: bool = True
    sample_rate: int = 44100
    steps: int = 24                       # «наращивание»: число шагов
    step_seconds: float = 0.6             # «наращивание»: длительность шага
    revolution_seconds: float = 4.0       # «наращивание» в видео: секунд на оборот контура
    # мелодия (audio == 'melody'; для kind == 'mid' всегда)
    voices: int = 8
    bar_seconds: float = 4.0              # один оборот контура = один такт
    scale: str = "minor_pentatonic"
    root_midi: int = 57
    quantize: int = 0

    def audio_params(self) -> AudioParams:
        return AudioParams(f0=self.f0, sample_rate=self.sample_rate, stereo=self.audio_stereo)

    def melody_params(self) -> MelodyParams:
        return MelodyParams(voices=self.voices, bar_seconds=self.bar_seconds, scale=self.scale,
                            root_midi=self.root_midi, quantize=self.quantize)

    @property
    def is_melody(self) -> bool:
        return self.kind == "mid" or (self.audio == "melody" and self.kind in ("mp4", "wav"))

    @property
    def bars(self) -> int:
        return max(1, int(round(self.duration / self.bar_seconds)))

    def schedule(self) -> list:
        """Числа эпициклов по шагам наращивания (только звучащие гармоники)."""
        try:
            total = len(audible_chain(self.spectrum, self.audio_params())[0])
        except AudioError:
            return []
        return buildup_schedule(total, self.steps)

    @property
    def seconds(self) -> float:
        """Длительность звука/видео в секундах (для gif/mp4/wav)."""
        if self.is_melody:
            return self.bars * self.bar_seconds
        if self.audio == "buildup" and self.kind in ("mp4", "wav"):
            return len(self.schedule()) * self.step_seconds
        return self.duration

    @property
    def n_frames(self) -> int:
        if self.kind in ("png", "svg", "wav", "mid"):
            return 1
        return max(1, int(round(self.fps * self.seconds)))


def validate_job(job: ExportJob) -> None:
    if job.kind not in KINDS:
        raise ExportError(f"Неизвестный формат: {job.kind}")
    if job.kind in ("wav", "mid"):
        if not (0.2 <= job.seconds <= 600):
            raise ExportError("Длительность должна быть от 0,2 до 600 секунд")
        if not str(job.path):
            raise ExportError("Не указан путь к файлу")
        _validate_audio(job)
        return
    if not (16 <= job.width <= 8192 and 16 <= job.height <= 8192):
        raise ExportError("Размер изображения должен быть от 16 до 8192 пикселей")
    if job.kind in ("gif", "mp4"):
        if not (1 <= job.fps <= 120):
            raise ExportError("Частота кадров должна быть от 1 до 120")
        if not (0.2 <= job.seconds <= 600):
            raise ExportError("Длительность должна быть от 0,2 до 600 секунд")
        if job.n_frames > MAX_FRAMES:
            raise ExportError(f"Слишком много кадров ({job.n_frames}); максимум {MAX_FRAMES}")
    if job.kind == "gif" and max(job.width, job.height) > MAX_GIF_SIDE:
        raise ExportError(f"Для GIF размер не более {MAX_GIF_SIDE} пикселей по стороне")
    if not (1 <= job.supersample <= 4):
        raise ExportError("Сглаживание (supersample) должно быть от 1 до 4")
    if not str(job.path):
        raise ExportError("Не указан путь к файлу")
    _validate_audio(job)


def _validate_audio(job: ExportJob) -> None:
    mode = "melody" if job.kind == "mid" else job.audio
    if mode not in AUDIO_MODES:
        raise ExportError(f"Неизвестный режим звука: {mode}")
    if job.kind == "wav" and mode == "none":
        raise ExportError("Для WAV выберите режим звука: тембр, наращивание или мелодия")
    if mode == "none" or job.kind not in ("mp4", "wav", "mid"):
        return
    if mode == "melody":
        try:
            if job.kind != "mid":
                job.audio_params().validate()
            job.melody_params().validate()
        except AudioError as e:
            raise ExportError(str(e))
        if not select_voices(job.spectrum, job.melody_params()):
            raise ExportError("Нет гармоник для мелодии (все отключены или слишком быстрые для такта)")
        if job.seconds > 600:
            raise ExportError("Слишком длинная мелодия (больше 600 секунд)")
        return
    try:
        job.audio_params().validate()
    except AudioError as e:
        raise ExportError(str(e))
    if mode == "buildup":
        if not (1 <= job.steps <= 200) or not (0.1 <= job.step_seconds <= 10):
            raise ExportError("Наращивание: шагов 1–200, длительность шага 0,1–10 секунд")
        if not job.schedule():
            raise ExportError("Нет звучащих гармоник для наращивания (все отключены или выше 20 кГц)")
        if job.seconds > 600:
            raise ExportError("Слишком длинное наращивание (больше 600 секунд)")
        if job.kind == "mp4" and not (0.5 <= job.revolution_seconds <= 120):
            raise ExportError("Время оборота в видео: 0,5–120 секунд")


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.part{path.suffix}")


def run_export(job: ExportJob, progress=None, cancel=None) -> Path:
    """Выполняет задание. progress(done, total); cancel() -> bool. Возвращает путь к файлу."""
    validate_job(job)
    path = Path(job.path)
    if job.kind == "wav":
        return _run_wav(job, path, progress, cancel)
    if job.kind == "mid":
        return _run_mid(job, path, progress)
    if job.kind == "mp4":                                   # yuv420p требует чётных размеров
        job.width -= job.width % 2
        job.height -= job.height % 2
    buildup = job.kind == "mp4" and job.audio == "buildup"
    melody = job.kind == "mp4" and job.audio == "melody"
    sched = job.schedule() if buildup else []
    spec_view, voices, chain_idx = job.spectrum, [], {}
    if melody:                                              # в видео — ровно те эпициклы, что играют
        voices = select_voices(job.spectrum, job.melody_params())
        spec_view = job.spectrum.copy()
        spec_view.enabled[:] = False
        for k in [0] + [v.k for v in voices]:
            spec_view.set_enabled(k, True)
    ctrl = AnimationController(spec_view,
                               max_epicycles=sched[-1] if buildup else (None if melody else job.max_epicycles),
                               path_samples=job.path_samples, trail_fraction=job.trail_fraction, speed=0)
    view = fit_viewport(job.width, job.height, job.target, ctrl.path, job.style.margin)
    size = (job.width, job.height)
    n = job.n_frames
    if melody:
        chain_idx = {int(k): i - 0 for i, k in enumerate(ctrl.chain.freqs)}

    def frame_image(i: int):
        if cancel and cancel():
            raise ExportCancelled()
        if buildup:                                         # число эпициклов растёт синхронно со звуком
            t = i / job.fps
            n_ep = sched[min(int(t / job.step_seconds), len(sched) - 1)]
            if ctrl.max_epicycles != n_ep:
                ctrl.max_epicycles = n_ep
            ctrl.seek((t / job.revolution_seconds) % 1.0)
        elif melody:                                        # один оборот = один такт, как в звуке
            ph = ((i / job.fps) / job.bar_seconds) % 1.0
            ctrl.seek(ph)
        else:
            ctrl.seek(job.phase if n == 1 else i / n)
        highlight = None
        if melody:
            highlight = {chain_idx[k]: s_ for k, s_ in
                         event_strength(voices, ctrl.phase, job.bar_seconds).items() if k in chain_idx}
        prims = build_primitives(ctrl.frame(), view, job.target, job.style, highlight)
        img = render_raster(prims, size, job.style.background, job.supersample)
        if progress:
            progress(i + 1, n)
        return img

    tmp = _tmp_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if job.kind == "png":
            frame_image(0).save(tmp, format="PNG")
        elif job.kind == "svg":
            ctrl.seek(job.phase)
            prims = build_primitives(ctrl.frame(), view, job.target, job.style)
            tmp.write_text(render_svg(prims, size, job.style.background), encoding="utf-8")
            if progress:
                progress(1, 1)
        elif job.kind == "gif":
            gen = (frame_image(i) for i in range(n))
            first = next(gen)
            first.save(tmp, format="GIF", save_all=True, append_images=gen,
                       duration=max(10, round(1000 / job.fps)), loop=0, optimize=False)
        else:
            audio_path = None
            try:
                if job.audio != "none":
                    audio_path = _make_audio_file(job, tmp.with_suffix(".wav"), n / job.fps)
                _write_mp4(tmp, (frame_image(i) for i in range(n)), job.fps, audio_path)
            finally:
                if audio_path is not None:
                    _cleanup(Path(audio_path))
        os.replace(tmp, path)
    except ExportCancelled:
        _cleanup(tmp)
        raise
    except ExportError:
        _cleanup(tmp)
        raise
    except OSError as e:
        _cleanup(tmp)
        log.exception("Ошибка записи файла экспорта")
        raise ExportError(f"Не удалось записать файл: {e}")
    except Exception as e:                                  # noqa: BLE001
        _cleanup(tmp)
        log.exception("Ошибка экспорта")
        raise ExportError(f"Ошибка экспорта: {e}")
    log.info("Экспорт %s: %s (%d кадров)", job.kind, path, n)
    return path


def _fit_length(w, n: int):
    if len(w) < n:
        w = np.vstack([w, np.zeros((n - len(w), w.shape[1]), w.dtype)])
    return w[:n]


def _audio_buffer(job: ExportJob, seconds: float):
    """Звук ровно на seconds секунд (round(seconds·sr) отсчётов)."""
    p = job.audio_params()
    n = int(round(seconds * p.sample_rate))
    if job.audio == "melody":
        return _fit_length(render_melody(job.spectrum, job.melody_params(), p,
                                         max(1, int(np.ceil(seconds / job.bar_seconds)))), n)
    if job.audio == "buildup":
        w = buildup_buffer(job.spectrum, p, job.schedule(), job.step_seconds)
        if len(w) < n:
            w = np.vstack([w, np.zeros((n - len(w), w.shape[1]), w.dtype)])
        return w[:n]
    return timbre_buffer(job.spectrum, p, seconds, job.max_epicycles)


def _make_audio_file(job: ExportJob, wav_path: Path, seconds: float) -> Path:
    try:
        return write_wav(wav_path, _audio_buffer(job, seconds), job.sample_rate)
    except AudioError as e:
        raise ExportError(str(e))


def _run_mid(job: ExportJob, path: Path, progress) -> Path:
    mp = job.melody_params()
    try:
        write_midi(path, select_voices(job.spectrum, mp), mp, job.bars)
    except AudioError as e:
        raise ExportError(str(e))
    if progress:
        progress(1, 1)
    log.info("Экспорт mid: %s (%d тактов)", path, job.bars)
    return path


def _run_wav(job: ExportJob, path: Path, progress, cancel) -> Path:
    if cancel and cancel():
        raise ExportCancelled()
    tmp = _tmp_path(path)
    try:
        _make_audio_file(job, tmp, job.seconds)
        if cancel and cancel():
            raise ExportCancelled()
        os.replace(tmp, path)
    except (ExportCancelled, ExportError):
        _cleanup(tmp)
        raise
    except OSError as e:
        _cleanup(tmp)
        log.exception("Ошибка записи WAV")
        raise ExportError(f"Не удалось записать файл: {e}")
    if progress:
        progress(1, 1)
    log.info("Экспорт wav (%s): %s", job.audio, path)
    return path


def _write_mp4(tmp: Path, images, fps: int, audio_path=None) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ExportError("Для MP4 нужны пакеты imageio и imageio-ffmpeg (pip install imageio imageio-ffmpeg)")
    try:
        extra = {"audio_path": str(audio_path), "audio_codec": "aac"} if audio_path else {}
        writer = imageio.get_writer(str(tmp), format="FFMPEG", mode="I", fps=fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p", macro_block_size=2,
                                    ffmpeg_log_level="error", **extra)
    except Exception as e:                                  # noqa: BLE001 — нет ffmpeg и т.п.
        raise ExportError(f"Не удалось запустить кодировщик MP4 (ffmpeg): {e}")
    try:
        for img in images:
            writer.append_data(np.asarray(img))
    finally:
        writer.close()


def _cleanup(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
