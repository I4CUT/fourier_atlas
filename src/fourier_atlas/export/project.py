"""Проект Fourier Atlas (JSON): контур, правки спектра, параметры анимации.

Контур хранится в проекте целиком (нормированный, N точек), поэтому проект открывается
без базы стран и без исходных файлов. Спектр хранится «разреженно»: только изменённые
коэффициенты и отключённые гармоники — исходный спектр восстанавливается FFT из контура.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import numpy as np

from ..core import Spectrum, compute_spectrum

FORMAT = "fourier-atlas-project"
VERSION = 1


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


class ProjectError(Exception):
    """Проект не удалось прочитать/записать (текст — для пользователя)."""


@dataclass
class ProjectState:
    name: str
    z: np.ndarray                                   # нормированный контур (complex)
    edits: dict = field(default_factory=dict)       # k -> complex
    disabled: list = field(default_factory=list)    # список k
    max_epicycles: int | None = 100
    speed: float = 0.08
    trail_fraction: float = 1.0
    phase: float = 0.0
    source: dict = field(default_factory=dict)      # справочно: имя региона, часть, проекция


def build_state(name: str, z, spec: Spectrum, max_epicycles, speed, trail_fraction, phase,
                source: dict | None = None) -> ProjectState:
    mask = spec.edited_mask()
    return ProjectState(
        name=name, z=np.asarray(z, complex),
        edits={int(k): complex(c) for k, c, m in zip(spec.freqs, spec.edited, mask) if m},
        disabled=[int(k) for k in spec.freqs[~spec.enabled]],
        max_epicycles=None if max_epicycles is None else int(max_epicycles),
        speed=float(speed), trail_fraction=float(trail_fraction), phase=float(phase),
        source=dict(source or {}))


def spectrum_from_state(st: ProjectState) -> Spectrum:
    spec = compute_spectrum(st.z)
    for k, c in st.edits.items():
        if k not in spec._index:
            raise ProjectError(f"В проекте есть правка несуществующей гармоники k={k}")
        spec.set_coeff(k, c)
    for k in st.disabled:
        if k not in spec._index:
            raise ProjectError(f"В проекте отключена несуществующая гармоника k={k}")
        spec.set_enabled(k, False)
    return spec


def save_project(path, st: ProjectState) -> Path:
    from .. import __version__
    path = Path(path)
    obj = {
        "format": FORMAT, "version": VERSION, "app_version": __version__,
        "name": st.name, "source": st.source,
        "contour": [[float(c.real), float(c.imag)] for c in st.z],
        "spectrum": {"edited": [{"k": k, "re": c.real, "im": c.imag} for k, c in sorted(st.edits.items())],
                     "disabled": sorted(st.disabled)},
        "view": {"max_epicycles": st.max_epicycles, "speed": st.speed,
                 "trail_fraction": st.trail_fraction, "phase": st.phase},
    }
    tmp = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        _safe_unlink(tmp)
        raise ProjectError(f"Не удалось сохранить проект: {e}")
    return path


def _num(v, what, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ProjectError(f"Некорректное значение: {what}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ProjectError(f"Значение вне допустимых пределов: {what}")
    return float(v)


def load_project(path) -> ProjectState:
    p = Path(path)
    try:
        obj = json.loads(p.read_text(encoding="utf-8-sig"))
    except OSError as e:
        raise ProjectError(f"Не удалось прочитать файл: {e}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProjectError(f"Файл не является корректным JSON: {e}")
    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise ProjectError("Это не проект Fourier Atlas")
    ver = obj.get("version")
    if not isinstance(ver, int) or ver < 1:
        raise ProjectError("Не указана версия проекта")
    if ver > VERSION:
        raise ProjectError(f"Проект создан более новой версией программы (формат v{ver}, поддерживается v{VERSION})")
    try:
        pts = np.asarray(obj["contour"], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8 or not np.all(np.isfinite(pts)):
            raise ValueError
        z = pts[:, 0] + 1j * pts[:, 1]
        spec = obj.get("spectrum", {})
        edits = {}
        for e in spec.get("edited", []):
            k = e["k"]
            if isinstance(k, bool) or not isinstance(k, int):
                raise ProjectError("Некорректный номер гармоники в правках")
            edits[k] = complex(_num(e["re"], "re"), _num(e["im"], "im"))
        disabled = []
        for k in spec.get("disabled", []):
            if isinstance(k, bool) or not isinstance(k, int):
                raise ProjectError("Некорректный номер отключённой гармоники")
            disabled.append(k)
        view = obj.get("view", {})
        me = view.get("max_epicycles", 100)
        if me is not None:
            me = int(_num(me, "число эпициклов", 0, 1024))
        st = ProjectState(
            name=str(obj.get("name", p.stem)), z=z, edits=edits, disabled=disabled, max_epicycles=me,
            speed=_num(view.get("speed", 0.08), "скорость", -10, 10),
            trail_fraction=_num(view.get("trail_fraction", 1.0), "длина следа", 1e-6, 1.0),
            phase=_num(view.get("phase", 0.0), "фаза", 0.0, 1.0) % 1.0,
            source=obj.get("source", {}) if isinstance(obj.get("source", {}), dict) else {})
    except ProjectError:
        raise
    except (KeyError, TypeError, ValueError):
        raise ProjectError("Файл проекта повреждён: не хватает данных или они некорректны")
    spectrum_from_state(st)                                   # проверка согласованности правок
    return st
