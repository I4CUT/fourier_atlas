"""Пользовательские настройки (JSON). Без Qt.

Файл: %APPDATA%\\FourierAtlas\\settings.json (Windows), $XDG_CONFIG_HOME/fourier_atlas или
~/.config/fourier_atlas (иначе); каталог можно переопределить переменной FOURIER_ATLAS_HOME.
Повреждённый файл не ломает запуск: берутся значения по умолчанию, ошибка пишется в журнал.
"""
from __future__ import annotations
import copy
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULTS = {
    "last_dir": "", "project_dir": "", "export_dir": "",
    "window_geometry": "", "window_state": "",
    "country": "", "projection": "auto",
    "max_epicycles": 100, "speed": 0.08, "trail_fraction": 1.0,
    "show_circles": True, "show_vectors": True, "show_target": True,
    "export": {"width": 1280, "height": 720, "fps": 30, "duration": 10.0, "supersample": 2},
    "audio": {"f0": 110.0, "stereo": True, "volume": 0.7, "steps": 24, "step_seconds": 0.6,
              "voices": 8, "bar_seconds": 4.0, "scale": "minor_pentatonic", "root": 57, "quantize": 0},
}
# ключ -> (тип, минимум, максимум) / допустимые значения
SCHEMA = {
    "max_epicycles": (int, 0, 1024), "speed": (float, -10.0, 10.0), "trail_fraction": (float, 1e-6, 1.0),
    "projection": ("choice", ("auto", "equirectangular", "plate_carree")),
    "export.width": (int, 16, 8192), "export.height": (int, 16, 8192), "export.fps": (int, 1, 120),
    "export.duration": (float, 0.2, 600.0), "export.supersample": (int, 1, 4),
    "audio.f0": (float, 20.0, 2000.0), "audio.volume": (float, 0.0, 1.0),
    "audio.steps": (int, 1, 200), "audio.step_seconds": (float, 0.1, 10.0),
    "audio.voices": (int, 1, 32), "audio.bar_seconds": (float, 1.0, 30.0), "audio.root": (int, 24, 84),
    "audio.quantize": ("choice", (0, 4, 8, 16, 32)),
    "audio.scale": ("choice", ("minor_pentatonic", "major_pentatonic", "natural_minor", "major", "chromatic")),
}


def config_dir() -> Path:
    env = os.environ.get("FOURIER_ATLAS_HOME")
    if env:
        return Path(env)
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "FourierAtlas"
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "fourier_atlas"


def _valid(rule, v) -> bool:
    if rule[0] == "choice":
        return v in rule[1]
    typ, lo, hi = rule
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    if typ is int and not float(v).is_integer():
        return False
    return lo <= v <= hi


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config_dir() / "settings.json"
        self.data = copy.deepcopy(DEFAULTS)

    @classmethod
    def load(cls, path=None) -> "Settings":
        s = cls(path)
        try:
            raw = json.loads(s.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("корневой элемент должен быть объектом")
        except FileNotFoundError:
            return s
        except (OSError, ValueError) as e:
            log.warning("Настройки не прочитаны (%s): использую значения по умолчанию", e)
            return s
        for key, default in DEFAULTS.items():
            if key not in raw:
                continue
            if isinstance(default, dict):
                if isinstance(raw[key], dict):
                    for sub, subdef in default.items():
                        s._accept(f"{key}.{sub}", raw[key].get(sub, subdef), subdef, s.data[key], sub)
                continue
            s._accept(key, raw[key], default, s.data, key)
        return s

    def _accept(self, dotted, value, default, target, key):
        if type(default) is bool:
            ok = isinstance(value, bool)
        elif isinstance(default, str):
            ok = isinstance(value, str)
        else:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok and dotted in SCHEMA:
            ok = _valid(SCHEMA[dotted], value)
        if ok:
            target[key] = type(default)(value) if isinstance(default, (int, float)) and not isinstance(default, bool) else value
        else:
            log.warning("Настройка %s имеет недопустимое значение %r — использую %r", dotted, value, default)

    def get(self, key, default=None):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key, value) -> None:
        parts = key.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(key)
        if key in SCHEMA and not _valid(SCHEMA[key], value):
            raise ValueError(f"Недопустимое значение настройки {key}: {value!r}")
        node[parts[-1]] = value

    def save(self) -> bool:
        tmp = self.path.with_suffix(".json.part")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
            return True
        except OSError as e:
            log.error("Не удалось сохранить настройки: %s", e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False
