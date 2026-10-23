"""Данные для таблицы коэффициентов и графиков спектра; разбор ввода комплексных чисел."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def parse_complex(text: str) -> complex:
    """Разбор ввода пользователя.

    Прямоугольная форма: '1.5-2i', '3+0.5j', '-4', '2i', 'i'; десятичная запятая допустима.
    Полярная форма: 'r@phi' (phi в радианах) или 'r@phi°' (в градусах), r >= 0.
    """
    s = (text or "").strip().replace("−", "-").replace(",", ".").replace(" ", "")
    if not s:
        raise ValueError("Пустое значение")
    if "@" in s or "∠" in s:
        r_s, _, p_s = s.replace("∠", "@").partition("@")
        deg = p_s.endswith("°")
        try:
            r, phi = float(r_s), float(p_s.rstrip("°"))
        except ValueError:
            raise ValueError(f"Не удалось разобрать полярную запись: {text!r}")
        if r < 0:
            raise ValueError("Модуль должен быть >= 0")
        value = r * complex(math.cos(math.radians(phi) if deg else phi),
                            math.sin(math.radians(phi) if deg else phi))
    else:
        try:
            value = complex(s.replace("i", "j").replace("I", "j"))
        except ValueError:
            raise ValueError(f"Не удалось разобрать число: {text!r}")
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError("Значение должно быть конечным")
    return value


def table_order(spec, sort_by: str = "k") -> np.ndarray:
    """Индексы строк таблицы: 'k' — по возрастанию частоты; 'amp' — по убыванию |c| исходного спектра."""
    if sort_by == "k":
        return np.argsort(spec.freqs, kind="stable")
    if sort_by == "amp":
        return np.argsort(-np.abs(spec.original), kind="stable")
    raise ValueError(f"Неизвестная сортировка: {sort_by}")


@dataclass
class SpectrumPlotData:
    k: np.ndarray            # по возрастанию
    amp: np.ndarray          # |c_k| (с учётом правок)
    phase: np.ndarray        # arg c_k, рад; NaN там, где фаза не определена
    enabled: np.ndarray
    edited: np.ndarray


def spectrum_plot_data(spec, phase_tol: float = 1e-9) -> SpectrumPlotData:
    o = np.argsort(spec.freqs, kind="stable")
    c = spec.edited[o]
    amp = np.abs(c)
    phase = np.angle(c)
    phase[amp <= phase_tol * max(float(amp.max()), 1e-300)] = np.nan
    return SpectrumPlotData(spec.freqs[o], amp, phase, spec.enabled[o].copy(), spec.edited_mask()[o])
