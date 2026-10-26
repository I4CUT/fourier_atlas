"""Экспорт в CSV: таблица коэффициентов и контуры (исходный/восстановленный)."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import numpy as np

from ..core import reconstruct


def _atomic_csv(path, header, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:      # BOM — Excel читает UTF-8
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def export_coefficients_csv(path, spec) -> Path:
    """Столбцы: k, re, im, amplitude, phase_rad, enabled, edited, original_re, original_im.
    re/im/amplitude/phase — ТЕКУЩИЕ значения (с правками); строки по возрастанию k.
    Нормировка: c_k = fft(z)[k]/N, z(t) = Σ c_k·exp(2πi·k·t)."""
    o = np.argsort(spec.freqs, kind="stable")
    edited = spec.edited_mask()
    rows = []
    for i in o:
        c, c0 = spec.edited[i], spec.original[i]
        rows.append([int(spec.freqs[i]), repr(float(c.real)), repr(float(c.imag)),
                     repr(float(abs(c))), repr(float(np.angle(c))),
                     int(spec.enabled[i]), int(edited[i]),
                     repr(float(c0.real)), repr(float(c0.imag))])
    return _atomic_csv(path, ["k", "re", "im", "amplitude", "phase_rad", "enabled", "edited",
                              "original_re", "original_im"], rows)


def export_contour_csv(path, original: np.ndarray, spec) -> Path:
    """Исходный и восстановленный (по текущему состоянию спектра) контуры в одних точках t = j/N."""
    z = np.asarray(original, complex)
    r = reconstruct(spec, len(z))
    n = len(z)
    rows = [[j, repr(j / n), repr(float(z[j].real)), repr(float(z[j].imag)),
             repr(float(r[j].real)), repr(float(r[j].imag)), repr(float(abs(z[j] - r[j])))]
            for j in range(n)]
    return _atomic_csv(path, ["index", "t", "x_original", "y_original", "x_reconstructed",
                              "y_reconstructed", "error"], rows)
