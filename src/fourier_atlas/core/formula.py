"""Ряд Фурье в аналитическом виде (текст и LaTeX). Без Qt.

Параметр t ∈ [0, 1) — доля периода; z(t) = x(t) + i·y(t).

Текстовая форма намеренно «выполнимая»: её можно вставить в Python
(i = 1j, из math/cmath: pi, exp, cos, sin) или в CAS и получить ту же кривую.

Три формы одного и того же ряда:
  комплексная   z(t) = Σ c_k · exp(2πi·k·t)
  полярная      z(t) = Σ r_k · exp(i·(2πk·t + φ_k)),  r_k = |c_k|, φ_k = arg c_k
  вещественная  x(t) = x0 + Σ_k>0 [ax_k cos(2πkt) + bx_k sin(2πkt)]
                y(t) = y0 + Σ_k>0 [ay_k cos(2πkt) + by_k sin(2πkt)]
Для вещественной формы вклад пары ±k (c_k = a+ib, c_-k = a'+ib'):
  ax = a + a',  bx = b' − b,  ay = b + b',  by = a − a'.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Term:
    k: int
    c: complex


def terms_from_chain(chain) -> list[Term]:
    """Члены ряда в порядке цепочки эпициклов (k=0 первым, далее по убыванию |c|)."""
    return [Term(int(k), complex(c)) for k, c in zip(chain.freqs, chain.coeffs)]


def terms_value(terms: list[Term], t) -> np.ndarray:
    """Прямое вычисление суммы ряда — эталон для проверок."""
    t = np.asarray(t, float)
    out = np.zeros(t.shape, complex)
    for tm in terms:
        out += tm.c * np.exp(2j * np.pi * tm.k * t)
    return out


# ---------- числа ----------
def _r(x: float, d: int) -> float:
    v = round(float(x), d)
    return 0.0 if v == 0 else v


def _num(x: float, d: int) -> str:
    return f"{_r(x, d):.{d}f}"


def _cplx(c: complex, d: int, tex: bool = False) -> str:
    """'a+b*i' (выполнимо в Python) либо 'a+bi' для LaTeX."""
    re, im = _r(c.real, d), _r(c.imag, d)
    sign = "-" if im < 0 else "+"
    return f"{re:.{d}f}{sign}{abs(im):.{d}f}" + ("i" if tex else "*i")


def _exp_arg_text(k: int) -> str:
    """Аргумент exp для частоты k (k != 0), напр. 2*pi*i*t, -2*pi*i*3*t."""
    sign = "-" if k < 0 else ""
    kk = abs(k)
    return f"{sign}2*pi*i*t" if kk == 1 else f"{sign}2*pi*i*{kk}*t"


def _exp_arg_tex(k: int) -> str:
    sign = "-" if k < 0 else ""
    kk = abs(k)
    return f"{sign}2\\pi i t" if kk == 1 else f"{sign}2\\pi i\\,{kk}t"


def _select(terms: list[Term], max_terms):
    if max_terms is None or max_terms >= len(terms):
        return list(terms), 0
    if max_terms < 1:
        raise ValueError("max_terms >= 1")
    return list(terms[:max_terms]), len(terms) - max_terms


def _tail_text(rest: int, fmt: str) -> str:
    if not rest:
        return ""
    return f" + ... ({rest} more)" if fmt == "text" else rf" + \dots\ (\text{{ещё {rest}}})"


def _signed_sum(parts: list[tuple[float, str]], d: int, const: float | None = None) -> str:
    """Сумма вида 'c0 + a*cos(..) - b*sin(..)' с аккуратными знаками; нулевые члены опускаются."""
    out = ""
    if const is not None:
        out = _num(const, d)
    for coef, body in parts:
        if _r(coef, d) == 0:
            continue
        mag = _num(abs(coef), d)
        sign = "-" if coef < 0 else "+"
        if not out:
            out = ("-" if coef < 0 else "") + f"{mag}*{body}"
        else:
            out += f" {sign} {mag}*{body}"
    return out or _num(0, d)


# ---------- комплексная ----------
def complex_series(terms: list[Term], digits: int = 4, max_terms: int | None = None,
                   fmt: str = "text", wrap: int = 3) -> str:
    sel, rest = _select(terms, max_terms)
    if not sel:
        return "z(t) = 0" if fmt == "text" else "z(t) &= 0"
    if fmt == "text":
        parts = []
        for tm in sel:
            parts.append(f"({_cplx(tm.c, digits)})" if tm.k == 0
                         else f"({_cplx(tm.c, digits)})*exp({_exp_arg_text(tm.k)})")
        return "z(t) = " + " + ".join(parts) + _tail_text(rest, "text")
    parts = [rf"\left({_cplx(tm.c, digits, tex=True)}\right)" + ("" if tm.k == 0 else rf"e^{{{_exp_arg_tex(tm.k)}}}")
             for tm in sel]
    return _latex_join("z(t)", parts, wrap) + _tail_text(rest, "latex")


# ---------- полярная ----------
def polar_series(terms: list[Term], digits: int = 4, max_terms: int | None = None,
                 fmt: str = "text", wrap: int = 3) -> str:
    sel, rest = _select(terms, max_terms)
    if not sel:
        return "z(t) = 0" if fmt == "text" else "z(t) &= 0"
    parts = []
    for tm in sel:
        r, phi = abs(tm.c), float(np.angle(tm.c))
        if _r(r, digits) == 0:
            phi = 0.0                       # фаза нулевого вектора не определена
        if fmt == "text":
            if tm.k == 0:
                parts.append(f"{_num(r, digits)}*exp(i*({_num(phi, digits)}))")
            else:
                sgn = "+" if _r(phi, digits) >= 0 else "-"
                parts.append(f"{_num(r, digits)}*exp(i*({_phase_text(tm.k)}{sgn}{_num(abs(phi), digits)}))")
        else:
            ph = f"{_phase_tex(tm.k)}{'+' if _r(phi, digits) >= 0 else '-'}{_num(abs(phi), digits)}" if tm.k else _num(phi, digits)
            parts.append(rf"{_num(r, digits)}\,e^{{i\left({ph}\right)}}")
    if fmt == "text":
        return "z(t) = " + " + ".join(parts) + _tail_text(rest, "text")
    return _latex_join("z(t)", parts, wrap) + _tail_text(rest, "latex")


def _phase_text(k: int) -> str:
    sign = "-" if k < 0 else ""
    kk = abs(k)
    return f"{sign}2*pi*t" if kk == 1 else f"{sign}2*pi*{kk}*t"


def _phase_tex(k: int) -> str:
    sign = "-" if k < 0 else ""
    kk = abs(k)
    return f"{sign}2\\pi t" if kk == 1 else f"{sign}2\\pi\\,{kk}t"


def _latex_join(lhs: str, parts: list[str], wrap: int) -> str:
    """Тело для окружения align*: переносы строк каждые wrap членов."""
    lines = [" + ".join(parts[i:i + wrap]) for i in range(0, len(parts), wrap)]
    return f"{lhs} &= " + " \\\\\n&\\quad + ".join(lines)


# ---------- вещественная ----------
@dataclass
class RealSeries:
    x0: float
    y0: float
    harmonics: list   # [(k, ax, bx, ay, by)], k > 0, по возрастанию


def real_coefficients(terms: list[Term]) -> RealSeries:
    c0 = sum((t.c for t in terms if t.k == 0), 0j)
    pos: dict[int, complex] = {}
    neg: dict[int, complex] = {}
    for t in terms:
        if t.k > 0:
            pos[t.k] = pos.get(t.k, 0j) + t.c
        elif t.k < 0:
            neg[-t.k] = neg.get(-t.k, 0j) + t.c
    harm = []
    for k in sorted(set(pos) | set(neg)):
        cp, cn = pos.get(k, 0j), neg.get(k, 0j)
        harm.append((k, cp.real + cn.real, cn.imag - cp.imag,
                     cp.imag + cn.imag, cp.real - cn.real))
    return RealSeries(c0.real, c0.imag, harm)


def eval_real(rs: RealSeries, t) -> np.ndarray:
    t = np.asarray(t, float)
    x = np.full(t.shape, rs.x0)
    y = np.full(t.shape, rs.y0)
    for k, ax, bx, ay, by in rs.harmonics:
        th = 2 * np.pi * k * t
        x = x + ax * np.cos(th) + bx * np.sin(th)
        y = y + ay * np.cos(th) + by * np.sin(th)
    return x + 1j * y


def real_series(terms: list[Term], digits: int = 4, max_terms: int | None = None,
                fmt: str = "text", wrap: int = 3) -> str:
    sel, rest = _select(terms, max_terms)
    rs = real_coefficients(sel)

    def arg(k):
        return "2*pi*t" if k == 1 else f"2*pi*{k}*t"

    def argt(k):
        return "2\\pi t" if k == 1 else f"2\\pi\\,{k}t"

    lines = []
    for name, (ci, si, c0) in (("x", (1, 2, rs.x0)), ("y", (3, 4, rs.y0))):
        if fmt == "text":
            parts = []
            for h in rs.harmonics:
                k = h[0]
                parts += [(h[ci], f"cos({arg(k)})"), (h[si], f"sin({arg(k)})")]
            lines.append(f"{name}(t) = " + _signed_sum(parts, digits, c0) + _tail_text(rest, "text"))
        else:
            parts = []
            for h in rs.harmonics:
                k = h[0]
                for coef, trig in ((h[ci], "\\cos"), (h[si], "\\sin")):
                    if _r(coef, digits) != 0:
                        parts.append(rf"{_num(coef, digits)}\,{trig}\!\left({argt(k)}\right)")
            if _r(c0, digits) != 0 or not parts:
                parts.insert(0, _num(c0, digits))
            lines.append(_latex_join(f"{name}(t)", parts, wrap).replace("+ -", "- ") + _tail_text(rest, "latex"))
    return ("\n" if fmt == "text" else " \\\\\n").join(lines)


FORMS = {"complex": complex_series, "polar": polar_series, "real": real_series}


def latex_document(body: str) -> str:
    """Минимальный компилируемый .tex-документ вокруг тела формулы."""
    return ("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[russian]{babel}\n\\begin{document}\n\\begin{align*}\n"
            f"{body}\n\\end{{align*}}\n\\end{{document}}\n")
