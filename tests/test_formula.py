"""Тесты формул ряда Фурье: текст выполняется и совпадает с реконструкцией."""
import re
import numpy as np
import pytest

from fourier_atlas.core import (build_chain, compute_spectrum, reconstruct_at, Term,
                                terms_from_chain, terms_value, complex_series, polar_series,
                                real_series, real_coefficients, eval_real, latex_document)
from fourier_atlas.data.samples import make_sample

T = np.linspace(0, 1, 37, endpoint=False)
NS = {"i": 1j, "pi": np.pi, "exp": np.exp, "cos": np.cos, "sin": np.sin}


def run_text(expr: str, t):
    """Выполняет правую часть 'z(t) = ...' как Python-выражение."""
    rhs = expr.split("=", 1)[1].strip()
    return eval(rhs, {"__builtins__": {}}, {**NS, "t": np.asarray(t)})


def chain_terms(name="Сердце", m=10):
    _, spec = make_sample(name, 512)
    chain = build_chain(spec, m)
    return chain, terms_from_chain(chain)


def test_terms_match_chain_and_reconstruction():
    chain, terms = chain_terms()
    assert len(terms) == 10 and terms[0].k == 0
    assert np.allclose(terms_value(terms, T), reconstruct_at(chain.freqs, chain.coeffs, T))


@pytest.mark.parametrize("form", [complex_series, polar_series])
def test_text_formula_is_executable_and_correct(form):
    _, terms = chain_terms()
    expr = form(terms, digits=12)
    assert expr.startswith("z(t) = ")
    assert np.allclose(run_text(expr, T), terms_value(terms, T), atol=1e-9)


def test_real_text_formula_correct_xy():
    _, terms = chain_terms("Звезда", 14)
    txt = real_series(terms, digits=12).split("\n")
    assert txt[0].startswith("x(t) = ") and txt[1].startswith("y(t) = ")
    x = run_text(txt[0], T)
    y = run_text(txt[1], T)
    ref = terms_value(terms, T)
    assert np.allclose(x, ref.real, atol=1e-9) and np.allclose(y, ref.imag, atol=1e-9)


def test_real_coefficients_formulas_pair_and_single():
    # эллипс: c1=2, c-1=1  ->  x = 3cos, y = 1 sin
    rs = real_coefficients([Term(0, 0j), Term(1, 2 + 0j), Term(-1, 1 + 0j)])
    (k, ax, bx, ay, by), = rs.harmonics
    assert (k, round(ax, 12), round(bx, 12), round(ay, 12), round(by, 12)) == (1, 3, 0, 0, 1)
    # одиночные члены без пары и со сдвигом фазы
    terms = [Term(0, 1 + 2j), Term(3, 0.5 - 0.25j), Term(-5, -1 + 0.7j)]
    rs = real_coefficients(terms)
    assert np.allclose(eval_real(rs, T), terms_value(terms, T))


def test_ellipse_text_is_human_readable():
    terms = [Term(0, 0j), Term(1, 2 + 0j), Term(-1, 1 + 0j)]
    assert real_series(terms, digits=2).split("\n") == [
        "x(t) = 0.00 + 3.00*cos(2*pi*t)", "y(t) = 0.00 + 1.00*sin(2*pi*t)"]
    s = complex_series(terms, digits=1)
    assert s == "z(t) = (0.0+0.0*i) + (2.0+0.0*i)*exp(2*pi*i*t) + (1.0+0.0*i)*exp(-2*pi*i*t)"


def test_negative_signs_and_no_negative_zero():
    terms = [Term(2, -0.12344 - 0.00001j), Term(-3, 1e-9 + 0j)]
    s = complex_series(terms, digits=4)
    assert "-0.0000" not in s and "(-0.1234-0.0000*i)" not in s
    assert "(-0.1234+0.0000*i)*exp(2*pi*i*2*t)" in s
    assert "exp(-2*pi*i*3*t)" in s


def test_truncation_notice_and_limit():
    _, terms = chain_terms(m=10)
    s = complex_series(terms, max_terms=4)
    assert s.count("exp(") == 3 and s.endswith("... (6 more)")   # k=0 без exp
    full = complex_series(terms, max_terms=None)
    assert "more" not in full
    with pytest.raises(ValueError):
        complex_series(terms, max_terms=0)


def test_latex_structure():
    _, terms = chain_terms(m=9)
    for form in (complex_series, polar_series, real_series):
        tex = form(terms, fmt="latex", max_terms=7, wrap=3)
        assert tex.count("{") == tex.count("}")
        assert tex.count("\\left(") == tex.count("\\right)") or form is real_series
        assert "\\pi" in tex and "&" in tex
        assert "*" not in tex.replace("\\*", "")           # никакого текстового '*'
    doc = latex_document(complex_series(terms, fmt="latex", max_terms=5))
    assert doc.count("\\begin{align*}") == 1 and doc.rstrip().endswith("\\end{document}")


def test_formula_follows_edits_and_mask():
    _, spec = make_sample("Эллипс", 256)
    spec.set_coeff(1, 5 + 0j)
    spec.set_enabled(-1, False)
    terms = terms_from_chain(build_chain(spec, 8))
    assert Term(1, 5 + 0j) in terms and all(t.k != -1 for t in terms)
    assert "(5.0000+0.0000*i)*exp(2*pi*i*t)" in complex_series(terms)


def test_formula_for_empty_chain_and_center_only():
    _, spec = make_sample("Эллипс", 256)
    spec.set_enabled(0, False)
    terms = terms_from_chain(build_chain(spec, max_rotating=0))
    assert terms == []
    assert complex_series(terms) == "z(t) = 0" and polar_series(terms) == "z(t) = 0"
    assert real_series(terms).split("\n") == ["x(t) = 0.0000", "y(t) = 0.0000"]
    assert complex_series(terms, fmt="latex") == "z(t) &= 0"
