"""Математическое ядро. Зависит только от NumPy (никакого Qt)."""
from .geometry import (close_contour, to_complex, from_complex,
                       resample_arclength, normalize_contour, contour_length)
from .fourier import Spectrum, compute_spectrum, reconstruct, reconstruct_at
from .epicycles import EpicycleChain, build_chain
from .formula import (Term, terms_from_chain, terms_value, complex_series, polar_series,
                      real_series, real_coefficients, eval_real, latex_document)
from .metrics import (rmse, max_error, hausdorff, ReconstructionError,
                      evaluate, convergence_curve, harmonics_for_rmse,
                      convergence_by_amplitude, epicycles_for_rmse)

__all__ = [n for n in dir() if not n.startswith("_")]
