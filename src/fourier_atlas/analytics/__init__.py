"""Аналитика и редактирование спектра. Не зависит от Qt."""
from .editor import SpectrumEditor
from .tables import parse_complex, table_order, spectrum_plot_data, SpectrumPlotData
from .analysis import compare_reconstruction, relative_error_percent, epicycles_for_relative_error, \
    contour_diagonal

__all__ = [n for n in dir() if not n.startswith("_")]
