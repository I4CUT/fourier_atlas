"""Панель «Ряд Фурье»: формула в трёх формах, копирование текста/LaTeX, сохранение."""
from __future__ import annotations
import logging
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from ..animation import AnimationController
from ..core.formula import FORMS, latex_document, terms_from_chain

log = logging.getLogger(__name__)
MODES = [("Комплексная  z(t)", "complex"), ("Полярная  r·e^{iφ}", "polar"),
         ("Вещественная  x(t), y(t)", "real")]


class FormulaPanel(QWidget):
    def __init__(self, controller: AnimationController, parent=None):
        super().__init__(parent)
        self.ctrl = controller
        self.mode = QComboBox()
        for title, key in MODES:
            self.mode.addItem(title, key)
        self.terms = QSpinBox(minimum=1, maximum=2048, value=8)
        self.digits = QSpinBox(minimum=1, maximum=12, value=4)
        self.view = QPlainTextEdit(readOnly=True)
        self.view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.view.setMinimumHeight(110)
        b_text = QPushButton("Копировать текст"); b_tex = QPushButton("Копировать LaTeX")
        b_save = QPushButton("Сохранить…")
        b_text.clicked.connect(lambda: self._copy("text"))
        b_tex.clicked.connect(lambda: self._copy("latex"))
        b_save.clicked.connect(self._save)

        top = QHBoxLayout()
        for w in (self.mode, QLabel("Членов:"), self.terms, QLabel("Знаков:"), self.digits,
                  b_text, b_tex, b_save):
            top.addWidget(w)
        top.addStretch(1)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.view)
        self.mode.currentIndexChanged.connect(self.refresh)
        self.terms.valueChanged.connect(self.refresh)
        self.digits.valueChanged.connect(self.refresh)
        self.refresh()

    def _build(self, fmt: str) -> str:
        terms = terms_from_chain(self.ctrl.chain)
        fn = FORMS[self.mode.currentData()]
        return fn(terms, digits=self.digits.value(), max_terms=self.terms.value(), fmt=fmt)

    def refresh(self, *_):
        """Вызывать после любых изменений спектра/цепочки — формула берётся из текущей цепочки."""
        n = len(self.ctrl.chain.freqs)
        self.view.setPlainText(self._build("text") + f"\n\n# всего членов в цепочке: {n}, "
                               "t ∈ [0,1); i — мнимая единица; строку можно выполнить в Python")

    def _copy(self, fmt: str):
        QApplication.clipboard().setText(self._build(fmt))

    def _save(self):
        path, flt = QFileDialog.getSaveFileName(self, "Сохранить формулу", "fourier_series",
                                                "Текст (*.txt);;LaTeX (*.tex)")
        if not path:
            return
        tex = path.lower().endswith(".tex") or "tex" in flt.lower()
        try:
            data = latex_document(self._build("latex")) if tex else self._build("text") + "\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            log.info("Формула сохранена: %s", path)
        except OSError as e:
            log.exception("Не удалось сохранить формулу")
            QMessageBox.critical(self, "Ошибка сохранения", str(e))
