"""Таблица комплексных коэффициентов: включение, ручное редактирование, undo/redo."""
from __future__ import annotations
import cmath
import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox, QTableView, QVBoxLayout)

from ..analytics import parse_complex, table_order
from .harmonic_menu import fmt_complex, show_harmonic_menu
from .plots import LazyPanel

HEAD = ["Вкл", "k", "Re c", "Im c", "|c|", "arg c, рад", "правка"]
GRAYB, ORANGEB = QBrush(QColor(125, 130, 145)), QBrush(QColor(255, 190, 60))


class CoeffModel(QAbstractTableModel):
    error = Signal(str)

    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.sort_by = "k"
        self.order = table_order(editor.spectrum, self.sort_by)

    def set_sort(self, by: str | None = None):
        self.beginResetModel()
        self.sort_by = by or self.sort_by
        self.order = table_order(self.editor.spectrum, self.sort_by)
        self.endResetModel()

    def refresh(self):
        if len(self.order):
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.order) - 1, len(HEAD) - 1))

    def k_at(self, row: int) -> int:
        return int(self.editor.spectrum.freqs[self.order[row]])

    # ---- модель ----
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.order)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEAD)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return HEAD[section]
        return None

    def flags(self, index):
        f = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            f |= Qt.ItemIsUserCheckable
        elif index.column() in (2, 3, 4, 5):
            f |= Qt.ItemIsEditable
        return f

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        s = self.editor.spectrum
        i, col = self.order[index.row()], index.column()
        c = s.edited[i]
        if role == Qt.CheckStateRole and col == 0:
            return Qt.Checked if s.enabled[i] else Qt.Unchecked
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 1:
                return int(s.freqs[i])
            if col == 6:
                return "✎" if (c != s.original[i]) else ""
            val = {2: c.real, 3: c.imag, 4: abs(c), 5: cmath.phase(c)}.get(col)
            if val is not None:
                return f"{val:.12g}" if role == Qt.EditRole else f"{val:.8g}"
            return None
        if role == Qt.ForegroundRole:
            if not s.enabled[i]:
                return GRAYB
            if c != s.original[i]:
                return ORANGEB
        if role == Qt.TextAlignmentRole and col >= 1:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False
        k, col = self.k_at(index.row()), index.column()
        try:
            if role == Qt.CheckStateRole and col == 0:
                self.editor.set_enabled(k, value == Qt.Checked or value == 2)
                return True
            if role == Qt.EditRole and col in (2, 3, 4, 5):
                x = float(str(value).replace(",", ".").replace("−", "-"))
                c = self.editor.spectrum.coeff(k)
                if col == 2:
                    new = complex(x, c.imag)
                elif col == 3:
                    new = complex(c.real, x)
                elif col == 4:
                    if x < 0:
                        raise ValueError("Модуль должен быть >= 0")
                    new = cmath.rect(x, cmath.phase(c))
                else:
                    new = cmath.rect(abs(c), x)
                self.editor.set_coeff(k, new)
                return True
        except ValueError as e:
            self.error.emit(str(e))
        return False


class TablePanel(LazyPanel):
    delay_ms = 20

    def __init__(self, hub, parent=None):
        super().__init__(hub, parent)
        self.editor = hub.editor
        self.model = CoeffModel(self.editor)
        self.model.error.connect(lambda m: self.status.setText(m))
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setAlternatingRowColors(True)
        self.view.verticalHeader().setVisible(False)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.selectionModel().selectionChanged.connect(self._on_selection)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._on_context_menu)

        self.sort = QComboBox()
        self.sort.addItem("Сортировка: по k", "k")
        self.sort.addItem("Сортировка: по |c| (исходный)", "amp")
        self.sort.currentIndexChanged.connect(lambda _: self.model.set_sort(self.sort.currentData()))

        def btn(text, fn):
            b = QPushButton(text)
            b.clicked.connect(fn)
            return b

        self.b_undo = btn("↶ Отменить", lambda: self.editor.undo())
        self.b_redo = btn("↷ Повторить", lambda: self.editor.redo())
        self.sp_band = QSpinBox(minimum=0, maximum=4096, value=10)
        row1 = QHBoxLayout()
        for w in (self.sort, self.b_undo, self.b_redo,
                  btn("Включить все", lambda: self.editor.enable_all(True)),
                  btn("Выключить все", lambda: self.editor.enable_all(False)),
                  btn("Вкл. выбранные", lambda: self._toggle_selected(True)),
                  btn("Выкл. выбранные", lambda: self._toggle_selected(False)),
                  QLabel("Оставить |k| ≤"), self.sp_band,
                  btn("Применить", lambda: self.editor.enable_band(self.sp_band.value()))):
            row1.addWidget(w)
        row1.addStretch(1)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("c_k = a+bi  или  r@φ (рад)  или  r@φ° — для выбранной строки")
        self.entry.returnPressed.connect(self._apply_entry)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Значение:"))
        row2.addWidget(self.entry, 1)
        row2.addWidget(btn("Задать", self._apply_entry))
        row2.addWidget(btn("Сбросить выбранные", self._reset_selected))
        row2.addWidget(btn("Сбросить всё", lambda: self.editor.reset_all()))
        self.status = QLabel()
        self.status.setStyleSheet("color: #c9a227;")
        lay = QVBoxLayout(self)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.status)
        lay.addWidget(self.view, 1)

    def _selected_ks(self) -> list[int]:
        return [self.model.k_at(i.row()) for i in self.view.selectionModel().selectedRows()]

    def _on_selection(self, *_):
        ks = self._selected_ks()
        if ks:
            self.entry.setText(fmt_complex(self.editor.spectrum.coeff(ks[0])))
        self.status.setText("")

    def _on_context_menu(self, pos):
        idx = self.view.indexAt(pos)
        if not idx.isValid():
            return
        if idx.row() not in {i.row() for i in self.view.selectionModel().selectedRows()}:
            self.view.selectRow(idx.row())            # правый клик вне выделения — выбрать строку
        show_harmonic_menu(self, self.editor, self._selected_ks(),
                           self.view.viewport().mapToGlobal(pos))

    def _toggle_selected(self, flag: bool):
        ks = self._selected_ks()
        if ks:
            self.editor.set_enabled_many(ks, flag)

    def _reset_selected(self):
        for k in self._selected_ks():
            self.editor.reset_coeff(k)

    def _apply_entry(self):
        ks = self._selected_ks()
        if not ks:
            self.status.setText("Выберите строку таблицы")
            return
        try:
            self.editor.set_coeff(ks[0], parse_complex(self.entry.text()))
            self.status.setText("")
        except ValueError as e:
            self.status.setText(str(e))

    def update_view(self, replaced: bool):
        if replaced:
            self.model.set_sort()
        else:
            self.model.refresh()
        self.b_undo.setEnabled(self.editor.can_undo)
        self.b_redo.setEnabled(self.editor.can_redo)
