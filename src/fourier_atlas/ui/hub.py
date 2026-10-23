"""Qt-обёртка над SpectrumEditor: превращает события редактора в сигналы."""
from __future__ import annotations
from PySide6.QtCore import QObject, Signal

from ..analytics import SpectrumEditor


class SpectrumHub(QObject):
    changed = Signal()      # правка коэффициента, включение гармоник, число эпициклов, undo/redo
    replaced = Signal()     # загружен другой контур/спектр

    def __init__(self, editor: SpectrumEditor, parent=None):
        super().__init__(parent)
        self.editor = editor
        editor.subscribe(self._on_event)

    def _on_event(self, kind: str) -> None:
        (self.replaced if kind == "replaced" else self.changed).emit()
