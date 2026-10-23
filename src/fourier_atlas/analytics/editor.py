"""Редактор спектра: правки коэффициентов, включение/отключение гармоник, undo/redo.

Единая точка изменения спектра: после каждой операции цепочка эпициклов и кэш
траектории пересчитываются ОДИН раз (controller.refresh()), затем слушатели
(спектр, таблица, формула, сходимость…) получают событие — так всё остаётся
синхронизированным. Исходный спектр (Spectrum.original) никогда не меняется.
"""
from __future__ import annotations
import numpy as np


class SpectrumEditor:
    def __init__(self, controller, contour: np.ndarray, history_limit: int = 100):
        self.ctrl = controller
        self.z = np.asarray(contour, complex)
        self.history_limit = history_limit
        self._undo: list = []
        self._redo: list = []
        self._listeners: list = []

    # ---------- события ----------
    def subscribe(self, fn) -> None:
        """fn(kind): kind = 'changed' (правка/состояние) или 'replaced' (новый контур)."""
        self._listeners.append(fn)

    def _emit(self, kind: str) -> None:
        for fn in list(self._listeners):
            fn(kind)

    @property
    def spectrum(self):
        return self.ctrl.spectrum

    # ---------- история ----------
    def _push(self) -> None:
        self._undo.append(self.spectrum.snapshot())
        del self._undo[:-self.history_limit]
        self._redo.clear()

    def _commit(self) -> None:
        self.ctrl.refresh()
        self._emit("changed")

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.spectrum.snapshot())
        self.spectrum.restore(self._undo.pop())
        self._commit()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.spectrum.snapshot())
        self.spectrum.restore(self._redo.pop())
        self._commit()
        return True

    # ---------- правки ----------
    def _index(self, k: int) -> int:
        try:
            return self.spectrum.index_of(k)
        except KeyError:
            raise ValueError(f"Нет гармоники k={k}")

    def set_coeff(self, k: int, value) -> bool:
        """Задать комплексный коэффициент c_k. False, если значение не изменилось."""
        value = complex(value)
        if not (np.isfinite(value.real) and np.isfinite(value.imag)):
            raise ValueError("Коэффициент должен быть конечным числом")
        i = self._index(k)
        if self.spectrum.edited[i] == value:
            return False
        self._push()
        self.spectrum.edited[i] = value
        self._commit()
        return True

    def reset_coeff(self, k: int) -> bool:
        i = self._index(k)
        if self.spectrum.edited[i] == self.spectrum.original[i]:
            return False
        self._push()
        self.spectrum.reset_coeff(k)
        self._commit()
        return True

    def reset_all(self) -> bool:
        """Вернуть исходные коэффициенты и включить все гармоники."""
        s = self.spectrum
        if not s.edited_mask().any() and s.enabled.all():
            return False
        self._push()
        s.reset_all()
        self._commit()
        return True

    def set_enabled(self, k: int, flag: bool) -> bool:
        return self.set_enabled_many([k], flag)

    def set_enabled_many(self, ks, flag: bool) -> bool:
        idx = [self._index(k) for k in ks]
        s = self.spectrum
        if all(s.enabled[i] == flag for i in idx):
            return False
        self._push()
        for i in idx:
            s.enabled[i] = bool(flag)
        self._commit()
        return True

    def enable_all(self, flag: bool = True) -> bool:
        return self.set_enabled_many(self.spectrum.freqs.tolist(), flag)

    def enable_band(self, max_k: int) -> bool:
        """Оставить включёнными только гармоники |k| <= max_k (k=0 включён всегда)."""
        s = self.spectrum
        target = np.abs(s.freqs) <= int(max_k)
        if np.array_equal(s.enabled, target):
            return False
        self._push()
        s.enabled = target
        self._commit()
        return True

    # ---------- не входит в историю ----------
    def set_max_epicycles(self, n: int | None) -> None:
        self.ctrl.max_epicycles = n          # внутри — refresh()
        self._emit("changed")

    def replace(self, contour: np.ndarray, spectrum) -> None:
        """Новый контур/спектр (смена страны): история сбрасывается."""
        self.z = np.asarray(contour, complex)
        self.ctrl.set_spectrum(spectrum)
        self._undo.clear()
        self._redo.clear()
        self._emit("replaced")
