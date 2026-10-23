"""Контекстное меню гармоники (правая кнопка): изменить, включить, отключить, сбросить."""
from __future__ import annotations
from PySide6.QtWidgets import QInputDialog, QLineEdit, QMenu, QMessageBox

from ..analytics import parse_complex


def fmt_complex(c: complex) -> str:
    return f"{c.real:.8g}{c.imag:+.8g}i"


def ask_coeff(parent, k: int, current: complex, original: complex):
    """Диалог ввода c_k. Возвращает complex или None (отмена). При ошибке ввода спрашивает снова."""
    text = fmt_complex(current)
    while True:
        text, ok = QInputDialog.getText(
            parent, f"Изменить гармонику k = {k}",
            f"c_k  (исходное значение: {fmt_complex(original)})\n"
            "Форматы: a+bi,  r@φ (рад),  r@φ° (градусы)",
            QLineEdit.Normal, text)
        if not ok:
            return None
        try:
            return parse_complex(text)
        except ValueError as e:
            QMessageBox.warning(parent, "Некорректное значение", str(e))


def show_harmonic_menu(parent, editor, ks, global_pos) -> None:
    ks = [int(k) for k in ks]
    spec = editor.spectrum
    try:
        idx = [spec.index_of(k) for k in ks]
    except KeyError:
        return
    if not idx:
        return
    enabled = [bool(spec.enabled[i]) for i in idx]
    edited = [bool(spec.edited[i] != spec.original[i]) for i in idx]
    single = len(ks) == 1

    menu = QMenu(parent)
    title = menu.addAction(f"Гармоника k = {ks[0]}" if single else f"Выбрано гармоник: {len(ks)}")
    title.setEnabled(False)
    menu.addSeparator()
    a_edit = menu.addAction("Изменить значение…")
    a_edit.setEnabled(single)
    a_on = menu.addAction("Включить")
    a_on.setEnabled(not all(enabled))
    a_off = menu.addAction("Отключить")
    a_off.setEnabled(any(enabled))
    menu.addSeparator()
    a_reset = menu.addAction("Сбросить значение")
    a_reset.setEnabled(any(edited))

    chosen = menu.exec(global_pos)
    if chosen is None:
        return
    if chosen is a_edit:
        value = ask_coeff(parent, ks[0], spec.coeff(ks[0]), complex(spec.original[idx[0]]))
        if value is not None:
            editor.set_coeff(ks[0], value)
    elif chosen is a_on:
        editor.set_enabled_many(ks, True)
    elif chosen is a_off:
        editor.set_enabled_many(ks, False)
    elif chosen is a_reset:
        for k in ks:
            editor.reset_coeff(k)
