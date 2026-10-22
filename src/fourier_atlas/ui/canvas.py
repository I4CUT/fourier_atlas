"""Виджет отрисовки эпициклов (QPainter). Вся математика — в animation/core."""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF, QBrush
from PySide6.QtWidgets import QWidget

from ..animation import AnimationController, Viewport

BG = QColor(18, 20, 28)
TARGET = QColor(120, 130, 160, 90)
CIRCLE = QColor(110, 150, 230, 70)
VECTOR = QColor(235, 235, 245, 210)
TRAIL = (255, 190, 60)


def _poly(sx, sy) -> QPolygonF:
    return QPolygonF([QPointF(float(x), float(y)) for x, y in zip(sx, sy)])


class EpicycleCanvas(QWidget):
    def __init__(self, controller: AnimationController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.view = Viewport(self.width(), self.height())
        self.show_circles = True
        self.show_vectors = True
        self.show_target = True
        self.target: np.ndarray | None = None
        self._drag = None
        self.setMinimumSize(420, 420)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    # ---------- API ----------
    def set_target(self, z: np.ndarray | None) -> None:
        self.target = None if z is None else np.asarray(z, complex)
        self.fit_view()

    def fit_view(self) -> None:
        pts = self.target if self.target is not None else self.controller.path
        self.view.fit(pts)
        self.update()

    # ---------- события ----------
    def resizeEvent(self, e):
        self.view.resize(self.width(), self.height())
        self.fit_view()

    def wheelEvent(self, e):
        p = e.position()
        self.view.zoom_at(p.x(), p.y(), 1.15 if e.angleDelta().y() > 0 else 1 / 1.15)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            p = e.position()
            self.view.pan(p.x() - self._drag.x(), p.y() - self._drag.y())
            self._drag = p
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.fit_view()

    # ---------- рисование ----------
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        f = self.controller.frame()
        v = self.view

        if self.show_target and self.target is not None:
            sx, sy = v.to_screen(np.append(self.target, self.target[0]))
            p.setPen(QPen(TARGET, 1.2))
            p.drawPolyline(_poly(sx, sy))

        jx, jy = v.to_screen(f.joints)
        if self.show_circles:
            p.setPen(QPen(CIRCLE, 1))
            p.setBrush(Qt.NoBrush)
            for i, r in enumerate(f.radii):
                rp = float(r) * v.scale
                if rp >= 1.0:                       # отсечение невидимых окружностей
                    p.drawEllipse(QPointF(jx[i], jy[i]), rp, rp)
        if self.show_vectors:
            p.setPen(QPen(VECTOR, 1.4))
            p.drawPolyline(_poly(jx, jy))

        # След: сегменты с нарастающей непрозрачностью (хвост бледнеет)
        tx, ty = v.to_screen(f.trail)
        parts = 8
        edges = np.linspace(0, len(tx) - 1, parts + 1).astype(int)
        for i in range(parts):
            a, b = edges[i], edges[i + 1] + 1
            if b - a < 2:
                continue
            alpha = int(40 + 215 * (i + 1) / parts)
            p.setPen(QPen(QColor(*TRAIL, alpha), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline(_poly(tx[a:b], ty[a:b]))

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(*TRAIL)))
        p.drawEllipse(QPointF(jx[-1], jy[-1]), 4.0, 4.0)
        p.end()
