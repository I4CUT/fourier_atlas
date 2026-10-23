"""Импорт контуров из SVG (path, polygon, polyline, rect, circle, ellipse).

Поддержано: все команды path (M L H V C S Q T A Z, абсолютные и относительные),
кривые Безье и дуги разворачиваются в ломаные; transform (translate, scale, rotate,
matrix, skewX, skewY) с наследованием от групп. Не поддержано: <use>, CSS-стили
и проценты в размерах (такие элементы пропускаются). Ось Y в SVG направлена вниз —
помечается y_down=True и разворачивается при подготовке контура.
"""
from __future__ import annotations
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from .model import GeoLoadError, Region, clean_ring

MAX_BYTES = 200 * 1024 * 1024
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SKIP = {"defs", "clipPath", "mask", "symbol", "marker", "pattern", "style", "title", "metadata", "desc"}


class _Scan:
    def __init__(self, s: str):
        self.s, self.i = s, 0

    def _skip(self):
        while self.i < len(self.s) and self.s[self.i] in " ,\t\r\n":
            self.i += 1

    def more(self) -> bool:
        self._skip()
        return self.i < len(self.s)

    def cmd(self):
        self._skip()
        if self.i < len(self.s) and self.s[self.i].isalpha():
            c = self.s[self.i]
            self.i += 1
            return c
        return None

    def num(self) -> float:
        self._skip()
        m = _NUM.match(self.s, self.i)
        if not m:
            raise GeoLoadError(f"SVG path: ожидалось число в позиции {self.i}")
        self.i = m.end()
        return float(m.group())

    def flag(self) -> bool:
        self._skip()
        if self.i >= len(self.s) or self.s[self.i] not in "01":
            raise GeoLoadError("SVG path: некорректный флаг дуги")
        self.i += 1
        return self.s[self.i - 1] == "1"


def _arc(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2, per_quarter):
    if (x1, y1) == (x2, y2):
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [(x2, y2)]
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    co = math.sqrt(max(0.0, num / den)) if den else 0.0
    if fa == fs:
        co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    th1 = ang(1, 0, ux, uy)
    dth = ang(ux, uy, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dth > 0:
        dth -= 2 * math.pi
    elif fs and dth < 0:
        dth += 2 * math.pi
    n = max(2, math.ceil(per_quarter * abs(dth) / (math.pi / 2)))
    pts = []
    for j in range(1, n + 1):
        th = th1 + dth * j / n
        pts.append((cp * rx * math.cos(th) - sp * ry * math.sin(th) + cx,
                    sp * rx * math.cos(th) + cp * ry * math.sin(th) + cy))
    pts[-1] = (x2, y2)
    return pts


def parse_path(d: str, steps: int = 24) -> list:
    """Разбирает атрибут d в список подконтуров (каждый — список (x, y))."""
    sc = _Scan(d)
    subs, cur = [], None
    pos = start = (0.0, 0.0)
    cmd, prev_c, prev_q = None, None, None     # prev_c/prev_q — последняя контрольная точка C/Q

    def bez3(p0, p1, p2, p3):
        t = np.linspace(0, 1, steps + 1)[1:, None]
        a = np.array(p0); b = np.array(p1); c = np.array(p2); e = np.array(p3)
        pts = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t**2 * c + t**3 * e
        return [tuple(q) for q in pts]

    def bez2(p0, p1, p2):
        t = np.linspace(0, 1, steps + 1)[1:, None]
        a = np.array(p0); b = np.array(p1); c = np.array(p2)
        pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t**2 * c
        return [tuple(q) for q in pts]

    while sc.more():
        c = sc.cmd()
        if c is not None:
            cmd = c
        elif cmd is None or cmd in "Zz":
            raise GeoLoadError("SVG path: лишние данные после команды")
        rel = cmd.islower()
        u = cmd.upper()
        ox, oy = pos if rel else (0.0, 0.0)
        if u == "Z":
            if cur:
                subs.append(cur)
            cur, pos = None, start
            prev_c = prev_q = None
            continue
        if u == "M":
            if cur:
                subs.append(cur)
            pos = start = (sc.num() + ox, sc.num() + oy)
            cur = [pos]
            cmd = "l" if rel else "L"
            prev_c = prev_q = None
            continue
        if cur is None:
            cur = [pos]
        if u == "L":
            pos = (sc.num() + ox, sc.num() + oy); cur.append(pos); prev_c = prev_q = None
        elif u == "H":
            pos = (sc.num() + ox, pos[1]); cur.append(pos); prev_c = prev_q = None
        elif u == "V":
            pos = (pos[0], sc.num() + oy); cur.append(pos); prev_c = prev_q = None
        elif u == "C":
            p1 = (sc.num() + ox, sc.num() + oy); p2 = (sc.num() + ox, sc.num() + oy)
            p3 = (sc.num() + ox, sc.num() + oy)
            cur += bez3(pos, p1, p2, p3); pos, prev_c, prev_q = p3, p2, None
        elif u == "S":
            p1 = (2 * pos[0] - prev_c[0], 2 * pos[1] - prev_c[1]) if prev_c else pos
            p2 = (sc.num() + ox, sc.num() + oy); p3 = (sc.num() + ox, sc.num() + oy)
            cur += bez3(pos, p1, p2, p3); pos, prev_c, prev_q = p3, p2, None
        elif u == "Q":
            p1 = (sc.num() + ox, sc.num() + oy); p2 = (sc.num() + ox, sc.num() + oy)
            cur += bez2(pos, p1, p2); pos, prev_q, prev_c = p2, p1, None
        elif u == "T":
            p1 = (2 * pos[0] - prev_q[0], 2 * pos[1] - prev_q[1]) if prev_q else pos
            p2 = (sc.num() + ox, sc.num() + oy)
            cur += bez2(pos, p1, p2); pos, prev_q, prev_c = p2, p1, None
        elif u == "A":
            rx, ry, rot = sc.num(), sc.num(), sc.num()
            fa, fs = sc.flag(), sc.flag()
            p = (sc.num() + ox, sc.num() + oy)
            cur += _arc(pos[0], pos[1], rx, ry, rot, fa, fs, p[0], p[1], max(2, steps // 4))
            pos, prev_c, prev_q = p, None, None
        else:
            raise GeoLoadError(f"SVG path: неизвестная команда '{cmd}'")
    if cur:
        subs.append(cur)
    return subs


# ---------- transform ----------
def _mat(a, b, c, d, e, f):
    return np.array([[a, c, e], [b, d, f], [0, 0, 1.0]])


def parse_transform(s: str | None) -> np.ndarray:
    m = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in _NUM.findall(args)]
        try:
            if name == "translate":
                t = _mat(1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
            elif name == "scale":
                t = _mat(v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
            elif name == "rotate":
                a = math.radians(v[0]); ca, sa = math.cos(a), math.sin(a)
                t = _mat(ca, sa, -sa, ca, 0, 0)
                if len(v) >= 3:
                    t = _mat(1, 0, 0, 1, v[1], v[2]) @ t @ _mat(1, 0, 0, 1, -v[1], -v[2])
            elif name == "matrix":
                t = _mat(*v[:6])
            elif name == "skewX":
                t = _mat(1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
            elif name == "skewY":
                t = _mat(1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
            else:
                continue
        except (IndexError, TypeError):
            raise GeoLoadError(f"SVG: некорректный transform '{name}({args})'")
        m = m @ t
    return m


def _apply(m: np.ndarray, pts) -> np.ndarray:
    a = np.asarray(pts, float)
    return np.column_stack([m[0, 0] * a[:, 0] + m[0, 1] * a[:, 1] + m[0, 2],
                            m[1, 0] * a[:, 0] + m[1, 1] * a[:, 1] + m[1, 2]])


def _f(el, key, default=0.0):
    m = _NUM.match(el.get(key, "") or "")
    return float(m.group()) if m else default


def _shape_polys(el, tag, steps):
    if tag == "path":
        return parse_path(el.get("d", ""), steps)
    if tag in ("polygon", "polyline"):
        v = [float(x) for x in _NUM.findall(el.get("points", ""))]
        return [list(zip(v[0::2], v[1::2]))] if len(v) >= 6 else []
    if tag == "rect":
        x, y, w, h = _f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height")
        return [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]] if w > 0 and h > 0 else []
    if tag in ("circle", "ellipse"):
        cx, cy = _f(el, "cx"), _f(el, "cy")
        rx = _f(el, "r") if tag == "circle" else _f(el, "rx")
        ry = rx if tag == "circle" else _f(el, "ry")
        if rx <= 0 or ry <= 0:
            return []
        a = np.linspace(0, 2 * np.pi, 4 * steps, endpoint=False)
        return [list(zip(cx + rx * np.cos(a), cy + ry * np.sin(a)))]
    return []


def load_svg(path, steps: int = 24, group: str = "Импорт") -> list[Region]:
    p = Path(path)
    if p.stat().st_size > MAX_BYTES:
        raise GeoLoadError("SVG слишком большой")
    raw = p.read_bytes()
    if re.search(rb"<!ENTITY", raw, re.I):
        raise GeoLoadError("SVG с объявлениями ENTITY не поддерживается (безопасность)")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise GeoLoadError(f"Некорректный SVG/XML: {e}")

    parts: list = []

    def walk(el, parent_m):
        tag = el.tag.split("}")[-1]
        if tag in _SKIP:
            return
        m = parent_m @ parse_transform(el.get("transform"))
        for sub in _shape_polys(el, tag, steps):
            if len(sub) >= 3:
                parts.append(_apply(m, sub))
        for ch in el:
            walk(ch, m)

    walk(root, np.eye(3))
    parts = [r for r in (clean_ring(x) for x in parts) if r is not None]
    if not parts:
        raise GeoLoadError("В SVG не найдено подходящих фигур (path/polygon/rect/circle/…)")
    return [Region(p.stem, parts, kind="planar", y_down=True, group=group, source=str(p))]
