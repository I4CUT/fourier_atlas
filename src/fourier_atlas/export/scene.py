"""Сцена эпициклов как набор примитивов + два бэкенда: растр (Pillow) и SVG (текст).

Геометрия строится один раз (build_primitives) и рисуется любым бэкендом, поэтому
PNG/GIF/MP4 и SVG выглядят одинаково. Без Qt — безопасно работает в фоновом потоке.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..animation.viewport import Viewport


@dataclass
class Style:
    background: tuple = (18, 20, 28)
    target: tuple = (120, 130, 160, 90)
    circle: tuple = (110, 150, 230, 70)
    vector: tuple = (235, 235, 245, 210)
    trail: tuple = (255, 190, 60)
    show_circles: bool = True
    show_vectors: bool = True
    show_target: bool = True
    trail_segments: int = 8
    margin: float = 0.08
    target_width: float = 1.2
    circle_width: float = 1.0
    vector_width: float = 1.4
    trail_width: float = 2.2
    tip_radius: float = 4.0
    min_circle_px: float = 1.0


@dataclass
class Prim:
    kind: str                 # 'polyline' | 'circle' | 'dot'
    color: tuple              # (r, g, b, a)
    width: float = 1.0
    pts: np.ndarray | None = None     # polyline: (M, 2) пиксели
    center: tuple = (0.0, 0.0)        # circle/dot
    radius: float = 0.0


def fit_viewport(width: int, height: int, target: np.ndarray, path: np.ndarray, margin: float) -> Viewport:
    v = Viewport(width, height)
    v.fit(np.concatenate([np.asarray(target, complex), np.asarray(path, complex)]), margin=margin)
    return v


def build_primitives(frame, viewport: Viewport, target, style: Style, highlight: dict | None = None) -> list[Prim]:
    prims: list[Prim] = []
    w, h = viewport.width, viewport.height
    if style.show_target and target is not None:
        t = np.append(target, target[0])
        sx, sy = viewport.to_screen(t)
        prims.append(Prim("polyline", style.target, style.target_width, np.column_stack([sx, sy])))
    jx, jy = viewport.to_screen(frame.joints)
    if style.show_circles:
        r = np.asarray(frame.radii, float) * viewport.scale
        for i, rp in enumerate(r):
            if rp < style.min_circle_px:
                continue
            if jx[i] + rp < 0 or jx[i] - rp > w or jy[i] + rp < 0 or jy[i] - rp > h:
                continue                                    # целиком за кадром
            prims.append(Prim("circle", style.circle, style.circle_width,
                              center=(float(jx[i]), float(jy[i])), radius=float(rp)))
    if style.show_vectors:
        prims.append(Prim("polyline", style.vector, style.vector_width, np.column_stack([jx, jy])))
    for idx, strength in (highlight or {}).items():          # вспышка вектора/окружности при ноте мелодии
        if not (0 <= idx < len(frame.radii)) or strength < 0.03:
            continue
        rp = float(frame.radii[idx]) * viewport.scale
        prims.append(Prim("polyline", (255, 255, 255, int(110 + 145 * strength)), 1.4 + 2.8 * strength,
                          np.column_stack([jx[idx:idx + 2], jy[idx:idx + 2]])))
        if rp >= style.min_circle_px:
            prims.append(Prim("circle", (*style.trail[:3], int(255 * strength)), 1.0 + 2.0 * strength,
                              center=(float(jx[idx]), float(jy[idx])), radius=rp))
    tx, ty = viewport.to_screen(frame.trail)
    n = style.trail_segments
    edges = np.linspace(0, len(tx) - 1, n + 1).astype(int)
    for i in range(n):
        a, b = edges[i], edges[i + 1] + 1
        if b - a < 2:
            continue
        alpha = int(40 + 215 * (i + 1) / n)
        prims.append(Prim("polyline", (*style.trail[:3], alpha), style.trail_width,
                          np.column_stack([tx[a:b], ty[a:b]])))
    prims.append(Prim("dot", (*style.trail[:3], 255), 0.0, center=(float(jx[-1]), float(jy[-1])),
                      radius=style.tip_radius))
    return prims


# ---------------- растр ----------------
def render_raster(prims: list[Prim], size: tuple, background, supersample: int = 1):
    from PIL import Image, ImageDraw
    w, h = size
    s = max(1, int(supersample))
    img = Image.new("RGB", (w * s, h * s), tuple(background))
    d = ImageDraw.Draw(img, "RGBA")
    for p in prims:
        col = tuple(int(c) for c in p.color)
        lw = max(1, round(p.width * s))
        if p.kind == "polyline":
            pts = [(float(x) * s, float(y) * s) for x, y in p.pts]
            if len(pts) >= 2:
                d.line(pts, fill=col, width=lw, joint="curve")
        else:
            cx, cy, r = p.center[0] * s, p.center[1] * s, p.radius * s
            box = [cx - r, cy - r, cx + r, cy + r]
            if p.kind == "circle":
                d.ellipse(box, outline=col, width=lw)
            else:
                d.ellipse(box, fill=col)
    if s > 1:
        img = img.resize((w, h), Image.LANCZOS)
    return img


# ---------------- SVG ----------------
def _rgb(c):
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def render_svg(prims: list[Prim], size: tuple, background) -> str:
    w, h = size
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           "<title>Fourier Atlas</title>",
           f'<rect width="{w}" height="{h}" fill="{_rgb(background)}"/>']
    for p in prims:
        op = f'{p.color[3] / 255:.3f}' if len(p.color) > 3 else "1"
        if p.kind == "polyline":
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in p.pts)
            out.append(f'<polyline fill="none" stroke="{_rgb(p.color)}" stroke-opacity="{op}" '
                       f'stroke-width="{p.width:g}" stroke-linejoin="round" stroke-linecap="round" '
                       f'points="{pts}"/>')
        elif p.kind == "circle":
            out.append(f'<circle cx="{p.center[0]:.2f}" cy="{p.center[1]:.2f}" r="{p.radius:.2f}" fill="none" '
                       f'stroke="{_rgb(p.color)}" stroke-opacity="{op}" stroke-width="{p.width:g}"/>')
        else:
            out.append(f'<circle cx="{p.center[0]:.2f}" cy="{p.center[1]:.2f}" r="{p.radius:g}" '
                       f'fill="{_rgb(p.color)}"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
