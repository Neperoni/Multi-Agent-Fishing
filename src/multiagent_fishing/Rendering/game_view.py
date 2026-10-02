"""
GameView: the observer. Draws ONE game's GameSnapshot into its own fixed-size tile surface.

  * Pull-based: the game never knows views exist. The viewer calls view.render(snapshot)
    when it wants a picture, then blits view.surface wherever it likes.
  * Tile size is a constructor argument (and view.resize() later). Fonts, boat size and HUD
    scale with the tile unless you pass ui_scale yourself.
  * Each view has its own camera. All views share one MapAssets (the pre-rendered base map),
    so 20 views cost one map image, not 20.
  * The per-game fish overlay is rebuilt only when (game_id, fish_version) changes.

Needs a pygame display to exist (surface.convert()). For headless use set
SDL_VIDEODRIVER=dummy and call pygame.display.set_mode((1, 1)) first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import pygame
import pygame.gfxdraw

from .snapshot import GameSnapshot, Role

KM_PER_DEGREE = 111.32
V_MIN, V_MAX = -5000.0, 2000.0
PORT_COLOR = (255, 0, 127)


# =========================================================================
# COLORING
# =========================================================================
def _hex(c):
    return [int(c[i:i + 2], 16) for i in (1, 3, 5)]


_OCEAN_X = np.array([-5000, -3500, -2000, -500, 0], dtype=float)
_OCEAN_C = np.array([_hex(c) for c in ("#000814", "#001d3d", "#003566", "#1e6091", "#4cc9f0")], dtype=float)
_LAND_X = np.array([0, 500, 2000], dtype=float)
_LAND_C = np.array([_hex(c) for c in ("#a3b18a", "#588157", "#344e41")], dtype=float)


def colorize_height(height: np.ndarray) -> np.ndarray:
    """Float RGB (rows, cols, 3). Hard edge at sea level."""
    h = np.clip(np.nan_to_num(height, nan=V_MIN), V_MIN, V_MAX)
    out = np.empty(h.shape + (3,), dtype=float)
    for c in range(3):
        ocean = np.interp(h, _OCEAN_X, _OCEAN_C[:, c])
        land = np.interp(h, _LAND_X, _LAND_C[:, c])
        out[..., c] = np.where(h < 0, ocean, land)
    return out


def _rgb_to_surface(rgb: np.ndarray) -> pygame.Surface:
    """(rows, cols, 3), row 0 = south  ->  surface with north at the top."""
    arr = np.clip(rgb[::-1], 0, 255).transpose(1, 0, 2)
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr).astype(np.uint8)).convert()


def _fish_to_overlay(fish: np.ndarray) -> pygame.Surface:
    """Red translucent overlay, alpha ramping 0 -> 0.85 with density (matches the matplotlib look)."""
    v = np.clip(np.nan_to_num(fish), 0.0, 1.0)[::-1]  # north up
    h, w = v.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255 * (1 - 0.1 * v)
    rgba[..., 1] = 51 * (1 - v)
    rgba[..., 2] = 0
    rgba[..., 3] = 217 * v
    return pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA").convert_alpha()


# =========================================================================
# SHARED, STATIC MAP DATA
# =========================================================================
@dataclass
class MapAssets:
    """Built once, shared by every view. Nothing in here changes between games."""
    base: pygame.Surface          # pre-rendered terrain, north at the top
    height: np.ndarray            # (rows, cols) elevation, row 0 = south (for hover readout)
    res_km: float
    ports: Sequence[dict]         # each: name, grid_x, grid_y
    bbox: Optional[dict] = None   # lon/lat bounds, only used for the lon/lat readout

    @classmethod
    def build(cls, height: np.ndarray, res_km: float, ports: Sequence[dict] = (), bbox: Optional[dict] = None):
        return cls(_rgb_to_surface(colorize_height(height)), height, res_km, list(ports), bbox)

    @property
    def shape(self):
        return self.height.shape  # (rows, cols)

    @property
    def world_w(self):
        return self.shape[1] * self.res_km

    @property
    def world_h(self):
        return self.shape[0] * self.res_km

    def km_to_lonlat(self, x, y):
        if self.bbox is None:
            return None
        lat0 = (self.bbox["lat_min"] + self.bbox["lat_max"]) / 2.0
        lon_scale = math.cos(math.radians(lat0))
        return (self.bbox["lon_min"] + x / (KM_PER_DEGREE * lon_scale),
                self.bbox["lat_min"] + y / KM_PER_DEGREE)


# =========================================================================
# LOOK-AND-FEEL PER ROLE
# =========================================================================
@dataclass(frozen=True)
class RoleStyle:
    name: str
    color: tuple
    size_mult: float = 1.0
    label_prefix: str = "?"


# Dict order = draw order (later entries drawn on top).
DEFAULT_STYLES = {
    Role.FISHING: RoleStyle("Fishing Boat", (255, 204, 0), 1.0, "F"),
    Role.PATROL: RoleStyle("Patrol Boat", (0, 255, 102), 1.15, "P"),
}

# Boat = square hull + triangle bow, pointing up (+y) at heading 0
_HULL_SQUARE = [(-0.3, 0.2), (0.3, 0.2), (0.3, -0.5), (-0.3, -0.5)]
_BOW_TRIANGLE = [(-0.3, 0.2), (0.3, 0.2), (0.0, 0.6)]


def draw_boat(surf, sx, sy, heading, size, fill, outline=(0, 0, 0)):
    """heading in radians; 0 = north/up, increasing clockwise."""
    c, s = math.cos(heading), math.sin(heading)

    def xf(pts):
        return [(round(sx + (vx * c + vy * s) * size), round(sy + (vx * s - vy * c) * size)) for vx, vy in pts]

    for poly in (xf(_HULL_SQUARE), xf(_BOW_TRIANGLE)):
        pygame.gfxdraw.filled_polygon(surf, poly, fill)
        pygame.gfxdraw.aapolygon(surf, poly, outline)


# =========================================================================
# CAMERA (per view)
# =========================================================================
class Camera:
    """World units = km, +y north. Screen coordinates are tile-local pixels, +y down."""

    def __init__(self, world_w, world_h, size, max_scale):
        self.world_w, self.world_h = world_w, world_h
        self.max_scale = max_scale
        self.W, self.H = size
        self.fit()

    def _fit_scale(self):
        return 0.95 * min(self.W / self.world_w, self.H / self.world_h)

    def fit(self):
        self.scale = self._fit_scale()
        self.min_scale = self.scale * 0.5
        self.cx, self.cy = self.world_w / 2, self.world_h / 2

    def set_size(self, size):
        self.W, self.H = size
        self.min_scale = self._fit_scale() * 0.5
        self.scale = min(max(self.scale, self.min_scale), self.max_scale)

    def to_screen(self, x, y):
        return (x - self.cx) * self.scale + self.W / 2, self.H / 2 - (y - self.cy) * self.scale

    def to_world(self, sx, sy):
        return (sx - self.W / 2) / self.scale + self.cx, self.cy - (sy - self.H / 2) / self.scale

    def _clamp(self):
        self.cx = min(max(self.cx, 0.0), self.world_w)
        self.cy = min(max(self.cy, 0.0), self.world_h)

    def pan_pixels(self, dx, dy):
        self.cx -= dx / self.scale
        self.cy += dy / self.scale
        self._clamp()

    def zoom_at(self, sx, sy, factor):
        wx, wy = self.to_world(sx, sy)
        self.scale = min(max(self.scale * factor, self.min_scale), self.max_scale)
        self.cx = wx - (sx - self.W / 2) / self.scale
        self.cy = wy + (sy - self.H / 2) / self.scale
        self._clamp()


def _blit_map_slice(target, cam: Camera, src: pygame.Surface, res):
    """Blit only the visible part of a map-sized surface (north-up), scaled to the camera."""
    Wc, Hc = src.get_size()
    left, top = cam.to_world(0, 0)
    right, bottom = cam.to_world(cam.W, cam.H)

    c0, c1 = max(0, math.floor(left / res)), min(Wc, math.ceil(right / res))
    r0, r1 = max(0, math.floor(Hc - top / res)), min(Hc, math.ceil(Hc - bottom / res))
    if c1 <= c0 or r1 <= r0:
        return

    x0, y0 = cam.to_screen(c0 * res, (Hc - r0) * res)
    x1, y1 = cam.to_screen(c1 * res, (Hc - r1) * res)
    dx, dy = round(x0), round(y0)
    dw, dh = max(1, round(x1) - dx), max(1, round(y1) - dy)

    piece = src.subsurface((c0, r0, c1 - c0, r1 - r0))
    if res * cam.scale < 1.0:  # zoomed out: smooth; zoomed in: crisp cells
        piece = pygame.transform.smoothscale(piece, (dw, dh))
    else:
        piece = pygame.transform.scale(piece, (dw, dh))
    target.blit(piece, (dx, dy))


# =========================================================================
# THE VIEW
# =========================================================================
@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", size, bold=bold)


class GameView:
    def __init__(
        self,
        assets: MapAssets,
        size=(640, 480),
        *,
        title: Optional[str] = None,       # default: "Game {game_id}"
        styles: Optional[dict] = None,     # Role -> RoleStyle
        ui_scale: Optional[float] = None,  # None = auto from tile size
        show_overlay: bool = True,
        show_labels: bool = True,
        show_legend: bool = False,
        show_scale_bar: bool = True,
        show_cursor_info: bool = True,
        label_threshold_km: float = 400.0,  # labels hidden when more than this many km are visible
        max_zoom: float = 60.0,             # px per km
        background=(8, 10, 16),
    ):
        self.assets = assets
        self.title = title
        self.styles = dict(styles or DEFAULT_STYLES)
        self._ui_scale_fixed = ui_scale
        self.show_overlay = show_overlay
        self.show_labels = show_labels
        self.show_legend = show_legend
        self.show_scale_bar = show_scale_bar
        self.show_cursor_info = show_cursor_info
        self.label_threshold_km = label_threshold_km
        self.background = background

        self.cam = Camera(assets.world_w, assets.world_h, size, max_zoom)
        self._overlay: Optional[pygame.Surface] = None
        self._overlay_key = None
        self._layer_key = None
        self._labels: dict = {}
        self._alloc(size)

    # ---- size / ui scale ------------------------------------------------
    @property
    def size(self):
        return self.surface.get_size()

    @property
    def ui(self) -> float:
        if self._ui_scale_fixed is not None:
            return self._ui_scale_fixed
        return min(1.0, max(0.6, min(self.size) / 700.0))

    def _alloc(self, size):
        size = (max(1, int(size[0])), max(1, int(size[1])))
        self.surface = pygame.Surface(size).convert()
        self._layer = pygame.Surface(size).convert()
        self._layer_key = None
        self._labels.clear()  # fonts depend on tile size

    def resize(self, size):
        """Change tile size. Keeps the current camera centre/zoom (clamped to the new limits)."""
        self._alloc(size)
        self.cam.set_size(self.size)

    # ---- camera controls (coordinates are tile-local pixels) -----------------
    def fit(self):
        self.cam.fit()

    def pan_pixels(self, dx, dy):
        self.cam.pan_pixels(dx, dy)

    def zoom_at(self, sx, sy, factor):
        self.cam.zoom_at(sx, sy, factor)

    # ---- rendering ----------------------------------------------------------
    def render(self, snap: GameSnapshot, mouse_pos=None) -> pygame.Surface:
        """Draw `snap` into self.surface and return it. mouse_pos is tile-local or None."""
        s = self.surface
        W, H = self.size
        self._ensure_layer(snap)
        s.blit(self._layer, (0, 0))

        labels_on = self.show_labels and (W / self.cam.scale <= self.label_threshold_km)
        self._draw_ports(s, labels_on)
        self._draw_boats(s, snap, labels_on)
        self._draw_info_panel(s, snap)
        if self.show_legend:
            self._draw_legend(s)
        if self.show_scale_bar:
            self._draw_scale_bar(s)
        if self.show_cursor_info and mouse_pos is not None:
            self._draw_cursor_info(s, snap, mouse_pos)
        return s

    def _ensure_layer(self, snap: GameSnapshot):
        """Terrain + fish overlay. Only redrawn when camera / size / overlay content changes."""
        cam, W, H = self.cam, *self.size
        use_overlay = self.show_overlay and snap.fish is not None
        ov_key = (snap.game_id, snap.fish_version) if use_overlay else None
        key = (cam.cx, cam.cy, cam.scale, W, H, ov_key)
        if key == self._layer_key:
            return

        if use_overlay and ov_key != self._overlay_key:
            if snap.fish.shape != self.assets.shape:
                raise ValueError(f"fish field shape {snap.fish.shape} != map shape {self.assets.shape}")
            self._overlay = _fish_to_overlay(snap.fish)
            self._overlay_key = ov_key

        self._layer.fill(self.background)
        _blit_map_slice(self._layer, cam, self.assets.base, self.assets.res_km)
        if use_overlay:
            _blit_map_slice(self._layer, cam, self._overlay, self.assets.res_km)
        self._layer_key = key

    def _label(self, text, color, bg_alpha):
        key = (text, color, bg_alpha)
        surf = self._labels.get(key)
        if surf is None:
            txt = _font(max(8, round(12 * self.ui)), True).render(text, True, color)
            surf = pygame.Surface((txt.get_width() + 6, txt.get_height() + 2), pygame.SRCALPHA)
            surf.fill((5, 5, 5, bg_alpha))
            surf.blit(txt, (3, 1))
            self._labels[key] = surf
        return surf

    def _on_screen(self, sx, sy, margin=60):
        W, H = self.size
        return -margin <= sx <= W + margin and -margin <= sy <= H + margin

    def _draw_ports(self, s, labels_on):
        r = max(3, round(5 * self.ui))
        for p in self.assets.ports:
            sx, sy = self.cam.to_screen(p["grid_x"], p["grid_y"])
            if not self._on_screen(sx, sy):
                continue
            ix, iy = round(sx), round(sy)
            pygame.gfxdraw.filled_circle(s, ix, iy, r, PORT_COLOR)
            pygame.gfxdraw.aacircle(s, ix, iy, r, (255, 255, 255))
            if labels_on:
                lbl = self._label(p["name"], (255, 255, 255), 90)
                s.blit(lbl, (ix + r + 4, iy - lbl.get_height() // 2 + 4))

    def _draw_boats(self, s, snap: GameSnapshot, labels_on):
        cam, W, H = self.cam, *self.size
        x, y = np.asarray(snap.x_km, float), np.asarray(snap.y_km, float)
        role, heading = np.asarray(snap.role), np.asarray(snap.heading, float)
        sx = (x - cam.cx) * cam.scale + W / 2
        sy = H / 2 - (y - cam.cy) * cam.scale

        m = 60
        visible = (sx >= -m) & (sx <= W + m) & (sy >= -m) & (sy <= H + m)
        if snap.active is not None:
            visible &= np.asarray(snap.active, bool)

        size = 14.0 * self.ui
        for r, style in self.styles.items():
            role_mask = role == int(r)
            ranks = np.cumsum(role_mask)  # for default names: F1, F2, ...
            for i in np.nonzero(visible & role_mask)[0]:
                draw_boat(s, sx[i], sy[i], heading[i], size * style.size_mult, style.color)
                if labels_on:
                    name = snap.names[i] if snap.names is not None else f"{style.label_prefix}{ranks[i]}"
                    lbl = self._label(str(name), style.color, 130)
                    s.blit(lbl, (round(sx[i]) + 10, round(sy[i]) - 10 - lbl.get_height()))

    def _draw_info_panel(self, s, snap: GameSnapshot):
        font = _font(max(9, round(14 * self.ui)))
        title = self.title if self.title is not None else f"Game {snap.game_id}"
        lines = [(title, (255, 255, 255)), (f"t = {snap.tick}", (170, 170, 170))]
        for k, v in snap.info.items():
            lines.append((f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}", (230, 230, 230)))
        rendered = [font.render(t, True, c) for t, c in lines]
        pad = 5
        w = max(r.get_width() for r in rendered) + 2 * pad
        h = sum(r.get_height() for r in rendered) + 2 * pad
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((10, 10, 10, 190))
        y = pad
        for r in rendered:
            panel.blit(r, (pad, y))
            y += r.get_height()
        s.blit(panel, (6, 6))

    def _draw_legend(self, s):
        font = _font(max(9, round(14 * self.ui)))
        rows = [(st.name, st) for st in self.styles.values()] + [("Port", None)]
        row_h = round(24 * self.ui)
        w = round(150 * self.ui)
        panel = pygame.Surface((w, row_h * len(rows) + 10), pygame.SRCALPHA)
        panel.fill((17, 17, 17, 210))
        for i, (name, st) in enumerate(rows):
            cy = 5 + row_h // 2 + row_h * i
            if st is not None:
                draw_boat(panel, 16, cy, 0.0, 12 * self.ui, st.color)
            else:
                pygame.gfxdraw.filled_circle(panel, 16, cy, 5, PORT_COLOR)
                pygame.gfxdraw.aacircle(panel, 16, cy, 5, (255, 255, 255))
            panel.blit(font.render(name, True, (230, 230, 230)), (34, cy - font.get_height() // 2))
        s.blit(panel, (self.size[0] - panel.get_width() - 6, 6))

    def _draw_scale_bar(self, s):
        W, H = self.size
        km = max(1e-9, 120 * self.ui / self.cam.scale)
        mag = 10 ** math.floor(math.log10(km))
        nice = max(m * mag for m in (1, 2, 5) if m * mag <= km)
        length = nice * self.cam.scale
        x1, y = W - 14, H - round(34 * self.ui)
        x0 = x1 - length
        white = (255, 255, 255)
        pygame.draw.line(s, white, (x0, y), (x1, y), 2)
        pygame.draw.line(s, white, (x0, y - 5), (x0, y + 5), 2)
        pygame.draw.line(s, white, (x1, y - 5), (x1, y + 5), 2)
        lbl = _font(max(9, round(14 * self.ui))).render(f"{nice:g} km", True, white)
        s.blit(lbl, (x1 - length / 2 - lbl.get_width() / 2, y - lbl.get_height() - 4))

    def _draw_cursor_info(self, s, snap: GameSnapshot, mouse_pos):
        mx, my = mouse_pos
        W, H = self.size
        if not (0 <= mx < W and 0 <= my < H):
            return
        wx, wy = self.cam.to_world(mx, my)
        parts = [f"x {wx:.1f}  y {wy:.1f} km"]
        ll = self.assets.km_to_lonlat(wx, wy)
        if ll:
            parts.append(f"lon {ll[0]:.3f}  lat {ll[1]:.3f}")
        res = self.assets.res_km
        col, row = int(wx // res), int(wy // res)
        rows, cols = self.assets.shape
        if 0 <= col < cols and 0 <= row < rows:
            parts.append(f"elev {self.assets.height[row, col]:.0f} m")
            if snap.fish is not None:
                parts.append(f"biomass {snap.fish[row, col]:.2f}")
        font = _font(max(9, round(14 * self.ui)))
        txt = font.render("  ".join(parts), True, (230, 230, 230))
        bar = pygame.Surface((W, txt.get_height() + 6), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        bar.blit(txt, (6, 3))
        s.blit(bar, (0, H - bar.get_height()))
