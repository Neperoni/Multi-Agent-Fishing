import math
import numpy as np
import pygame
import pygame.gfxdraw
from multiagent_fishing.world import Layers


class MapView:
    V_MIN, V_MAX = -5000.0, 2000.0
    OCEAN_X = np.array([-5000, -3500, -2000, -500, 0], dtype=float)
    LAND_X = np.array([0, 500, 2000], dtype=float)
    OCEAN_COLORS = np.array([[0, 8, 20], [0, 29, 61], [0, 53, 102], [30, 96, 145], [76, 201, 240]], dtype=float)
    LAND_COLORS = np.array([[163, 177, 138], [88, 129, 87], [52, 78, 65]], dtype=float)

    # Ramps for the distance overlays: t in [0, 1] -> color
    SHORE_RAMP_X = np.array([0.0, 0.5, 1.0])
    SHORE_RAMP_C = np.array([[255, 240, 120], [255, 140, 40], [120, 20, 90]], dtype=float)   # near shore = bright
    PORT_RAMP_X = np.array([0.0, 0.33, 0.66, 1.0])
    PORT_RAMP_C = np.array([[253, 231, 37], [53, 183, 121], [49, 104, 142], [68, 1, 84]], dtype=float)  # viridis-ish

    OVERLAY_ORDER = ("shore", "port_dist", "biomass")  # compositing order, last is on top

    def __init__(self, map_data, port_color=(255, 0, 127), label_threshold_km=400.0):
        self.map_data = map_data
        self.port_color = port_color
        self.label_threshold_km = label_threshold_km

        L = self.map_data.layers
        self._base_rgb = self._colorize_height(L[..., Layers.HEIGHT])
        self._water = np.nan_to_num(L[..., Layers.HEIGHT], nan=self.V_MIN) < 0

        # Normalisation constants (computed once)
        self._shore_max = self._robust_max(L[..., Layers.SHORE_DIST], self._water)
        port_valid = self._water & (L[..., Layers.NEAREST_PORT_ID] >= 0)
        self._port_max = self._robust_max(L[..., Layers.NEAREST_PORT_DIST], port_valid)

        # Overlay toggles
        self.overlays = {"shore": False, "port_dist": False, "biomass": True}
        self.show_flow = False

        # Lazily built textures keyed by tuple of active overlay names
        self._textures = {(): self._to_surface(self._base_rgb)}
        self._layer_surface = None
        self._layer_cache_key = None

    # ------------------------------------------------------------------ toggles
    def toggle_overlay(self, name):
        self.overlays[name] = not self.overlays[name]

    def _active_key(self):
        return tuple(n for n in self.OVERLAY_ORDER if self.overlays[n])

    def _get_texture(self):
        key = self._active_key()
        if key not in self._textures:
            L = self.map_data.layers
            rgb = self._base_rgb
            for name in key:
                if name == "shore":
                    rgb = self._composite_shore_dist(rgb, L[..., Layers.SHORE_DIST])
                elif name == "port_dist":
                    rgb = self._composite_port_dist(
                        rgb, L[..., Layers.NEAREST_PORT_DIST], L[..., Layers.NEAREST_PORT_ID])
                elif name == "biomass":
                    rgb = self._composite_hotspots(rgb, L[..., Layers.BIOMASS])
            self._textures[key] = self._to_surface(rgb)
        return self._textures[key]

    # ------------------------------------------------------------------ drawing
    def draw_background(self, target_surf, cam):
        W, H = target_surf.get_size()
        current_map_texture = self._get_texture()
        res = self.map_data.resolution_km

        cache_key = (cam.cx, cam.cy, cam.scale, W, H, self._active_key())

        if cache_key != self._layer_cache_key:
            if self._layer_surface is None or self._layer_surface.get_size() != (W, H):
                self._layer_surface = pygame.Surface((W, H)).convert()
            self._layer_surface.fill((8, 10, 16))

            Wc, Hc = current_map_texture.get_size()
            left, right = cam.to_world(0, 0)[0], cam.to_world(W, 0)[0]
            top, bottom = cam.to_world(0, 0)[1], cam.to_world(0, H)[1]

            c0 = max(0, math.floor(left / res))
            c1 = min(Wc, math.ceil(right / res))
            r0 = max(0, math.floor(Hc - top / res))
            r1 = min(Hc, math.ceil(Hc - bottom / res))

            if c1 > c0 and r1 > r0:
                x0, y0 = cam.to_screen(c0 * res, (Hc - r0) * res)
                x1, y1 = cam.to_screen(c1 * res, (Hc - r1) * res)
                dx, dy = round(x0), round(y0)
                dw, dh = max(1, round(x1) - dx), max(1, round(y1) - dy)

                src_slice = current_map_texture.subsurface((c0, r0, c1 - c0, r1 - r0))
                if res * cam.scale < 1.0:
                    scaled = pygame.transform.smoothscale(src_slice, (dw, dh))
                else:
                    scaled = pygame.transform.scale(src_slice, (dw, dh))
                self._layer_surface.blit(scaled, (dx, dy))

            self._layer_cache_key = cache_key

        target_surf.blit(self._layer_surface, (0, 0))

    def draw_flow(self, target_surf, cam, spacing_px=30, length_px=16, color=(255, 255, 255)):
        """
        Vector-field overlay: a regular screen-space grid of short arrows pointing
        toward the nearest port. Only drawn over water cells that have a reachable port.
        """
        if not self.show_flow:
            return

        W, H = target_surf.get_size()
        res = self.map_data.resolution_km
        L = self.map_data.layers
        rows_n, cols_n = L.shape[:2]

        # Sample grid in screen space
        sx = np.arange(spacing_px / 2, W, spacing_px)
        sy = np.arange(spacing_px / 2, H, spacing_px)
        sxx, syy = np.meshgrid(sx, sy)

        # Camera is an axis-aligned affine map, so interpolate world coords from the corners
        wx0, wy0 = cam.to_world(0, 0)
        wx1, wy1 = cam.to_world(W, H)
        wx = wx0 + (sxx / W) * (wx1 - wx0)
        wy = wy0 + (syy / H) * (wy1 - wy0)

        cols = np.floor(wx / res).astype(int)
        rws = np.floor(wy / res).astype(int)
        inb = (cols >= 0) & (cols < cols_n) & (rws >= 0) & (rws < rows_n)
        cols, rws, sxx, syy = cols[inb], rws[inb], sxx[inb], syy[inb]

        cell = L[rws, cols]
        dirx = cell[:, Layers.NEAREST_PORT_DIR_X]
        diry = cell[:, Layers.NEAREST_PORT_DIR_Y]
        mag = np.hypot(dirx, diry)
        ok = (cell[:, Layers.NEAREST_PORT_ID] >= 0) & (cell[:, Layers.SHORE_DIST] > 0) & (mag > 1e-6)

        sxx, syy = sxx[ok], syy[ok]
        ux = dirx[ok] / mag[ok]
        uy = -diry[ok] / mag[ok]  # world +y is up, screen +y is down

        half = length_px / 2
        head = length_px * 0.35
        for x, y, vx, vy in zip(sxx, syy, ux, uy):
            tail = (x - vx * half, y - vy * half)
            tip = (x + vx * half, y + vy * half)
            pygame.draw.line(target_surf, color, tail, tip, 1)
            # arrow barbs: tip rotated back by +/- 150 degrees
            for ang in (2.62, -2.62):
                ca, sa = math.cos(ang), math.sin(ang)
                bx = vx * ca - vy * sa
                by = vx * sa + vy * ca
                pygame.draw.line(target_surf, color, tip, (tip[0] + bx * head, tip[1] + by * head), 1)

    def draw_ports(self, target_surf, cam, labels_on, label_cache):
        margin = 80
        for p in self.map_data.ports:
            sx, sy = cam.to_screen(p["grid_x"], p["grid_y"])
            if not (-margin <= sx <= target_surf.get_width() + margin and -margin <= sy <= target_surf.get_height() + margin):
                continue
            ix, iy = round(sx), round(sy)
            pygame.gfxdraw.filled_circle(target_surf, ix, iy, 5, self.port_color)
            pygame.gfxdraw.aacircle(target_surf, ix, iy, 5, (255, 255, 255))
            if labels_on:
                lbl = label_cache.get(p["name"], (255, 255, 255))
                target_surf.blit(lbl, (ix + 9, iy - lbl.get_height() // 2 + 4))

    # ------------------------------------------------------------ layer colorizers
    def _colorize_height(self, height):
        h = np.clip(np.nan_to_num(height, nan=self.V_MIN), self.V_MIN, self.V_MAX)
        out = np.empty(h.shape + (3,), dtype=float)
        for c in range(3):
            ocean = np.interp(h, self.OCEAN_X, self.OCEAN_COLORS[:, c])
            land = np.interp(h, self.LAND_X, self.LAND_COLORS[:, c])
            out[..., c] = np.where(h < 0, ocean, land)
        return out

    def _composite_hotspots(self, base_rgb, hotspots):
        v = np.clip(np.nan_to_num(hotspots), 0.0, 1.0)
        a = (0.85 * v)[..., None]
        red = np.stack([255 * (1 - 0.1 * v), 255 * 0.2 * (1 - v), np.zeros_like(v)], axis=-1)
        return base_rgb * (1 - a) + red * a

    def _composite_shore_dist(self, base_rgb, shore_dist, alpha=0.6):
        """Tint water by distance to land (sqrt-scaled so nearshore detail is visible)."""
        d = np.clip(np.nan_to_num(shore_dist, nan=0.0), 0.0, self._shore_max)
        t = np.sqrt(d / self._shore_max)
        t = 1.0 - t  # bright near the coast, dark offshore
        color = self._ramp(t, self.SHORE_RAMP_X, self.SHORE_RAMP_C)
        a = (alpha * self._water)[..., None]
        return base_rgb * (1 - a) + color * a

    def _composite_port_dist(self, base_rgb, port_dist, port_id, alpha=0.6):
        """Tint water by along-water distance to the nearest port; unreachable cells untouched."""
        valid = self._water & (port_id >= 0) & np.isfinite(port_dist)
        d = np.clip(np.nan_to_num(port_dist, nan=0.0), 0.0, self._port_max)
        t = d / self._port_max
        color = self._ramp(t, self.PORT_RAMP_X, self.PORT_RAMP_C)
        a = (alpha * valid)[..., None]
        return base_rgb * (1 - a) + color * a

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ramp(t, xs, colors):
        return np.stack([np.interp(t, xs, colors[:, c]) for c in range(3)], axis=-1)

    @staticmethod
    def _robust_max(values, mask):
        v = values[mask & np.isfinite(values)]
        return float(np.percentile(v, 99)) if v.size else 1.0

    def _to_surface(self, rgb):
        arr = np.clip(rgb[::-1], 0, 255).transpose(1, 0, 2)
        return pygame.surfarray.make_surface(np.ascontiguousarray(arr).astype(np.uint8)).convert()