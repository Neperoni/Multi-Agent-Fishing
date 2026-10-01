"""
MARL fishing map viewer (pygame)

Controls
    left/middle drag ... pan
    mouse wheel ........ zoom (about the cursor)
    arrows / WASD ...... pan        + / - ... zoom
    H .................. toggle biomass overlay
    L .................. toggle labels
    R .................. reset view
    Esc ................ quit

Requires: pygame-ce (or pygame >= 2), numpy, xarray, scipy, opensimplex
"""
import json
import math
from pathlib import Path

import numpy as np
import pygame
import pygame.gfxdraw
import xarray as xr
from opensimplex import OpenSimplex
from scipy.ndimage import distance_transform_edt

# =========================================================================
# CONFIG
# =========================================================================
FILE_PATH = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
PORTS_PATH = "datasets/ports.json"
CACHE_PATH = Path("datasets/cache/marl_cache.npz")
RESOLUTION = 1.0  # km per grid cell
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
HOTSPOT_PARAMS = {"seed": 42, "noise_feature_scale": 30.0, "fade_margin_px": 5.0}

KM_PER_DEGREE = 111.32
V_MIN, V_MAX = -5000.0, 2000.0
LABEL_THRESHOLD_KM = 400.0  # hide labels when more than this many km are visible
MAX_SCALE = 60.0            # max zoom, in screen pixels per km
BOAT_SIZE_PX = 14.0         # boat size on screen (constant regardless of zoom)

BG_COLOR = (8, 10, 16)
FISHING_COLOR = (255, 204, 0)
PATROL_COLOR = (0, 255, 102)
PORT_COLOR = (255, 0, 127)


# =========================================================================
# DATA (same logic as the matplotlib version)
# =========================================================================
def generate_dynamic_hotspots(height_grid, seed=42, noise_feature_scale=30.0, fade_margin_px=5.0):
    mask = (height_grid >= -220) & (height_grid <= -80)
    if np.any(mask):
        ramp = np.clip(distance_transform_edt(mask) / fade_margin_px, 0.0, 1.0)
    else:
        ramp = np.zeros_like(height_grid, dtype=float)

    gen = OpenSimplex(seed=seed)
    ny, nx = height_grid.shape
    yi, xi = np.indices((ny, nx))
    noise = np.vectorize(gen.noise2)(xi / noise_feature_scale, yi / noise_feature_scale)
    noise = (noise + 1.0) / 2.0
    return ramp * noise


def get_active_simulation_ports(ports_list, bounds):
    center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2.0
    lon_scale = np.cos(np.radians(center_lat))
    out = []
    for port in ports_list:
        lat, lon = port["lat"], port["lon"]
        if bounds["lon_min"] <= lon <= bounds["lon_max"] and bounds["lat_min"] <= lat <= bounds["lat_max"]:
            out.append({
                "name": port["name"].split("(")[0].strip(),
                "grid_x": (lon - bounds["lon_min"]) * KM_PER_DEGREE * lon_scale,
                "grid_y": (lat - bounds["lat_min"]) * KM_PER_DEGREE,
                "jurisdiction": port.get("jurisdiction", ""),
            })
    return out


def load_and_interpolate_grid(file_path, grid_res_km=1.0, bounding_box=None):
    ds = xr.open_dataset(file_path)
    height_var = "elevation" if "elevation" in ds else "height"
    if bounding_box is not None:
        ds = ds.sel(
            lon=slice(bounding_box["lon_min"], bounding_box["lon_max"]),
            lat=slice(bounding_box["lat_min"], bounding_box["lat_max"]),
        )
    lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
    lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())

    lon_scale = np.cos(np.radians((lat_min + lat_max) / 2.0))
    width_km = (lon_max - lon_min) * KM_PER_DEGREE * lon_scale
    height_km = (lat_max - lat_min) * KM_PER_DEGREE

    x_km = np.arange(0, width_km, grid_res_km)
    y_km = np.arange(0, height_km, grid_res_km)
    lons = lon_min + x_km / (KM_PER_DEGREE * lon_scale)
    lats = lat_min + y_km / KM_PER_DEGREE
    lons2d, lats2d = np.meshgrid(lons, lats)

    grid = ds[height_var].interp(
        lon=xr.DataArray(lons2d, dims=("y", "x")),
        lat=xr.DataArray(lats2d, dims=("y", "x")),
        method="linear",
    ).values
    return grid, width_km, height_km


def load_world():
    """Loads the height + hotspot grids, using an on-disk cache keyed on the parameters."""
    key = json.dumps(
        {"file": FILE_PATH, "res": RESOLUTION, "box": COASTAL_SUBSET, "hot": HOTSPOT_PARAMS},
        sort_keys=True,
    )
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH) as z:
            if str(z["key"]) == key:
                return z["height"], z["hotspots"]

    height, _, _ = load_and_interpolate_grid(FILE_PATH, RESOLUTION, COASTAL_SUBSET)
    hotspots = generate_dynamic_hotspots(height, **HOTSPOT_PARAMS)
    np.savez_compressed(CACHE_PATH, key=key, height=height, hotspots=hotspots)
    return height, hotspots


# =========================================================================
# MAP COLORING (replicates the matplotlib colormap + red alpha overlay)
# =========================================================================
def _hex(c):
    return [int(c[i:i + 2], 16) for i in (1, 3, 5)]


OCEAN_X = np.array([-5000, -3500, -2000, -500, 0], dtype=float)
OCEAN_C = np.array([_hex(c) for c in ("#000814", "#001d3d", "#003566", "#1e6091", "#4cc9f0")], dtype=float)
LAND_X = np.array([0, 500, 2000], dtype=float)
LAND_C = np.array([_hex(c) for c in ("#a3b18a", "#588157", "#344e41")], dtype=float)


def colorize_height(height):
    """Float RGB (rows, cols, 3). Hard edge at sea level, like the original colormap."""
    h = np.clip(np.nan_to_num(height, nan=V_MIN), V_MIN, V_MAX)
    out = np.empty(h.shape + (3,), dtype=float)
    for c in range(3):
        ocean = np.interp(h, OCEAN_X, OCEAN_C[:, c])
        land = np.interp(h, LAND_X, LAND_C[:, c])
        out[..., c] = np.where(h < 0, ocean, land)
    return out


def composite_hotspots(base_rgb, hotspots):
    v = np.clip(np.nan_to_num(hotspots), 0.0, 1.0)
    a = (0.85 * v)[..., None]  # alpha ramps 0 -> 0.85
    red = np.stack([255 * (1 - 0.1 * v), 255 * 0.2 * (1 - v), np.zeros_like(v)], axis=-1)
    return base_rgb * (1 - a) + red * a


def to_surface(rgb):
    """(rows, cols, 3) with row 0 = south  ->  pygame surface with north at the top."""
    arr = np.clip(rgb[::-1], 0, 255).transpose(1, 0, 2)
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr).astype(np.uint8)).convert()


# =========================================================================
# CAMERA  (world units = km, +Y is north; screen +Y is down)
# =========================================================================
class Camera:
    def __init__(self, world_w, world_h):
        self.world_w, self.world_h = world_w, world_h
        self.cx = world_w / 2
        self.cy = world_h / 2
        self.scale = 1.0  # px per km
        self.min_scale = 0.1
        self.W = self.H = 1

    def fit(self, W, H):
        self.W, self.H = W, H
        self.scale = 0.95 * min(W / self.world_w, H / self.world_h)
        self.min_scale = self.scale * 0.5
        self.cx, self.cy = self.world_w / 2, self.world_h / 2

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
        self.scale = min(max(self.scale * factor, self.min_scale), MAX_SCALE)
        self.cx = wx - (sx - self.W / 2) / self.scale
        self.cy = wy + (sy - self.H / 2) / self.scale
        self._clamp()


# =========================================================================
# DRAWING
# =========================================================================
# Boat = square hull + triangle bow, pointing up (+y) at heading 0
HULL_SQUARE = [(-0.3, 0.2), (0.3, 0.2), (0.3, -0.5), (-0.3, -0.5)]
BOW_TRIANGLE = [(-0.3, 0.2), (0.3, 0.2), (0.0, 0.6)]


def draw_boat(surf, sx, sy, heading, size, fill, outline=(0, 0, 0)):
    """heading in radians; 0 = north/up, increasing clockwise."""
    c, s = math.cos(heading), math.sin(heading)

    def xf(pts):
        return [(round(sx + (vx * c + vy * s) * size), round(sy + (vx * s - vy * c) * size)) for vx, vy in pts]

    for poly in (xf(HULL_SQUARE), xf(BOW_TRIANGLE)):
        pygame.gfxdraw.filled_polygon(surf, poly, fill)
        pygame.gfxdraw.aapolygon(surf, poly, outline)


class LabelCache:
    """Pre-renders text with a translucent dark box so labels are cheap to blit."""

    def __init__(self, font, bg_alpha):
        self.font, self.bg_alpha, self.cache = font, bg_alpha, {}

    def get(self, text, color):
        key = (text, color)
        if key not in self.cache:
            txt = self.font.render(text, True, color)
            surf = pygame.Surface((txt.get_width() + 6, txt.get_height() + 2), pygame.SRCALPHA)
            surf.fill((5, 5, 5, self.bg_alpha))
            surf.blit(txt, (3, 1))
            self.cache[key] = surf
        return self.cache[key]


def draw_map(target, cam, map_surf, res):
    """Blit only the visible slice of the pre-rendered map, scaled to the current zoom."""
    Wc, Hc = map_surf.get_size()  # cells; surface row 0 = north edge
    left, right = cam.to_world(0, 0)[0], cam.to_world(cam.W, 0)[0]
    top, bottom = cam.to_world(0, 0)[1], cam.to_world(0, cam.H)[1]

    c0 = max(0, math.floor(left / res))
    c1 = min(Wc, math.ceil(right / res))
    r0 = max(0, math.floor(Hc - top / res))
    r1 = min(Hc, math.ceil(Hc - bottom / res))
    if c1 <= c0 or r1 <= r0:
        return

    x0, y0 = cam.to_screen(c0 * res, (Hc - r0) * res)
    x1, y1 = cam.to_screen(c1 * res, (Hc - r1) * res)
    dx, dy = round(x0), round(y0)
    dw, dh = max(1, round(x1) - dx), max(1, round(y1) - dy)

    src = map_surf.subsurface((c0, r0, c1 - c0, r1 - r0))
    if res * cam.scale < 1.0:  # zoomed out: smooth downsample; zoomed in: crisp cells
        scaled = pygame.transform.smoothscale(src, (dw, dh))
    else:
        scaled = pygame.transform.scale(src, (dw, dh))
    target.blit(scaled, (dx, dy))


def draw_scale_bar(screen, cam, font):
    km = 140 / cam.scale
    mag = 10 ** math.floor(math.log10(km))
    nice = max(m * mag for m in (1, 2, 5) if m * mag <= km)
    length = nice * cam.scale
    x1, y = cam.W - 20, cam.H - 40
    x0 = x1 - length
    pygame.draw.line(screen, (255, 255, 255), (x0, y), (x1, y), 2)
    pygame.draw.line(screen, (255, 255, 255), (x0, y - 5), (x0, y + 5), 2)
    pygame.draw.line(screen, (255, 255, 255), (x1, y - 5), (x1, y + 5), 2)
    lbl = font.render(f"{nice:g} km", True, (255, 255, 255))
    screen.blit(lbl, (x1 - length / 2 - lbl.get_width() / 2, y - 22))


def draw_legend(screen, font):
    rows = [("Fishing Boat", "fish"), ("Patrol Boat", "patrol"), ("Port", "port")]
    panel = pygame.Surface((150, 12 + 24 * len(rows)), pygame.SRCALPHA)
    panel.fill((17, 17, 17, 210))
    screen.blit(panel, (10, 10))
    for i, (name, kind) in enumerate(rows):
        cy = 10 + 18 + 24 * i
        if kind == "fish":
            draw_boat(screen, 28, cy, 0.0, 12, FISHING_COLOR)
        elif kind == "patrol":
            draw_boat(screen, 28, cy, 0.0, 12, PATROL_COLOR)
        else:
            pygame.gfxdraw.filled_circle(screen, 28, cy, 5, PORT_COLOR)
            pygame.gfxdraw.aacircle(screen, 28, cy, 5, (255, 255, 255))
        screen.blit(font.render(name, True, (230, 230, 230)), (48, cy - 8))


# =========================================================================
# MAIN LOOP
# =========================================================================
def run(height, hotspots, ports, fishing_boats, patrol_boats, max_frames=None):
    screen = pygame.display.get_surface()

    base_rgb = colorize_height(height)
    surf_plain = to_surface(base_rgb)
    surf_hot = to_surface(composite_hotspots(base_rgb, hotspots))

    Hc, Wc = height.shape
    world_w, world_h = Wc * RESOLUTION, Hc * RESOLUTION
    cam = Camera(world_w, world_h)
    cam.fit(*screen.get_size())

    mono = "consolas,menlo,dejavusansmono,monospace"
    font = pygame.font.SysFont(mono, 14)
    label_font = pygame.font.SysFont(mono, 12, bold=True)
    port_labels = LabelCache(label_font, 90)
    boat_labels = LabelCache(label_font, 130)

    lon_scale = np.cos(np.radians((COASTAL_SUBSET["lat_min"] + COASTAL_SUBSET["lat_max"]) / 2))

    show_hot, show_labels = True, True
    layer, layer_key = None, None
    dragging = False
    clock = pygame.time.Clock()
    frame = 0
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        screen = pygame.display.get_surface()
        W, H = screen.get_size()
        cam.W, cam.H = W, H

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_h:
                    show_hot = not show_hot
                elif ev.key == pygame.K_l:
                    show_labels = not show_labels
                elif ev.key == pygame.K_r:
                    cam.fit(W, H)
            elif ev.type == pygame.MOUSEWHEEL:
                cam.zoom_at(*pygame.mouse.get_pos(), 1.2 ** ev.y)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 2):
                dragging = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button in (1, 2):
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                cam.pan_pixels(*ev.rel)

        keys = pygame.key.get_pressed()
        step = 600 * dt
        dx = (keys[pygame.K_LEFT] or keys[pygame.K_a]) - (keys[pygame.K_RIGHT] or keys[pygame.K_d])
        dy = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
        if dx or dy:
            cam.pan_pixels(dx * step, dy * step)
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            cam.zoom_at(W / 2, H / 2, 1 + 1.5 * dt)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            cam.zoom_at(W / 2, H / 2, 1 / (1 + 1.5 * dt))

        # --- map layer: only re-rendered when the camera/size/overlay changes ---
        key = (cam.cx, cam.cy, cam.scale, W, H, show_hot)
        if key != layer_key:
            if layer is None or layer.get_size() != (W, H):
                layer = pygame.Surface((W, H)).convert()
            layer.fill(BG_COLOR)
            draw_map(layer, cam, surf_hot if show_hot else surf_plain, RESOLUTION)
            layer_key = key
        screen.blit(layer, (0, 0))

        labels_on = show_labels and (W / cam.scale <= LABEL_THRESHOLD_KM)
        margin = 80

        def visible(sx, sy):
            return -margin <= sx <= W + margin and -margin <= sy <= H + margin

        # --- ports ---
        for p in ports:
            sx, sy = cam.to_screen(p["grid_x"], p["grid_y"])
            if not visible(sx, sy):
                continue
            ix, iy = round(sx), round(sy)
            pygame.gfxdraw.filled_circle(screen, ix, iy, 5, PORT_COLOR)
            pygame.gfxdraw.aacircle(screen, ix, iy, 5, (255, 255, 255))
            if labels_on:
                lbl = port_labels.get(p["name"], (255, 255, 255))
                screen.blit(lbl, (ix + 9, iy - lbl.get_height() // 2 + 4))

        # --- boats (patrol drawn on top of fishing) ---
        for fleet, color, mult, default_prefix in (
            (fishing_boats, FISHING_COLOR, 1.0, "F"),
            (patrol_boats, PATROL_COLOR, 1.15, "P"),
        ):
            for i, b in enumerate(fleet):
                sx, sy = cam.to_screen(b["x_km"], b["y_km"])
                if not visible(sx, sy):
                    continue
                draw_boat(screen, sx, sy, b.get("heading", 0.0), BOAT_SIZE_PX * mult, color)
                if labels_on:
                    lbl = boat_labels.get(b.get("name", f"{default_prefix}{i + 1}"), color)
                    screen.blit(lbl, (round(sx) + 10, round(sy) - 10 - lbl.get_height()))

        # --- HUD ---
        draw_legend(screen, font)
        draw_scale_bar(screen, cam, font)

        mx, my = pygame.mouse.get_pos()
        wx, wy = cam.to_world(mx, my)
        info = f"x {wx:7.1f} km  y {wy:7.1f} km"
        info += f"  lon {COASTAL_SUBSET['lon_min'] + wx / (KM_PER_DEGREE * lon_scale):8.3f}"
        info += f"  lat {COASTAL_SUBSET['lat_min'] + wy / KM_PER_DEGREE:7.3f}"
        col, row = int(wx // RESOLUTION), int(wy // RESOLUTION)
        if 0 <= col < Wc and 0 <= row < Hc:
            info += f"  elev {height[row, col]:7.0f} m  biomass {hotspots[row, col]:.2f}"
        hint = "drag: pan | wheel: zoom | H: overlay | L: labels | R: reset"
        bar = pygame.Surface((W, 24), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        screen.blit(bar, (0, H - 24))
        screen.blit(font.render(info, True, (230, 230, 230)), (8, H - 21))
        h_surf = font.render(hint, True, (150, 150, 150))
        screen.blit(h_surf, (W - h_surf.get_width() - 8, H - 21))

        pygame.display.flip()
        frame += 1
        if max_frames is not None and frame >= max_frames:
            break


def main():
    pygame.init()
    pygame.display.set_mode((1280, 900), pygame.RESIZABLE)
    pygame.display.set_caption("MARL Fishing Environment")

    print("Loading map data (cached after first run)...")
    height, hotspots = load_world()

    with open(PORTS_PATH, "r", encoding="utf-8") as f:
        ports = get_active_simulation_ports(json.load(f), COASTAL_SUBSET)

    fishing_boats = [
        {"x_km": 45.0, "y_km": 120.0, "heading": np.pi / 4, "name": "ORION"},
        {"x_km": 90.0, "y_km": 340.0, "heading": np.pi, "name": "MARINER"},
        {"x_km": 150.0, "y_km": 510.0, "heading": 1.5 * np.pi, "name": "SEAWOLF"},
    ]
    patrol_boats = [
        {"x_km": 60.0, "y_km": 150.0, "heading": 0.0, "name": "PATROL_A"},
        {"x_km": 110.0, "y_km": 400.0, "heading": 0.75 * np.pi, "name": "PATROL_B"},
    ]

    run(height, hotspots, ports, fishing_boats, patrol_boats)
    pygame.quit()


if __name__ == "__main__":
    main()