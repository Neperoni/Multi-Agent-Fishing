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

Requires: pygame-ce (or pygame >= 2), numpy, xarray, scipy, opensimplex, multiagent_fishing
"""
import json
import math
from pathlib import Path

import numpy as np
import pygame
import pygame.gfxdraw

# Import the decoupled module namespace
import multiagent_fishing as mf


# =========================================================================
# CONFIG
# =========================================================================
FILE_PATH = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
PORTS_PATH = "datasets/ports.json"
RESOLUTION = 1.0  # km per grid cell
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42

KM_PER_DEGREE = 111.32
V_MIN, V_MAX = -5000.0, 2000.0
LABEL_THRESHOLD_KM = 400.0  # hide labels when more than this many km are visible
MAX_SCALE = 60.0            # max zoom, in screen pixels per km
BOAT_SIZE_PX = 14.0         # boat size on screen (constant regardless of zoom)

BG_COLOR = (8, 10, 16)
FISHING_COLOR = (255, 204, 0)
PATROL_COLOR = (0, 255, 102)
PORT_COLOR = (255, 0, 127)


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
HULL_SQUARE = [(-0.3, 0.2), (0.3, 0.2), (0.3, -0.5), (-0.3, -0.5)]
BOW_TRIANGLE = [(-0.3, 0.2), (0.3, 0.2), (0.0, 0.6)]


def draw_boat(surf, sx, sy, heading, size, fill, outline=(0, 0, 0)):
    c, s = math.cos(heading), math.sin(heading)

    def xf(pts):
        return [(round(sx + (vx * c + vy * s) * size), round(sy + (vx * s - vy * c) * size)) for vx, vy in pts]

    for poly in (xf(HULL_SQUARE), xf(BOW_TRIANGLE)):
        pygame.gfxdraw.filled_polygon(surf, poly, fill)
        pygame.gfxdraw.aapolygon(surf, poly, outline)


class LabelCache:
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
    Wc, Hc = map_surf.get_size()
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
    if res * cam.scale < 1.0:
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
def run(map_data, boat_views, max_frames=None):
    screen = pygame.display.get_surface()

    # Extract clean raw numpy grids directly out of the map data instance
    height = map_data.height_data
    hotspots = map_data.biomass_data

    base_rgb = colorize_height(height)
    surf_plain = to_surface(base_rgb)
    surf_hot = to_surface(composite_hotspots(base_rgb, hotspots))

    Hc, Wc = map_data.shape
    world_w, world_h = map_data.width_km, map_data.height_km
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

        # --- ports (Reading directly from map_data) ---
        for p in map_data.ports:
            sx, sy = cam.to_screen(p["grid_x"], p["grid_y"])
            if not visible(sx, sy):
                continue
            ix, iy = round(sx), round(sy)
            pygame.gfxdraw.filled_circle(screen, ix, iy, 5, PORT_COLOR)
            pygame.gfxdraw.aacircle(screen, ix, iy, 5, (255, 255, 255))
            if labels_on:
                lbl = port_labels.get(p["name"], (255, 255, 255))
                screen.blit(lbl, (ix + 9, iy - lbl.get_height() // 2 + 4))

        # --- boats (Clean polymorphic loop over all unified BoatView items) ---
        for view in boat_views:
            view.draw(screen, cam, labels_on, boat_labels)

        # --- HUD ---
        draw_legend(screen, font)
        draw_scale_bar(screen, cam, font)

        mx, my = pygame.mouse.get_pos()
        wx, wy = cam.to_world(mx, my)
        info = f"x {wx:7.1f} km  y {wy:7.1f} km"
        info += f"  lon {COASTAL_SUBSET['lon_min'] + wx / (KM_PER_DEGREE * lon_scale):8.3f}"
        info += f"  lat {COASTAL_SUBSET['lat_min'] + wy / KM_PER_DEGREE:7.3f}"
        
        col = int(wx // map_data.km_per_pixel) if map_data.km_per_pixel > 0 else 0
        row = int(wy // map_data.km_per_pixel) if map_data.km_per_pixel > 0 else 0
        
        if 0 <= col < Wc and 0 <= row < Hc:
            info += f"  elev {map_data.get_height_at_pixel(col, row):7.0f} m  biomass {map_data.get_biomass_at_pixel(col, row):.2f}"
            
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

    print("Loading data via package MapGenerator...")
    # Pass PORTS_PATH directly into the generator constructor
    generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
    map_data = generator.get_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)

    ports = map_data.ports

    # Instantiate the underlying pure simulation layer entities
    sim_poachers = [
        mf.simulation.PoacherVessel("ORION", start_x=45.0, start_y=120.0),
        mf.simulation.PoacherVessel("MARINER", start_x=90.0, start_y=340.0),
        mf.simulation.PoacherVessel("SEAWOLF", start_x=150.0, start_y=510.0)
    ]
    sim_patrols = [
        mf.simulation.PatrolVessel("PATROL_A", start_x=60.0, start_y=150.0),
        mf.simulation.PatrolVessel("PATROL_B", start_x=110.0, start_y=400.0)
    ]

    # Initialize their headings for display
    sim_poachers[0].current_heading = np.pi / 4
    sim_poachers[1].current_heading = np.pi
    sim_poachers[2].current_heading = 1.5 * np.pi
    sim_patrols[1].current_heading = 0.75 * np.pi

    boat_views = []
    for p in sim_poachers:
        boat_views.append(mf.rendering.BoatView(p, FISHING_COLOR, boat_size_px=12.0))
    for p in sim_patrols:
        boat_views.append(mf.rendering.BoatView(p, PATROL_COLOR, boat_size_px=12.0))

    run(map_data, boat_views)
    pygame.quit()


if __name__ == "__main__":
    main()
