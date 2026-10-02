"""
MARL fishing map viewer (pygame)

Controls
    left/middle drag ... pan
    mouse wheel ........ zoom (about the cursor)
    arrows / WASD ...... pan        + / - ... zoom
    H .................. toggle biomass overlay
    L .................. toggle labels
    R .................. reset view
    Space .............. pause/resume simulation
    Right-click drag ... measure distance and bearing
    Esc ................ quit

Requires: pygame-ce (or pygame >= 2), numpy, xarray, scipy, opensimplex, multiagent_fishing
"""
import math
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
SHIP_LABEL_THRESHOLD_KM = 4000.0  # Basically always on
PORT_LABEL_THRESHOLD_KM = 700.0   
CARGO_LABEL_THRESHOLD_KM = 350.0  # Cargo displays when fairly close
VELOCITY_LABEL_THRESHOLD_KM = 150.0 # Velocity displays only at extreme zoom

MAX_SCALE = 60.0            # max zoom, in screen pixels per km

BG_COLOR = (8, 10, 16)
FISHING_COLOR = (255, 204, 0)
PATROL_COLOR = (0, 255, 102)
PORT_COLOR = (255, 0, 127)


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

import json
import random
from pathlib import Path

# Global memory cache to hold name arrays so we only read the disk once at startup
_FRONTEND_NAME_POOL = None

def spawn_random_vessel(env, boat_views_list, is_poacher, names_json_path="datasets/boat_names.json"):
    global _FRONTEND_NAME_POOL
    
    if not env.map_data.ports:
        return None

    if _FRONTEND_NAME_POOL is None:
        p = Path(names_json_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _FRONTEND_NAME_POOL = json.load(f)
        else:
            _FRONTEND_NAME_POOL = {
                "fishing_boats": {"first_word": ["MYSTIC"], "last_word": ["WAVE"]},
                "patrol_boats": {"first_word": ["VIGILANT"], "last_word": ["GUARDIAN"]}
            }

    port = random.choice(env.map_data.ports)
    pool_key = "fishing_boats" if is_poacher else "patrol_boats"
    pool = _FRONTEND_NAME_POOL.get(pool_key, {})
    
    first = random.choice(pool.get("first_word", ["MYSTIC" if is_poacher else "VIGILANT"]))
    last = random.choice(pool.get("last_word", ["WAVE" if is_poacher else "GUARDIAN"]))
    vessel_name = f"{first} {last}"
    
    # Instantiate the vessel WITHOUT an ID (None)
    if is_poacher:
        new_vessel = mf.simulation.PoacherVessel(
            name=vessel_name, 
            start_x=port["grid_x"], 
            start_y=port["grid_y"],
            start_heading=random.uniform(0.0, 2 * math.pi)
        )
        # Environment assigns the ID right here and returns it
        vessel_id = env.add_vessel(new_vessel)
        color = FISHING_COLOR
    else:
        new_vessel = mf.simulation.PatrolVessel(
            name=vessel_name, 
            start_x=port["grid_x"], 
            start_y=port["grid_y"],
            start_heading=random.uniform(0.0, 2 * math.pi)
        )
        vessel_id = env.add_vessel(new_vessel)
        color = PATROL_COLOR

    new_view = mf.rendering.BoatView(new_vessel, color, boat_size_px=12.0)
    boat_views_list.append(new_view)
    
    return vessel_id

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


def draw_legend(screen, font, legend_fish, legend_patrol):
    rows = [("Fishing Boat", "fish"), ("Patrol Boat", "patrol"), ("Port", "port")]
    panel = pygame.Surface((150, 12 + 24 * len(rows)), pygame.SRCALPHA)
    panel.fill((17, 17, 17, 210))
    screen.blit(panel, (10, 10))
    
    class UI_Camera:
        W = H = 9999
        def to_screen(self, x, y): return x, y
    ui_cam = UI_Camera()

    for i, (name, kind) in enumerate(rows):
        cy = 10 + 18 + 24 * i
        if kind == "fish":
            legend_fish.vessel.x, legend_fish.vessel.y = 28, cy
            legend_fish.draw(screen, ui_cam, labels_on=False, cargo_on=False, velocity_on=False, label_cache=None)
        elif kind == "patrol":
            legend_patrol.vessel.x, legend_patrol.vessel.y = 28, cy
            legend_patrol.draw(screen, ui_cam, labels_on=False, cargo_on=False, velocity_on=False, label_cache=None)
        else:
            pygame.gfxdraw.filled_circle(screen, 28, cy, 5, PORT_COLOR)
            pygame.gfxdraw.aacircle(screen, 28, cy, 5, (255, 255, 255))
            
        screen.blit(font.render(name, True, (230, 230, 230)), (48, cy - 8))






# =========================================================================
# MAIN LOOP
# =========================================================================
def run(env, boat_views, map_view, max_frames=None):
    screen = pygame.display.get_surface()

    world_w = map_view.map_data.width_km
    world_h = map_view.map_data.height_km
    
    cam = Camera(world_w, world_h)
    cam.fit(*screen.get_size())

    mono = "consolas,menlo,dejavusansmono,monospace"
    font = pygame.font.SysFont(mono, 14)
    label_font = pygame.font.SysFont(mono, 12, bold=True)
    port_labels = LabelCache(label_font, 90)
    boat_labels = LabelCache(label_font, 130)

    dummy_fish_vessel = mf.simulation.PoacherVessel(name="DummyFish", start_x=0.0, start_y=0.0, start_heading=0.0)
    dummy_patrol_vessel = mf.simulation.PatrolVessel(name="DummyPatrol", start_x=0.0, start_y=0.0, start_heading=0.0)

    legend_fish = mf.rendering.BoatView(dummy_fish_vessel, FISHING_COLOR, boat_size_px=12.0)
    legend_patrol = mf.rendering.BoatView(dummy_patrol_vessel, PATROL_COLOR, boat_size_px=12.0)

    center_lat = (COASTAL_SUBSET["lat_min"] + COASTAL_SUBSET["lat_max"]) / 2
    lon_scale = np.cos(np.radians(center_lat))

    ui_messages = []
    show_labels = True
    dragging = False
    is_paused = False  # Runtime Simulation Pause Toggle State
    
    # Measuring Tool Vector States
    measuring = False
    measure_start_screen = (0, 0)

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
                    map_view.show_biomass = not map_view.show_biomass
                elif ev.key == pygame.K_l:
                    show_labels = not show_labels
                elif ev.key == pygame.K_r:
                    cam.fit(W, H)
                elif ev.key == pygame.K_SPACE:
                    is_paused = not is_paused  # Toggle active physics state updates
                    ui_messages.append("SIMULATION PAUSED" if is_paused else "SIMULATION RESUMED")
                    if len(ui_messages) > 20: ui_messages.pop(0)
            elif ev.type == pygame.MOUSEWHEEL:
                cam.zoom_at(*pygame.mouse.get_pos(), 1.2 ** ev.y)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button in (1, 2):
                    dragging = True
                elif ev.button == 3:  # Right-click down: spawn ruler anchor node
                    measuring = True
                    measure_start_screen = ev.pos
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button in (1, 2):
                    dragging = False
                elif ev.button == 3:  # Right-click release: scrub ruler line
                    measuring = False
            elif ev.type == pygame.MOUSEMOTION:
                if dragging:
                    cam.pan_pixels(*ev.rel)

        keys = pygame.key.get_pressed()
        step_speed = 600 * dt
        dx = (keys[pygame.K_LEFT] or keys[pygame.K_a]) - (keys[pygame.K_RIGHT] or keys[pygame.K_d])
        dy = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
        if dx or dy:
            cam.pan_pixels(dx * step_speed, dy * step_speed)
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            cam.zoom_at(W / 2, H / 2, 1 + 1.5 * dt)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            cam.zoom_at(W / 2, H / 2, 1 / (1 + 1.5 * dt))

        # =========================================================================
        # UPDATE SIMULATION PHYSICS STEP
        # =========================================================================
        if not is_paused:

            actions = {}
        #generating dummy actions
            for v in boat_views:
                vessel_id = v.vessel.id
                heading = v.vessel.current_heading
                
                # Check if this vessel is a poacher to supply the fish_toggle parameter
                if hasattr(v.vessel, 'is_fishing'):
                    # For a simple test, we can keep its current fishing toggle state,
                    # or force it to fish (1) if it isn't full yet.
                    fish_toggle = 1 if v.vessel.cargo < v.vessel.max_cargo else 0
                    actions[vessel_id] = (heading, 0.5, fish_toggle)
                else:
                    # Patrol boats only accept (heading, speed)
                    actions[vessel_id] = (heading, 0.5)


        #Main physics call
        # =========================================================================
            step_events = env.step(actions, dt=dt)
            
            for app in step_events.get("apprehensions", []):
                p_id, pat_id = app["poacher_id"], app["by_patrol_id"]
                p_name = env.poachers[p_id].name if p_id in env.poachers else f"Poacher {p_id}"
                pat_name = env.patrol_boats[pat_id].name if pat_id in env.patrol_boats else f"Patrol {pat_id}"
                ui_messages.append(f"ALERT: {p_name} captured by {pat_name}!")
                # Apprehensions are strictly poachers being caught
                spawn_random_vessel(env, boat_views, is_poacher=True)

            for g_id in step_events.get("groundings", []):
                is_poach = g_id in env.poachers
                name = env.poachers[g_id].name if is_poach else (env.patrol_boats[g_id].name if g_id in env.patrol_boats else f"Vessel {g_id}")
                ui_messages.append(f"CRASH: {name} ran aground on land!")
                # Dynamically match the exact ship type that crashed
                spawn_random_vessel(env, boat_views, is_poacher=is_poach)

            for o_id in step_events.get("out_of_bounds", []):
                is_poach = o_id in env.poachers
                name = env.poachers[o_id].name if is_poach else (env.patrol_boats[o_id].name if o_id in env.patrol_boats else f"Vessel {o_id}")
                ui_messages.append(f"BOUNDARY: {name} sailed out of bounds!")
                # Dynamically match the exact ship type that exited
                spawn_random_vessel(env, boat_views, is_poacher=is_poach)

            if len(ui_messages) > 20:
                ui_messages = ui_messages[-20:]


        # =========================================================================
        # VISIBILITY CALCULATIONS
        # =========================================================================
        # W / cam.scale calculates the current horizontal kilometer width of the viewport window
        view_width_km = W / cam.scale
        
        ports_text_on = show_labels and (view_width_km <= PORT_LABEL_THRESHOLD_KM)
        ships_text_on = show_labels and (view_width_km <= SHIP_LABEL_THRESHOLD_KM)
        cargo_text_on = show_labels and (view_width_km <= CARGO_LABEL_THRESHOLD_KM)
        velocity_text_on = show_labels and (view_width_km <= VELOCITY_LABEL_THRESHOLD_KM)

        map_view.draw_background(screen, cam)
        map_view.draw_ports(screen, cam, ports_text_on, port_labels)
        
        for view in boat_views:
            # Updated to pass the new boolean tracking triggers
            view.draw(screen, cam, ships_text_on, cargo_text_on, velocity_text_on, boat_labels)

        # =========================================================================

        draw_legend(screen, font, legend_fish, legend_patrol)
        draw_scale_bar(screen, cam, font)


        # =========================================================================
        # RENDER MEASURING RULER OVERLAY
        # =========================================================================
        if measuring:
            current_mouse_pos = pygame.mouse.get_pos()
            
            # Translate vector pixel points to continuous kilometer world tracking positions
            x0, y0 = cam.to_world(*measure_start_screen)
            x1, y1 = cam.to_world(*current_mouse_pos)
            
            # Compute Euclidean range distance (km) and compass bearing angle (degrees)
            dist_km = math.hypot(x1 - x0, y1 - y0)
            
            # Compass degrees clockwise from North (+Y axis)
            bearing_rad = math.atan2(x1 - x0, y1 - y0)
            bearing_deg = (math.degrees(bearing_rad)) % 360.0
            
            # Draw vector guide line nodes
            pygame.draw.line(screen, (0, 255, 255), measure_start_screen, current_mouse_pos, 2)
            
            # FIXED: Explicitly unpack the tuples into discrete integer x, y arguments
            pygame.gfxdraw.filled_circle(screen, int(measure_start_screen[0]), int(measure_start_screen[1]), 4, (0, 255, 255))
            pygame.gfxdraw.filled_circle(screen, int(current_mouse_pos[0]), int(current_mouse_pos[1]), 4, (0, 255, 255))
            
            # Blit textual telemetry values next to the dragging crosshair anchor point
            measure_str = f" {dist_km:.1f} km | {bearing_deg:.1f}°"
            m_surf = font.render(measure_str, True, (0, 255, 255))
            
            m_bg = pygame.Surface((m_surf.get_width() + 6, m_surf.get_height() + 2), pygame.SRCALPHA)
            m_bg.fill((10, 15, 20, 200))
            m_bg.blit(m_surf, (3, 1))
            screen.blit(m_bg, (current_mouse_pos[0] + 12, current_mouse_pos[1] - 12))


        # --- Draw Scrolling Messages List ---
        msg_y = 50
        for msg in ui_messages:
            txt_surf = font.render(msg, True, (255, 255, 255))
            bg_box = pygame.Surface((380, txt_surf.get_height() + 4), pygame.SRCALPHA)
            bg_box.fill((10, 10, 15, 180))
            bg_box.blit(txt_surf, (5, 2))
            screen.blit(bg_box, (screen.get_width() - 400, msg_y))
            msg_y += txt_surf.get_height() + 8

        # =========================================================================
        # 5. DRAW BOTTOM HUD & KEYS INSTRUCTIONS PANELS (UPDATED STACKED VERSION)
        # =========================================================================
        mx, my = pygame.mouse.get_pos()
        wx, wy = cam.to_world(mx, my)
        
        total_hours = env.sim_time_hours
        sim_days = int(total_hours // 24)
        sim_hours = int(total_hours % 24)
        sim_minutes = int((total_hours * 60) % 60)
        
        status_prefix = "PAUSED" if is_paused else "RUNNING"
        info = f"[{status_prefix}] TIME: {sim_days:02d}d::{sim_hours:02d}h::{sim_minutes:02d}m"
        info += f"  |  x {wx:7.1f} km  y {wy:7.1f} km"
        info += f"  lon {COASTAL_SUBSET['lon_min'] + wx / (KM_PER_DEGREE * lon_scale):8.3f}"
        info += f"  lat {COASTAL_SUBSET['lat_min'] + wy / KM_PER_DEGREE:7.3f}"
        
        if map_view.map_data.is_in_bounds(wx, wy):
            elev = map_view.map_data.get_height_at_pixel(wx, wy, method="bilinear")
            biomass = map_view.map_data.get_biomass_at_pixel(wx, wy, method="bilinear")
            info += f"  elev {elev:7.0f} m  biomass {biomass:.2f}"
        else:
            info += "  elev ------- m  biomass ----"
            
        hint = "drag: pan | wheel: zoom | H: overlay | L: labels | Space: pause | R-click: measure | R: reset"
        
        # 1. Paint the solid bottom dashboard layer bar
        bar = pygame.Surface((W, 24), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        screen.blit(bar, (0, H - 24))
        
        # 2. Blit Telemetry values left-aligned inside the bottom background bar channel
        screen.blit(font.render(info, True, (230, 230, 230)), (8, H - 21))
        
        # 3. Pre-render instructions text on a separate panel layer stacked 
        # SAFELY ABOVE the dark taskbar strip to prevent screen overlapping completely
        h_surf = font.render(hint, True, (160, 170, 180))
        hint_box = pygame.Surface((h_surf.get_width() + 10, h_surf.get_height() + 4), pygame.SRCALPHA)
        hint_box.fill((10, 12, 18, 190)) 
        hint_box.blit(h_surf, (5, 2))
        
        # Blitted directly above the 24px bottom bar line height margin
        screen.blit(hint_box, (W - hint_box.get_width() - 8, H - 24 - hint_box.get_height() - 4))

        pygame.display.flip()
        frame += 1
        if max_frames is not None and frame >= max_frames:
            break


def main():
    pygame.init()
    pygame.display.set_mode((1280, 900), pygame.RESIZABLE)
    pygame.display.set_caption("MARL Fishing Environment")

    print("Loading data via package MapGenerator...")
    generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
    map_data = generator.generate_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)
    
    map_view = mf.rendering.MapView(map_data)
    env = mf.simulation.Environment(map_data)

    sim_poachers = [
        mf.simulation.PoacherVessel(name="ORION", start_x=45.0, start_y=120.0, start_heading=np.pi / 4),
        mf.simulation.PoacherVessel(name="MARINER", start_x=90.0, start_y=340.0, start_heading=np.pi),
        mf.simulation.PoacherVessel(name="SEAWOLF", start_x=150.0, start_y=510.0, start_heading=1.5 * np.pi)
    ]
    sim_patrols = [
        mf.simulation.PatrolVessel(name="PATROL_A", start_x=60.0, start_y=150.0, start_heading=0.0),
        mf.simulation.PatrolVessel(name="PATROL_B", start_x=110.0, start_y=400.0, start_heading=0.75 * np.pi)
    ]

    for p in sim_poachers:
        env.add_vessel(p)
    for p in sim_patrols:
        env.add_vessel(p)

    boat_views = []
    for p in sim_poachers:
        boat_views.append(mf.rendering.BoatView(p, FISHING_COLOR, boat_size_px=12.0))
    for p in sim_patrols:
        boat_views.append(mf.rendering.BoatView(p, PATROL_COLOR, boat_size_px=12.0))

    run(env, boat_views, map_view)
    pygame.quit()


if __name__ == "__main__":
    main()
