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
RESOLUTION = 0.5  # km per grid cell
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42

KM_PER_DEGREE = 111.32
SHIP_LABEL_THRESHOLD_KM = 9000.0  # hide labels when more than this many km are visible
PORT_LABEL_THRESHOLD_KM = 700.0
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
            legend_fish.draw(screen, ui_cam, labels_on=False, label_cache=None)
        elif kind == "patrol":
            legend_patrol.vessel.x, legend_patrol.vessel.y = 28, cy
            legend_patrol.draw(screen, ui_cam, labels_on=False, label_cache=None)
        else:
            pygame.gfxdraw.filled_circle(screen, 28, cy, 5, PORT_COLOR)
            pygame.gfxdraw.aacircle(screen, 28, cy, 5, (255, 255, 255))
            
        screen.blit(font.render(name, True, (230, 230, 230)), (48, cy - 8))


# Inside tools/pygame_load_map.py (or your main frontend controller loop)
import random
import json
from pathlib import Path

# Cache the name arrays globally in your front-end script memory to avoid slow, repeated disk reads
_NAME_POOL = None

def spawn_random_poacher_frontend(env, boat_views_list, FISHING_COLOR, names_json_path="datasets/boat_names.json"):
    """Handles frontend orchestration: generates name, registers sim model, creates visual view."""
    global _NAME_POOL
    
    if not env.map_data.ports:
        return None

    # Load JSON only once into frontend global memory cache
    if _NAME_POOL is None:
        p = Path(names_json_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _NAME_POOL = json.load(f)
        else:
            _NAME_POOL = {"fishing_boats": {"first_word": ["MYSTIC"], "last_word": ["WAVE"]}}

    # 1. Selection logic: select random port node from the clean map data list
    port = random.choice(env.map_data.ports)
    
    # 2. Form string name configuration paths
    pool = _NAME_POOL.get("fishing_boats", {})
    first = random.choice(pool.get("first_word", ["MYSTIC"]))
    last = random.choice(pool.get("last_word", ["WAVE"]))
    vessel_name = f"{first} {last}"
    
    # 3. Dynamic increment ID assignments pass
    # Environment tracks all IDs globally to avoid conflicts
    next_id = 1
    while env._is_id_conflicting(next_id):
        next_id += 1

    # 4. Instantiate pure data model object and attach to mathematical physics frame tracking
    new_vessel = mf.simulation.PoacherVessel(
        vessel_id=next_id, 
        name=vessel_name, 
        start_x=port["grid_x"], 
        start_y=port["grid_y"],
        start_heading=random.uniform(0.0, 2 * math.pi)
    )
    
    # Register with the environment data engine
    env.add_poacher(new_vessel)

    # 5. Instantly encapsulate it in a rendering layer wrapper object
    new_view = mf.rendering.BoatView(new_vessel, FISHING_COLOR, boat_size_px=12.0)
    boat_views_list.append(new_view)
    
    return next_id


# =========================================================================
# MAIN LOOP
# =========================================================================
def run(env, boat_views, map_view, max_frames=None):
    screen = pygame.display.get_surface()

    # Extract world dimensions directly from the encapsulated map data
    Wc, Hc = map_view.map_data.shape
    world_w = map_view.map_data.width_km
    world_h = map_view.map_data.height_km
    
    cam = Camera(world_w, world_h)
    cam.fit(*screen.get_size())

    mono = "consolas,menlo,dejavusansmono,monospace"
    font = pygame.font.SysFont(mono, 14)
    label_font = pygame.font.SysFont(mono, 12, bold=True)
    port_labels = LabelCache(label_font, 90)
    boat_labels = LabelCache(label_font, 130)

    # Instantiate actual simulation vessel objects for strict type consistency inside the legend
    dummy_fish_vessel = mf.simulation.PoacherVessel(vessel_id=-1, name="DummyFish", start_x=0.0, start_y=0.0, start_heading=0.0)
    dummy_patrol_vessel = mf.simulation.PatrolVessel(vessel_id=-2, name="DummyPatrol", start_x=0.0, start_y=0.0, start_heading=0.0)

    legend_fish = mf.rendering.BoatView(dummy_fish_vessel, FISHING_COLOR, boat_size_px=12.0)
    legend_patrol = mf.rendering.BoatView(dummy_patrol_vessel, PATROL_COLOR, boat_size_px=12.0)

    center_lat = (COASTAL_SUBSET["lat_min"] + COASTAL_SUBSET["lat_max"]) / 2
    lon_scale = np.cos(np.radians(center_lat))

    # Plain Python structures to hold scrolling alerts and layout states
    ui_messages = []
    show_labels = True
    dragging = False
    clock = pygame.time.Clock()
    frame = 0
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        screen = pygame.display.get_surface()
        W, H = screen.get_size()
        cam.W, cam.H = W, H

        # 1. HANDLE USER/KEYBOARD INPUTS
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
            elif ev.type == pygame.MOUSEWHEEL:
                cam.zoom_at(*pygame.mouse.get_pos(), 1.2 ** ev.y)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 2):
                dragging = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button in (1, 2):
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
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
        # 2. RUNTIME LIFECYCLE MANAGEMENT & ACTION ROUTING
        # =========================================================================
        actions = {}
        views_to_remove = []

        for view in boat_views:
            vessel = view.vessel
            
            # Pass A: Handle general environmental inactivations (groundings, captures)
            if not vessel.is_active:
                views_to_remove.append(view)
                env.remove_vessel(vessel.id)
                continue
            
            # Pass B: Evaluate if a fully loaded poacher has arrived back at its home harbor
            if hasattr(vessel, "cargo") and vessel.cargo >= vessel.max_cargo:
                dx_home = vessel.home_x - vessel.x
                dy_home = vessel.home_y - vessel.y
                distance_home = math.hypot(dx_home, dy_home)
                
                if distance_home <= 1.5:  # Destination reached safely!
                    print(f"🚢 {vessel.name} successfully unloaded cargo at home harbor and despawned!")
                    vessel.is_active = False
                    views_to_remove.append(view)
                    env.remove_vessel(vessel.id)
                    
                    # Deploy a replacement immediately at a random harbor port
                    spawn_random_poacher_frontend(env, boat_views, FISHING_COLOR)
                    continue
                else:
                    # Steer straight towards home harbor coords at full throttle (1.0)
                    target_heading = math.atan2(dx_home, dy_home)
                    actions[vessel.id] = (target_heading, 1.0)
            else:
                # Default Cruise: Maintain initial heading at half throttle (0.5)
                actions[vessel.id] = (vessel.current_heading, 0.5)

        # Clean out dead visual views to prevent index-skipping loop errors
        for dead_view in views_to_remove:
            if dead_view in boat_views:
                boat_views.remove(dead_view)

        # =========================================================================
        # 3. ADVANCE SIMULATION STATE & INTERCEPT LOGGING
        # =========================================================================
        step_events = env.step(actions, dt=dt)
        
        # Parse returned physics events directly into the text alert lists
        for app in step_events.get("apprehensions", []):
            p_id, pat_id = app["poacher_id"], app["by_patrol_id"]
            poacher_name = env.poachers[p_id].name if p_id in env.poachers else f"Poacher {p_id}"
            patrol_name = env.patrol_boats[pat_id].name if pat_id in env.patrol_boats else f"Patrol {pat_id}"
            ui_messages.append(f"💥 ALERT: {poacher_name} captured by {patrol_name}!")

        for g_id in step_events.get("groundings", []):
            name = env.poachers[g_id].name if g_id in env.poachers else (env.patrol_boats[g_id].name if g_id in env.patrol_boats else f"Vessel {g_id}")
            ui_messages.append(f"🏝️ CRASH: {name} ran aground on land!")
            
            # Spawn a replacement boat automatically following a terrain collision
            spawn_random_poacher_frontend(env, boat_views, FISHING_COLOR)

        for o_id in step_events.get("out_of_bounds", []):
            name = env.poachers[o_id].name if o_id in env.poachers else (env.patrol_boats[o_id].name if o_id in env.patrol_boats else f"Vessel {o_id}")
            ui_messages.append(f"🌊 BOUNDARY: {name} sailed out of bounds!")
            
            # Spawn a replacement boat automatically following a boundary breach
            spawn_random_poacher_frontend(env, boat_views, FISHING_COLOR)

        # Enforce history limit: remove oldest item if list size exceeds 20 items
        if len(ui_messages) > 20:
            ui_messages = ui_messages[-20:]

        # =========================================================================
        # 4. RENDERING PASSES
        # =========================================================================
        vessel_labels_on = show_labels and (W / cam.scale <= SHIP_LABEL_THRESHOLD_KM)
        port_labels_on = show_labels and (W / cam.scale <= PORT_LABEL_THRESHOLD_KM)


        # Draw map backgrounds and active port vectors
        map_view.draw_background(screen, cam)
        map_view.draw_ports(screen, cam, port_labels_on, port_labels)

        # Polymorphically render remaining active ships
        for view in boat_views:
            view.draw(screen, cam, vessel_labels_on, boat_labels)

        # Render overlays, legends, and scales
        draw_legend(screen, font, legend_fish, legend_patrol)
        draw_scale_bar(screen, cam, font)

        # --- Draw Scrolling Messages List ---
        msg_y = 50
        for msg in ui_messages:
            txt_surf = font.render(msg, True, (255, 255, 255))
            
            # FIXED: Increased container width from 320 to 380 to fit longer text strings safely
            bg_box = pygame.Surface((380, txt_surf.get_height() + 4), pygame.SRCALPHA)
            bg_box.fill((10, 10, 15, 180))
            bg_box.blit(txt_surf, (5, 2))
            
            # FIXED: Shifted position left to screen.get_width() - 400 to account for the wider boxes
            screen.blit(bg_box, (screen.get_width() - 400, msg_y))
            msg_y += txt_surf.get_height() + 8



        # --- Draw Bottom HUD Bar Metrics ---
        mx, my = pygame.mouse.get_pos()
        wx, wy = cam.to_world(mx, my)
        
        # Calculate simulation clock metrics out of total elapsed hours
        total_hours = env.sim_time_hours
        sim_days = int(total_hours // 24)
        sim_hours = int(total_hours % 24)
        sim_minutes = int((total_hours * 60) % 60)
        
        # Build HUD metadata string with the time signature appended directly to the front
        info = f"TIME: {sim_days:02d}d::{sim_hours:02d}h::{sim_minutes:02d}m"
        info += f"  |  x {wx:7.1f} km  y {wy:7.1f} km"
        info += f"  lon {COASTAL_SUBSET['lon_min'] + wx / (KM_PER_DEGREE * lon_scale):8.3f}"
        info += f"  lat {COASTAL_SUBSET['lat_min'] + wy / KM_PER_DEGREE:7.3f}"
        
        if map_view.map_data.is_in_bounds(wx, wy):
            elev = map_view.map_data.get_height_at_pixel(wx, wy, method="bilinear")
            biomass = map_view.map_data.get_biomass_at_pixel(wx, wy, method="bilinear")
            info += f"  elev {elev:7.0f} m  biomass {biomass:.2f}"
        else:
            info += "  elev ------- m  biomass ----"
            
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
    generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
    map_data = generator.get_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)
    
    # Initialize both the View Presenter and the Data Environment Tracker
    map_view = mf.rendering.MapView(map_data)
    env = mf.simulation.Environment(map_data)

    # 1. Instantiate pure simulation fleet items (With explicit initial headings)
    sim_poachers = [
        mf.simulation.PoacherVessel(vessel_id=1, name="ORION", start_x=45.0, start_y=120.0, start_heading=np.pi / 4),
        mf.simulation.PoacherVessel(vessel_id=2, name="MARINER", start_x=90.0, start_y=340.0, start_heading=np.pi),
        mf.simulation.PoacherVessel(vessel_id=3, name="SEAWOLF", start_x=150.0, start_y=510.0, start_heading=1.5 * np.pi)
    ]
    sim_patrols = [
        mf.simulation.PatrolVessel(vessel_id=10, name="PATROL_A", start_x=60.0, start_y=150.0, start_heading=0.0),
        mf.simulation.PatrolVessel(vessel_id=11, name="PATROL_B", start_x=110.0, start_y=400.0, start_heading=0.75 * np.pi)
    ]

    # 2. Register instances inside the mathematical environment engine
    for p in sim_poachers:
        env.add_poacher(p)
    for p in sim_patrols:
        env.add_patrol(p)

    # 3. Create view wrappers mapping directly over the active environment vessels list
    boat_views = []
    for p in sim_poachers:
        boat_views.append(mf.rendering.BoatView(p, FISHING_COLOR, boat_size_px=12.0))
    for p in sim_patrols:
        boat_views.append(mf.rendering.BoatView(p, PATROL_COLOR, boat_size_px=12.0))

    run(env, boat_views, map_view)
    pygame.quit()


if __name__ == "__main__":
    main()
