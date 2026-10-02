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
RESOLUTION = 1.0  # km per grid cell
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42

KM_PER_DEGREE = 111.32
LABEL_THRESHOLD_KM = 400.0  # hide labels when more than this many km are visible
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

    # Clean dummy layout targets using explicit class setups
    dummy_fish_vessel = mf.simulation.PoacherVessel(vessel_id=-1, name="DummyFish", start_x=0.0, start_y=0.0, start_heading=0.0)
    dummy_patrol_vessel = mf.simulation.PatrolVessel(vessel_id=-2, name="DummyPatrol", start_x=0.0, start_y=0.0, start_heading=0.0)

    legend_fish = mf.rendering.BoatView(dummy_fish_vessel, FISHING_COLOR, boat_size_px=12.0)
    legend_patrol = mf.rendering.BoatView(dummy_patrol_vessel, PATROL_COLOR, boat_size_px=12.0)

    center_lat = (COASTAL_SUBSET["lat_min"] + COASTAL_SUBSET["lat_max"]) / 2
    lon_scale = np.cos(np.radians(center_lat))

    show_labels = True
    dragging = False
    clock = pygame.time.Clock()
    frame = 0
    running = True

    # A simple list of strings to store active alerts
    ui_messages = []

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
        # UPDATE SIMULATION STEP
        # =========================================================================
        # Build standard actions mappings to feed headings dynamically into vessels
        # Keeps boats moving forward along their initial heading setups at half-throttle (0.5)
        # =========================================================================
        # UPDATE SIMULATION STEP
        # =========================================================================
        actions = {v.vessel.id: (v.vessel.current_heading, 0.5) for v in boat_views}
        
        # 1. Capture the logs dictionary from the step
        step_events = env.step(actions, dt=dt)
        
        # 2. Extract events and append plain text strings directly to our list
        for app in step_events.get("apprehensions", []):
            p_id, pat_id = app["poacher_id"], app["by_patrol_id"]
            poacher_name = env.poachers[p_id].name if p_id in env.poachers else f"Poacher {p_id}"
            patrol_name = env.patrol_boats[pat_id].name if pat_id in env.patrol_boats else f"Patrol {pat_id}"
            ui_messages.append(f"💥 ALERT: {poacher_name} captured by {patrol_name}!")

        for g_id in step_events.get("groundings", []):
            name = env.poachers[g_id].name if g_id in env.poachers else (env.patrol_boats[g_id].name if g_id in env.patrol_boats else f"Vessel {g_id}")
            ui_messages.append(f"🏝️ CRASH: {name} ran aground on land!")

        for o_id in step_events.get("out_of_bounds", []):
            name = env.poachers[o_id].name if o_id in env.poachers else (env.patrol_boats[o_id].name if o_id in env.patrol_boats else f"Vessel {o_id}")
            ui_messages.append(f"🌊 BOUNDARY: {name} sailed out of bounds!")

        # Keep only the last 5 messages so they don't fill up the screen
        if len(ui_messages) > 5:
            ui_messages = ui_messages[-5:]

        # Visibility pass
        labels_on = show_labels and (W / cam.scale <= LABEL_THRESHOLD_KM)

        # Drawing steps
        map_view.draw_background(screen, cam)
        map_view.draw_ports(screen, cam, labels_on, port_labels)

        for view in boat_views:
            view.draw(screen, cam, labels_on, boat_labels)

        draw_legend(screen, font, legend_fish, legend_patrol)
        draw_scale_bar(screen, cam, font)


        # --- Draw Scrolling Messages List ---
        msg_y = 50  # Start position on the right side of the screen
        for msg in ui_messages:
            txt_surf = font.render(msg, True, (255, 255, 255))
            
            # Simple translucent background box for readability
            bg_box = pygame.Surface((320, txt_surf.get_height() + 4), pygame.SRCALPHA)
            bg_box.fill((10, 10, 15, 180))
            bg_box.blit(txt_surf, (5, 2))
            
            screen.blit(bg_box, (screen.get_width() - 340, msg_y))
            msg_y += txt_surf.get_height() + 8


        # HUD Text Output Pass
        mx, my = pygame.mouse.get_pos()
        wx, wy = cam.to_world(mx, my)
        info = f"x {wx:7.1f} km  y {wy:7.1f} km"
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
