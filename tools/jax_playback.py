"""
MARL fishing map playback viewer (pygame)

Controls
    left/middle drag ... pan
    mouse wheel ........ zoom (about the cursor)
    arrows / WASD ...... pan        + / - ... zoom
    H .................. toggle biomass overlay
    L .................. toggle labels
    R .................. reset view
    Space .............. pause/resume playback
    Left / Right Arrow . scrub frames manually when paused
    Right-click drag ... measure distance and bearing
    Esc ................ quit
"""
import math
import numpy as np
import pygame
import pygame.gfxdraw

import multiagent_fishing as mf
from multiagent_fishing import Layers
from pathlib import Path

from multiagent_fishing.jax_sim.batch_rollout import BatchRollout



# =========================================================================
# CONFIG
# =========================================================================
FILE_PATH = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
PORTS_PATH = "datasets/ports.json"

# Automatically resolve the path to the jax_runs directory relative to this viewer file
CURRENT_DIR = Path(__file__).resolve().parent
ROLLOUT_FILE = str(CURRENT_DIR / "../datasets/jax_runs/generation_001.npz")


RESOLUTION = 1.0  # km per grid cell
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42
RESOLUTION = 1.0  # km per grid cell
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42

MAX_SCALE = 60.0  # max zoom, in screen pixels per km

BG_COLOR = (8, 10, 16)
FISHING_COLOR = (255, 204, 0)
PATROL_COLOR = (0, 255, 102)
PORT_COLOR = (255, 0, 127)

SHIP_LABEL_THRESHOLD_KM = 4000.0
PORT_LABEL_THRESHOLD_KM = 700.0   
CARGO_LABEL_THRESHOLD_KM = 350.0  
VELOCITY_LABEL_THRESHOLD_KM = 150.0 

class Camera:
    def __init__(self, world_w, world_h):
        self.world_w, self.world_h = world_w, world_h
        self.cx = world_w / 2
        self.cy = world_h / 2
        self.scale = 1.0  
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
            legend_fish.draw(screen, ui_cam, labels_on=False, cargo_on=False, velocity_on=False, label_cache=None)
        elif kind == "patrol":
            legend_patrol.vessel.x, legend_patrol.vessel.y = 28, cy
            legend_patrol.draw(screen, ui_cam, labels_on=False, cargo_on=False, velocity_on=False, label_cache=None)
        else:
            pygame.gfxdraw.filled_circle(screen, 28, cy, 5, PORT_COLOR)
            pygame.gfxdraw.aacircle(screen, 28, cy, 5, (255, 255, 255))
            
        screen.blit(font.render(name, True, (230, 230, 230)), (48, cy - 8))


# =========================================================================
# PLAYBACK MAIN LOOP
# =========================================================================
def run(env, poacher_views, patrol_views, map_view, playback_data):
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

    legend_fish_vessel = mf.simulation.PoacherVessel(name="DummyFish", start_x=0.0, start_y=0.0, home_port_id=0, start_heading=0.0)
    legend_patrol_vessel = mf.simulation.PatrolVessel(name="DummyPatrol", start_x=0.0, start_y=0.0, start_heading=0.0)

    legend_fish = mf.rendering.BoatView(legend_fish_vessel, FISHING_COLOR, boat_size_px=12.0)
    legend_patrol = mf.rendering.BoatView(legend_patrol_vessel, PATROL_COLOR, boat_size_px=12.0)

    ui_messages = []
    show_labels = True
    dragging = False
    is_paused = False  
    
    measuring = False
    measure_start_screen = (0, 0)

    clock = pygame.time.Clock()
    
    # ---------------------------------------------------------------------
    # PLAYBACK TIMELINE CONTROL SETUP
    # ---------------------------------------------------------------------
    current_frame = 0
    # Pull dictionary keys containing position data (Modify keys to exactly match your JAX tracking keys)
    # Assumes shapes: (horizon, num_agents) or (horizon, num_agents, coords)
    poacher_x = playback_data["poacher_x"]  
    poacher_y = playback_data["poacher_y"]
    poacher_heading = playback_data.get("poacher_heading", np.zeros_like(poacher_x))
    
    patrol_x = playback_data["patrol_x"]
    patrol_y = playback_data["patrol_y"]
    patrol_heading = playback_data.get("patrol_heading", np.zeros_like(patrol_x))
    
    horizon = poacher_x.shape[0]
    ui_messages.append(f"Loaded playback run with {horizon} frames.")

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
                    map_view.toggle_overlay("biomass")
                elif ev.key == pygame.K_j:
                    map_view.toggle_overlay("shore")
                elif ev.key == pygame.K_k:
                    map_view.toggle_overlay("port_dist")
                elif ev.key == pygame.K_f:
                    map_view.show_flow = not map_view.show_flow
                elif ev.key == pygame.K_l:
                    show_labels = not show_labels
                elif ev.key == pygame.K_r:
                    cam.fit(W, H)
                elif ev.key == pygame.K_SPACE:
                    is_paused = not is_paused  
                    ui_messages.append("PLAYBACK PAUSED" if is_paused else "PLAYBACK RESUMED")
                # Manual frame scrubbing tools when simulation is paused
                elif ev.key == pygame.K_LEFT and is_paused:
                    current_frame = max(0, current_frame - 1)
                elif ev.key == pygame.K_RIGHT and is_paused:
                    current_frame = min(horizon - 1, current_frame + 1)
                    
            elif ev.type == pygame.MOUSEWHEEL:
                cam.zoom_at(*pygame.mouse.get_pos(), 1.2 ** ev.y)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button in (1, 2):
                    dragging = True
                elif ev.button == 3:  
                    measuring = True
                    measure_start_screen = ev.pos
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button in (1, 2):
                    dragging = False
                elif ev.button == 3:  
                    measuring = False
            elif ev.type == pygame.MOUSEMOTION:
                if dragging:
                    cam.pan_pixels(*ev.rel)

        # Smooth Map Panning System
        keys = pygame.key.get_pressed()
        step_speed = 600 * dt
        dx = (keys[pygame.K_LEFT] or keys[pygame.K_a]) - (keys[pygame.K_RIGHT] or keys[pygame.K_d])
        dy = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
        
        # Avoid panning with Arrow keys if we are using them to scrub frames in Paused Mode
        if not is_paused:
            if dx or dy:
                cam.pan_pixels(dx * step_speed, dy * step_speed)
        else:
            # While paused, allow continuous map pan using the WASD configuration
            letter_dx = keys[pygame.K_a] - keys[pygame.K_d]
            letter_dy = keys[pygame.K_w] - keys[pygame.K_s]
            if letter_dx or letter_dy:
                cam.pan_pixels(letter_dx * step_speed, letter_dy * step_speed)

        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            cam.zoom_at(W / 2, H / 2, 1 + 1.5 * dt)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            cam.zoom_at(W / 2, H / 2, 1 / (1 + 1.5 * dt))

        # =========================================================================
        # UPDATE PLAYBACK TIMELINE INDEX
        # =========================================================================
        if not is_paused:
            current_frame += 1
            if current_frame >= horizon:
                current_frame = 0  # Loop timeline seamlessly back to step zero

        # Inject logged positional matrix states into underlying view entities
        for idx, view in enumerate(poacher_views):
            if idx < poacher_x.shape[1]:
                view.vessel.x = poacher_x[current_frame, idx]
                view.vessel.y = poacher_y[current_frame, idx]
                view.vessel.heading = poacher_heading[current_frame, idx]

        for idx, view in enumerate(patrol_views):
            if idx < patrol_x.shape[1]:
                view.vessel.x = patrol_x[current_frame, idx]
                view.vessel.y = patrol_y[current_frame, idx]
                view.vessel.heading = patrol_heading[current_frame, idx]

        # =========================================================================
        # VISIBILITY CALCULATIONS & MAP RENDERING
        # =========================================================================
        view_width_km = W / cam.scale
        ports_text_on = show_labels and (view_width_km <= PORT_LABEL_THRESHOLD_KM)
        ships_text_on = show_labels and (view_width_km <= SHIP_LABEL_THRESHOLD_KM)
        cargo_text_on = show_labels and (view_width_km <= CARGO_LABEL_THRESHOLD_KM)
        velocity_text_on = show_labels and (view_width_km <= VELOCITY_LABEL_THRESHOLD_KM)

        map_view.draw_background(screen, cam)
        map_view.draw_flow(screen, cam)
        map_view.draw_ports(screen, cam, ports_text_on, port_labels)
                
        for view in poacher_views + patrol_views:
            view.draw(screen, cam, ships_text_on, cargo_text_on, velocity_text_on, boat_labels)

        draw_legend(screen, font, legend_fish, legend_patrol)
        draw_scale_bar(screen, cam, font)

        # MEASURING TOOL RADIAL OVERLAY
        if measuring:
            current_mouse_pos = pygame.mouse.get_pos()
            x0, y0 = cam.to_world(*measure_start_screen)
            x1, y1 = cam.to_world(*current_mouse_pos)
            dist_km = math.hypot(x1 - x0, y1 - y0)
            bearing_rad = math.atan2(x1 - x0, y1 - y0)
            bearing_deg = (math.degrees(bearing_rad)) % 360.0
            
            pygame.draw.line(screen, (0, 255, 255), measure_start_screen, current_mouse_pos, 2)
            pygame.gfxdraw.filled_circle(screen, int(measure_start_screen[0]), int(measure_start_screen[1]), 4, (0, 255, 255))
            pygame.gfxdraw.filled_circle(screen, int(current_mouse_pos[0]), int(current_mouse_pos[1]), 4, (0, 255, 255))
            
            measure_str = f" {dist_km:.1f} km | {bearing_deg:.1f}°"
            m_surf = font.render(measure_str, True, (0, 255, 255))
            m_bg = pygame.Surface((m_surf.get_width() + 6, m_surf.get_height() + 2), pygame.SRCALPHA)
            m_bg.fill((10, 15, 20, 200))
            m_bg.blit(m_surf, (3, 1))
            screen.blit(m_bg, (current_mouse_pos[0] + 12, current_mouse_pos[1] - 12))

        # Scrolling Messages Box Channel
        msg_y = 50
        if len(ui_messages) > 20: 
            ui_messages = ui_messages[-20:]
        for msg in ui_messages:
            txt_surf = font.render(msg, True, (255, 255, 255))
            bg_box = pygame.Surface((380, txt_surf.get_height() + 4), pygame.SRCALPHA)
            bg_box.fill((10, 10, 15, 180))
            bg_box.blit(txt_surf, (5, 2))
            screen.blit(bg_box, (screen.get_width() - 400, msg_y))
            msg_y += txt_surf.get_height() + 8

        # =========================================================================
        # HUD DASHBOARD FOOTER
        # =========================================================================
        mx, my = pygame.mouse.get_pos()
        wx, wy = cam.to_world(mx, my)
        
        status_prefix = "PLAYBACK_PAUSED" if is_paused else "PLAYBACK_PLAYING"
        info = f"[{status_prefix}] FRAME: {current_frame:04d}/{horizon:04d}"
        info += f"  |  x {wx:7.1f} km  y {wy:7.1f} km"
        lon, lat = map_view.map_data.km_to_lonlat(wx, wy)
        info += f"  lon {lon:8.3f}  lat {lat:7.3f}"

        if map_view.map_data.is_in_bounds(wx, wy):
            cell = map_view.map_data.sample_point((wx, wy))
            info += f"  elev {cell[Layers.HEIGHT]:7.0f} m  biomass {cell[Layers.BIOMASS]:.2f}"
        else:
            info += "  elev ------- m  biomass ----"
            
        hint_lines = [
            "drag: pan | wheel: zoom | R: reset view | Space: pause | R-click: measure",
            "H: biomass | J: shore dist | K: port dist | F: port flow | L: labels",
            "PAUSED MODE -> Left/Right Arrow: Step frames manually | WASD: Pan map"
        ]
        
        bar = pygame.Surface((W, 24), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        screen.blit(bar, (0, H - 24))
        screen.blit(font.render(info, True, (230, 230, 230)), (8, H - 21))
        
        h_surfs = [font.render(line, True, (160, 170, 180)) for line in hint_lines]
        line_h = h_surfs[0].get_height()
        box_w = max(s.get_width() for s in h_surfs) + 10
        box_h = line_h * len(h_surfs) + 4 + 2 * (len(h_surfs) - 1)

        hint_box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        hint_box.fill((10, 12, 18, 190))
        
        for i, s in enumerate(h_surfs):
            hint_box.blit(s, (5, 2 + i * (line_h + 2)))  

        screen.blit(hint_box, (8, H - 24 - box_h - 4))

        pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_mode((900, 900), pygame.RESIZABLE)
    pygame.display.set_caption("MARL Fishing Playback Viewer")

    # Load rollout file via your BatchRollout definition
    rollout = BatchRollout.load(ROLLOUT_FILE)
    best_idx, best_score, single_run_data = rollout.get_representative_run()
    print(f"Playing back Environment Index {best_idx} (Cumulative Score: {best_score:.2f})")

    print("Loading data via package MapGenerator...")
    generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
    map_data = generator.generate_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)
    
    map_view = mf.rendering.MapView(map_data)
    env = mf.simulation.Environment(map_data)

    # Automatically size agent layout counts based on trajectory shape sizes
    num_poachers = single_run_data["poacher_x"].shape[1]
    num_patrols = single_run_data["patrol_x"].shape[1]

    poacher_views = []
    for i in range(num_poachers):
        p = mf.simulation.PoacherVessel(name=f"POACHER_{i}", start_x=0, start_y=0, home_port_id=0, start_heading=0)
        poacher_views.append(mf.rendering.BoatView(p, FISHING_COLOR, boat_size_px=12.0))

    patrol_views = []
    for i in range(num_patrols):
        p = mf.simulation.PatrolVessel(name=f"PATROL_{i}", start_x=0, start_y=0, start_heading=0)
        patrol_views.append(mf.rendering.BoatView(p, PATROL_COLOR, boat_size_px=12.0))

    run(env, poacher_views, patrol_views, map_view, single_run_data)
    pygame.quit()


if __name__ == "__main__":
    main()
