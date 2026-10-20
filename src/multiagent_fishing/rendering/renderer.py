"""
src/multiagent_fishing/rendering/renderer.py
"""
import math
import pygame
import pygame.gfxdraw

# Relative imports within the multiagent_fishing package structure
from ..world import Layers 
# Adjust these relative paths if your actual agent/vessel file structure differs
from ..simulation.vessels import PoacherVessel
from ..simulation.vessels import PatrolVessel

# Visual Thresholds and Styling
SHIP_LABEL_THRESHOLD_KM = 4000.0
PORT_LABEL_THRESHOLD_KM = 700.0   
CARGO_LABEL_THRESHOLD_KM = 350.0  
VELOCITY_LABEL_THRESHOLD_KM = 150.0
MAX_SCALE = 60.0

PORT_COLOR = (255, 0, 127)


class Camera:
    """Manages world-to-screen coordinate translations, zooming, and panning."""
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
    """Caches text surface textures to save CPU cycles on font rendering."""
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


class MapViewerRenderer:
    """Composite rendering engine that orchestrates drawing maps, vessels, and overlays."""
    def __init__(self, camera, fishing_color, patrol_color):
        self.cam = camera
        self.fishing_color = fishing_color
        self.patrol_color = patrol_color
        
        # Initialize Fonts
        mono = "consolas,menlo,dejavusansmono,monospace"
        self.font = pygame.font.SysFont(mono, 14)
        self.label_font = pygame.font.SysFont(mono, 12, bold=True)
        
        self.port_labels = LabelCache(self.label_font, 90)
        self.boat_labels = LabelCache(self.label_font, 130)
        
        # Deferred initialization for the legend views (setup when drawing context is ready)
        self._legend_initialized = False
        self.legend_fish = None
        self.legend_patrol = None

    def _init_legend_views(self, boat_view_cls):
        """Pre-construct dummy vessel assets to illustrate the UI Legend using imported classes."""
        f_vessel = PoacherVessel(name="DummyFish", start_x=0.0, start_y=0.0, home_port_id=0, start_heading=0.0)
        p_vessel = PatrolVessel(name="DummyPatrol", start_x=0.0, start_y=0.0, start_heading=0.0)
        self.legend_fish = boat_view_cls(f_vessel, self.fishing_color, boat_size_px=12.0)
        self.legend_patrol = boat_view_cls(p_vessel, self.patrol_color, boat_size_px=12.0)
        self._legend_initialized = True

    def render(self, screen, map_view, boat_views, sim_time_hours, ui_messages, show_labels, is_paused, measuring, measure_start_screen):
        W, H = screen.get_size()
        view_width_km = W / self.cam.scale
        
        # 1. Initialize Legend items statically if view class is available
        if not self._legend_initialized and boat_views:
            boat_view_cls = boat_views[0].__class__
            self._init_legend_views(boat_view_cls)

        # 2. Level of Detail Threshold Visibility Triggers
        ports_text_on = show_labels and (view_width_km <= PORT_LABEL_THRESHOLD_KM)
        ships_text_on = show_labels and (view_width_km <= SHIP_LABEL_THRESHOLD_KM)
        cargo_text_on = show_labels and (view_width_km <= CARGO_LABEL_THRESHOLD_KM)
        velocity_text_on = show_labels and (view_width_km <= VELOCITY_LABEL_THRESHOLD_KM)

        # 3. Base Background Map Passes
        map_view.draw_background(screen, self.cam)
        map_view.draw_flow(screen, self.cam)
        map_view.draw_ports(screen, self.cam, ports_text_on, self.port_labels)
                
        # 4. Simulation Entities Pass
        for view in boat_views:
            view.draw(screen, self.cam, ships_text_on, cargo_text_on, velocity_text_on, self.boat_labels)

        # 5. UI Panels, Overlays & HUD Elements
        if self._legend_initialized:
            self._draw_legend(screen)
            
        self._draw_scale_bar(screen)
        
        if measuring:
            self._draw_measurement_ruler(screen, measure_start_screen)
            
        self._draw_ui_logs(screen, ui_messages)
        self._draw_bottom_dashboard(screen=screen, sim_time_hours=sim_time_hours, map_view=map_view, is_paused=is_paused)

    def _draw_scale_bar(self, screen):
        km = 140 / self.cam.scale
        mag = 10 ** math.floor(math.log10(km))
        nice = max(m * mag for m in (1, 2, 5) if m * mag <= km)
        length = nice * self.cam.scale
        x1, y = self.cam.W - 20, self.cam.H - 40
        x0 = x1 - length
        pygame.draw.line(screen, (255, 255, 255), (x0, y), (x1, y), 2)
        pygame.draw.line(screen, (255, 255, 255), (x0, y - 5), (x0, y + 5), 2)
        pygame.draw.line(screen, (255, 255, 255), (x1, y - 5), (x1, y + 5), 2)
        lbl = self.font.render(f"{nice:g} km", True, (255, 255, 255))
        screen.blit(lbl, (x1 - length / 2 - lbl.get_width() / 2, y - 22))

    def _draw_legend(self, screen):
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
            if kind == "fish" and self.legend_fish:
                self.legend_fish.vessel.x, self.legend_fish.vessel.y = 28, cy
                self.legend_fish.draw(screen, ui_cam, labels_on=False, cargo_on=False, velocity_on=False, label_cache=None)
            elif kind == "patrol" and self.legend_patrol:
                self.legend_patrol.vessel.x, self.legend_patrol.vessel.y = 28, cy
                self.legend_patrol.draw(screen, ui_cam, labels_on=False, cargo_on=False, velocity_on=False, label_cache=None)
            else:
                pygame.gfxdraw.filled_circle(screen, 28, cy, 5, PORT_COLOR)
                pygame.gfxdraw.aacircle(screen, 28, cy, 5, (255, 255, 255))
                
            screen.blit(self.font.render(name, True, (230, 230, 230)), (48, cy - 8))

    def _draw_measurement_ruler(self, screen, start_screen):
        current_mouse_pos = pygame.mouse.get_pos()
        x0, y0 = self.cam.to_world(*start_screen)
        x1, y1 = self.cam.to_world(*current_mouse_pos)
        
        dist_km = math.hypot(x1 - x0, y1 - y0)
        bearing_rad = math.atan2(x1 - x0, y1 - y0)
        bearing_deg = (math.degrees(bearing_rad)) % 360.0
        
        pygame.draw.line(screen, (0, 255, 255), start_screen, current_mouse_pos, 2)
        pygame.gfxdraw.filled_circle(screen, int(start_screen[0]), int(start_screen[1]), 4, (0, 255, 255))
        pygame.gfxdraw.filled_circle(screen, int(current_mouse_pos[0]), int(current_mouse_pos[1]), 4, (0, 255, 255))
        
        measure_str = f" {dist_km:.1f} km | {bearing_deg:.1f}°"
        m_surf = self.font.render(measure_str, True, (0, 255, 255))
        
        m_bg = pygame.Surface((m_surf.get_width() + 6, m_surf.get_height() + 2), pygame.SRCALPHA)
        m_bg.fill((10, 15, 20, 200))
        m_bg.blit(m_surf, (3, 1))
        screen.blit(m_bg, (current_mouse_pos[0] + 12, current_mouse_pos[1] - 12))

    def _draw_ui_logs(self, screen, ui_messages):
        msg_y = 50
        for msg in ui_messages:
            txt_surf = self.font.render(msg, True, (255, 255, 255))
            bg_box = pygame.Surface((380, txt_surf.get_height() + 4), pygame.SRCALPHA)
            bg_box.fill((10, 10, 15, 180))
            bg_box.blit(txt_surf, (5, 2))
            screen.blit(bg_box, (screen.get_width() - 400, msg_y))
            msg_y += txt_surf.get_height() + 8

    def _draw_bottom_dashboard(self, screen, sim_time_hours, map_view, is_paused):
        W, H = screen.get_size()
        mx, my = pygame.mouse.get_pos()
        wx, wy = self.cam.to_world(mx, my)
        
        total_hours = sim_time_hours
        sim_days = int(total_hours // 24)
        sim_hours = int(total_hours % 24)
        sim_minutes = int((total_hours * 60) % 60)
        
        status_prefix = "PAUSED" if is_paused else "RUNNING"
        info = f"[{status_prefix}] TIME: {sim_days:02d}d::{sim_hours:02d}h::{sim_minutes:02d}m"
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
        ]
        
        bar = pygame.Surface((W, 24), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        screen.blit(bar, (0, H - 24))
        screen.blit(self.font.render(info, True, (230, 230, 230)), (8, H - 21))
        
        h_surfs = [self.font.render(line, True, (160, 170, 180)) for line in hint_lines]
        line_h = h_surfs[0].get_height()
        box_w = max(s.get_width() for s in h_surfs) + 10
        box_h = line_h * len(h_surfs) + 4 + 2 * (len(h_surfs) - 1)

        hint_box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        hint_box.fill((10, 12, 18, 190))
        for i, s in enumerate(h_surfs):
            hint_box.blit(s, (5, 2 + i * (line_h + 2)))

        screen.blit(hint_box, (8, H - 24 - box_h - 4))
