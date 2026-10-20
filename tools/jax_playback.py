"""
MARL fishing replay viewer (pygame)
Plays back a pre-simulated game from a GameStore file.
"""
import bisect

import numpy as np
import pygame
import multiagent_fishing as mf

from multiagent_fishing.rendering import MapViewerRenderer, Camera
from multiagent_fishing.jax_sim.game_store import GameStore

# =========================================================================
# CONFIG
# =========================================================================
STORE_PATH = "datasets/jax_runs/generation_001.npz"
GAME_METRIC = "fish"
GAME_HOW = "median"

FILE_PATH = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
PORTS_PATH = "datasets/ports.json"
RESOLUTION = 1.0
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42

FISHING_COLOR = (255, 204, 0)
PATROL_COLOR = (0, 255, 102)

BASE_FPS = 10.0                      # game frames per real second at 1x
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_SPEED_IDX = 3                # 1x
MAX_MESSAGES = 20

BAR_HEIGHT = 10
BAR_BOTTOM_MARGIN = 40


# =========================================================================
# REPLAY DATA
# =========================================================================
class ReplayLog:
    """Holds one game's frame arrays plus derived event messages."""

    def __init__(self, store_path, metric, how):
        store = GameStore.load(store_path)
        (g,) = store.select(metric=metric, how=how)
        data = store.game_data(g)

        self.game_id = g
        self.frames = {k: np.asarray(v) for k, v in data["frames"].items()}
        self.n_steps = self.frames["poachers.x"].shape[0]
        self.n_poachers = self.frames["poachers.x"].shape[1]
        self.n_patrols = self.frames["patrols.x"].shape[1]
        self.t = self.frames.get("t")

        # The log has no names, so generate simple ones
        self.poacher_names = [f"POACHER {k + 1}" for k in range(self.n_poachers)]
        self.patrol_names = [f"PATROL {k + 1}" for k in range(self.n_patrols)]

        self.events = self._build_events()
        self.event_frames = [f for f, _ in self.events]

    def time_at(self, i):
        return float(self.t[i]) if self.t is not None else float(i)

    def _build_events(self):
        """Derive messages from state transitions so no event data is needed."""
        events = []

        caught = self.frames["poachers.caught"].astype(bool)
        for t, k in np.argwhere(caught[1:] & ~caught[:-1]):
            events.append((int(t) + 1, f"ALERT: {self.poacher_names[k]} captured!"))

        for prefix, names in (("poachers", self.poacher_names), ("patrols", self.patrol_names)):
            active = self.frames[f"{prefix}.active"].astype(bool)
            for t, k in np.argwhere(active[:-1] & ~active[1:]):
                events.append((int(t) + 1, f"{names[k]} went inactive (aground or out of bounds)"))

        events.sort(key=lambda e: e[0])
        return events

    def messages_at(self, i):
        """Last MAX_MESSAGES events that have happened by frame i."""
        j = bisect.bisect_right(self.event_frames, i)
        return [text for _, text in self.events[max(0, j - MAX_MESSAGES):j]]


# =========================================================================
# VESSELS
# =========================================================================
def build_vessels(log):
    f = log.frames
    poachers, patrols = [], []

    for k in range(log.n_poachers):
        poachers.append(mf.simulation.PoacherVessel(
            name=log.poacher_names[k],
            start_x=float(f["poachers.x"][0, k]),
            start_y=float(f["poachers.y"][0, k]),
            home_port_id=None,
            start_heading=float(f["poachers.heading"][0, k]),
        ))

    for k in range(log.n_patrols):
        patrols.append(mf.simulation.PatrolVessel(
            name=log.patrol_names[k],
            start_x=float(f["patrols.x"][0, k]),
            start_y=float(f["patrols.y"][0, k]),
            start_heading=float(f["patrols.heading"][0, k]),
        ))

    boat_views = (
        [mf.rendering.BoatView(v, FISHING_COLOR, boat_size_px=12.0) for v in poachers]
        + [mf.rendering.BoatView(v, PATROL_COLOR, boat_size_px=12.0) for v in patrols]
    )
    return poachers, patrols, boat_views


def apply_frame(log, poachers, patrols, i):
    """Copy the recorded state for frame i onto the vessel objects."""
    f = log.frames

    for k, v in enumerate(poachers):
        v.x = float(f["poachers.x"][i, k])
        v.y = float(f["poachers.y"][i, k])
        v.current_heading = float(f["poachers.heading"][i, k])
        v.current_speed = float(f["poachers.speed"][i, k])
        v.is_active = bool(f["poachers.active"][i, k])
        v.cargo = float(f["poachers.cargo"][i, k])
        v.is_caught = bool(f["poachers.caught"][i, k])
        v.is_fishing = bool(f["poachers.fishing"][i, k])

    for k, v in enumerate(patrols):
        v.x = float(f["patrols.x"][i, k])
        v.y = float(f["patrols.y"][i, k])
        v.current_heading = float(f["patrols.heading"][i, k])
        v.current_speed = float(f["patrols.speed"][i, k])
        v.is_active = bool(f["patrols.active"][i, k])


# =========================================================================
# PLAYBACK CONTROL
# =========================================================================
class Playback:
    def __init__(self, n_steps):
        self.n = n_steps
        self.pos = 0.0
        self.playing = True
        self.speed_idx = DEFAULT_SPEED_IDX

    @property
    def index(self):
        return min(int(self.pos), self.n - 1)

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]

    def update(self, dt):
        if not self.playing:
            return
        self.pos += dt * BASE_FPS * self.speed
        if self.pos >= self.n - 1:
            self.pos = float(self.n - 1)
            self.playing = False

    def seek(self, idx):
        self.pos = float(np.clip(idx, 0, self.n - 1))

    def toggle(self):
        if not self.playing and self.index >= self.n - 1:
            self.pos = 0.0
        self.playing = not self.playing

    def change_speed(self, delta):
        self.speed_idx = int(np.clip(self.speed_idx + delta, 0, len(SPEEDS) - 1))


def timeline_rect(W, H):
    return pygame.Rect(14, H - BAR_BOTTOM_MARGIN, W - 28, BAR_HEIGHT)


def draw_timeline(screen, bar, pb, font, log):
    pygame.draw.rect(screen, (60, 60, 60), bar)
    frac = pb.index / max(1, pb.n - 1)
    pygame.draw.rect(screen, (200, 200, 200), (bar.left, bar.top, int(bar.width * frac), bar.height))
    handle_x = bar.left + int(bar.width * frac)
    pygame.draw.circle(screen, (255, 255, 255), (handle_x, bar.centery), 7)

    state = "PAUSED" if not pb.playing else "PLAYING"
    hud = (f"{state}  {pb.speed:g}x  frame {pb.index}/{pb.n - 1}  "
           f"t={log.time_at(pb.index):.2f}  game {log.game_id}")
    controls = ("SPACE play/pause   [ ] speed   , . step   HOME/END   "
                "H/J/K overlays   F flow   L labels   R fit   ESC quit")
    screen.blit(font.render(hud, True, (230, 230, 230)), (14, bar.top - 22))
    screen.blit(font.render(controls, True, (160, 160, 160)), (14, bar.bottom + 8))


# =========================================================================
# MAIN LOOP
# =========================================================================
def run(log, poachers, patrols, boat_views, map_view, max_frames=None):
    screen = pygame.display.get_surface()
    world_w, world_h = map_view.map_data.width_km, map_view.map_data.height_km

    cam = Camera(world_w, world_h)
    cam.fit(*screen.get_size())
    renderer = MapViewerRenderer(cam, fishing_color=FISHING_COLOR, patrol_color=PATROL_COLOR)
    font = pygame.font.SysFont("consolas,menlo,monospace", 14)

    pb = Playback(log.n_steps)
    show_labels = True
    dragging = False
    scrubbing = False
    measuring = False
    measure_start_screen = (0, 0)

    clock = pygame.time.Clock()
    frame = 0
    running = True

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)   # clamp so window drags don't skip the replay
        screen = pygame.display.get_surface()
        W, H = screen.get_size()
        cam.W, cam.H = W, H
        bar = timeline_rect(W, H)

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
                    pb.toggle()
                elif ev.key == pygame.K_LEFTBRACKET:
                    pb.change_speed(-1)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    pb.change_speed(+1)
                elif ev.key == pygame.K_COMMA:
                    pb.playing = False
                    pb.seek(pb.index - 1)
                elif ev.key == pygame.K_PERIOD:
                    pb.playing = False
                    pb.seek(pb.index + 1)
                elif ev.key == pygame.K_HOME:
                    pb.seek(0)
                elif ev.key == pygame.K_END:
                    pb.seek(pb.n - 1)

            elif ev.type == pygame.MOUSEWHEEL:
                cam.zoom_at(*pygame.mouse.get_pos(), 1.2 ** ev.y)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1 and bar.inflate(0, 16).collidepoint(ev.pos):
                    scrubbing = True
                    pb.playing = False
                    pb.seek(((ev.pos[0] - bar.left) / bar.width) * (pb.n - 1))
                elif ev.button in (1, 2):
                    dragging = True
                elif ev.button == 3:
                    measuring = True
                    measure_start_screen = ev.pos

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    scrubbing = False
                    dragging = False
                elif ev.button == 2:
                    dragging = False
                elif ev.button == 3:
                    measuring = False

            elif ev.type == pygame.MOUSEMOTION:
                if scrubbing:
                    frac = np.clip((ev.pos[0] - bar.left) / bar.width, 0.0, 1.0)
                    pb.seek(frac * (pb.n - 1))
                elif dragging:
                    cam.pan_pixels(*ev.rel)

        # Keyboard pan/zoom
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

        # Advance playback and sync vessels to the current recorded frame
        pb.update(dt)
        idx = pb.index
        apply_frame(log, poachers, patrols, idx)
        ui_messages = log.messages_at(idx)

        renderer.render(
            screen=screen,
            sim_time_hours=log.time_at(idx),
            map_view=map_view,
            boat_views=boat_views,
            ui_messages=ui_messages,
            show_labels=show_labels,
            is_paused=not pb.playing,
            measuring=measuring,
            measure_start_screen=measure_start_screen,
        )

        draw_timeline(screen, bar, pb, font, log)
        pygame.display.flip()

        frame += 1
        if max_frames is not None and frame >= max_frames:
            break


def main():
    pygame.init()
    pygame.display.set_mode((900, 900), pygame.RESIZABLE)
    pygame.display.set_caption("MARL Fishing Replay")

    print("Loading map...")
    generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
    map_data = generator.generate_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)
    map_view = mf.rendering.MapView(map_data)

    print("Loading game log...")
    log = ReplayLog(STORE_PATH, GAME_METRIC, GAME_HOW)
    print(f"Game {log.game_id}: {log.n_steps} frames, "
          f"{log.n_poachers} poachers, {log.n_patrols} patrols, {len(log.events)} events")

    poachers, patrols, boat_views = build_vessels(log)
    run(log, poachers, patrols, boat_views, map_view)

    pygame.quit()


if __name__ == "__main__":
    main()