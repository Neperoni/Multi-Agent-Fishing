"""
MARL fishing map viewer (pygame)
"""
import math
import json
import random
from pathlib import Path
import numpy as np
import pygame
import multiagent_fishing as mf

from multiagent_fishing.rendering import MapViewerRenderer, Camera

# =========================================================================
# CONFIG
# =========================================================================
FILE_PATH = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
PORTS_PATH = "datasets/ports.json"
RESOLUTION = 1.0
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42

FISHING_COLOR = (255, 204, 0)
PATROL_COLOR = (0, 255, 102)

# Global memory cache to hold name arrays so we only read the disk once at startup
_FRONTEND_NAME_POOL = None


# 2. UPDATED: Retained local simulation lifecycle state helper
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

    port_id = random.randrange(len(env.map_data.ports))
    port = env.map_data.ports[port_id]
    pool_key = "fishing_boats" if is_poacher else "patrol_boats"
    pool = _FRONTEND_NAME_POOL.get(pool_key, {})
    
    first = random.choice(pool.get("first_word", ["MYSTIC" if is_poacher else "VIGILANT"]))
    last = random.choice(pool.get("last_word", ["WAVE" if is_poacher else "GUARDIAN"]))
    vessel_name = f"{first} {last}"
    
    if is_poacher:
        new_vessel = mf.simulation.PoacherVessel(
            name=vessel_name,
            start_x=port["grid_x"],
            start_y=port["grid_y"],
            home_port_id=port_id,
            start_heading=random.uniform(0.0, 2 * math.pi)
        )
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


def run(env, boat_views, map_view, max_frames=None):
    screen = pygame.display.get_surface()
    world_w, world_h = map_view.map_data.width_km, map_view.map_data.height_km
    
    # Initialize Core Camera Systems
    cam = Camera(world_w, world_h)
    cam.fit(*screen.get_size())
    
    # Initialize our package-integrated Composite Renderer
    renderer = MapViewerRenderer(cam, fishing_color=FISHING_COLOR, patrol_color=PATROL_COLOR)

    ui_messages = []
    show_labels = True
    dragging = False
    is_paused = False
    measuring = False
    measure_start_screen = (0, 0)

    sim_time_hours=0

    clock = pygame.time.Clock()
    frame = 0
    running = True

    # =========================================================================
    # MAIN EVENT LOOP
    # =========================================================================
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
                    ui_messages.append("SIMULATION PAUSED" if is_paused else "SIMULATION RESUMED")
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

        # Keyboard Pan/Zoom continuous input processing
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
        # UPDATE SIMULATION PHYSICS
        # =========================================================================
        if not is_paused:
            actions = env.get_actions(dt*0.5)
            step_events = env.step(actions, dt=dt*0.5)
            
            for app in step_events.get("apprehensions", []):
                p_id, pat_id = app["poacher_id"], app["by_patrol_id"]
                p_name = env.poachers[p_id].name if p_id in env.poachers else f"Poacher {p_id}"
                pat_name = env.patrol_boats[pat_id].name if pat_id in env.patrol_boats else f"Patrol {pat_id}"
                ui_messages.append(f"ALERT: {p_name} captured by {pat_name}!")
                spawn_random_vessel(env, boat_views, is_poacher=True)

            for g_id in step_events.get("groundings", []):
                is_poach = g_id in env.poachers
                name = env.poachers[g_id].name if is_poach else (env.patrol_boats[g_id].name if g_id in env.patrol_boats else f"Vessel {g_id}")
                ui_messages.append(f"CRASH: {name} ran aground on land!")
                spawn_random_vessel(env, boat_views, is_poacher=is_poach)

            for o_id in step_events.get("out_of_bounds", []):
                is_poach = o_id in env.poachers
                name = env.poachers[o_id].name if is_poach else (env.patrol_boats[o_id].name if o_id in env.patrol_boats else f"Vessel {o_id}")
                ui_messages.append(f"BOUNDARY: {name} sailed out of bounds!")
                spawn_random_vessel(env, boat_views, is_poacher=is_poach)

            if len(ui_messages) > 20:
                ui_messages = ui_messages[-20:]

        # =========================================================================
        # RENDER ENGINE DRAW PASS
        # =========================================================================
        renderer.render(
            screen=screen,
            sim_time_hours=env.sim_time_hours,
            map_view=map_view,
            boat_views=boat_views,
            ui_messages=ui_messages,
            show_labels=show_labels,
            is_paused=is_paused,
            measuring=measuring,
            measure_start_screen=measure_start_screen
        )

        pygame.display.flip()
        frame += 1
        if max_frames is not None and frame >= max_frames:
            break


def main():
    pygame.init()
    pygame.display.set_mode((900, 900), pygame.RESIZABLE)
    pygame.display.set_caption("MARL Fishing Environment")

    print("Loading data via package MapGenerator...")
    generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
    map_data = generator.generate_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)
    
    map_view = mf.rendering.MapView(map_data)
    env = mf.simulation.Environment(map_data)

    # Initialize empty collections for runtime structures
    boat_views = []

    #Spawn initial fleet randomly at ports
    num_poachers = 12
    num_patrols = 5

    for _ in range(num_poachers):
        spawn_random_vessel(env, boat_views, is_poacher=True)

    for _ in range(num_patrols):
        spawn_random_vessel(env, boat_views, is_poacher=False)

    # Attach the AI brains
    env.set_poacher_brain(mf.brains.BasicPoacherBrain())
    env.set_patrol_brain(mf.brains.BasicPatrolBrain())

    # Hand off collections directly to the runner loop
    run(env, boat_views, map_view)
    pygame.quit()
    
    
if __name__ == "__main__":
    main()