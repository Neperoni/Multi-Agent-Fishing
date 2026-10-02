# multiagent_fishing/rendering/boat_view.py

import math
import pygame
import pygame.gfxdraw

class BoatView:
    """
    Renders a simulation vessel onto a Pygame surface using camera tracking coordinates.
    Decouples raw numerical coordinates from visual drawing operations.
    """
    HULL_SQUARE = [(-0.3, 0.2), (0.3, 0.2), (0.3, -0.5), (-0.3, -0.5)]
    BOW_TRIANGLE = [(-0.3, 0.2), (0.3, 0.2), (0.0, 0.6)]

    def __init__(self, vessel, color, boat_size_px=14.0):
        self.vessel = vessel
        self.color = color
        self.size = boat_size_px

    def draw(self, surf, cam, labels_on, cargo_on, velocity_on, label_cache):
        """Renders the vessel and its conditional telemetry labels if within view thresholds."""
        if not self.vessel.is_active:
            return  # Don't render sunken or caught ships

        # Convert vessel positions directly to screen coordinates via camera parameters
        sx, sy = cam.to_screen(self.vessel.x, self.vessel.y)
        
        # Margins check to ignore off-screen elements safely
        margin = 80
        if not (-margin <= sx <= cam.W + margin and -margin <= sy <= cam.H + margin):
            return

        # Vector rotation: 0 radians points North/Up, increasing clockwise
        heading = self.vessel.current_heading
        c, s = math.cos(heading), math.sin(heading)

        def transform_points(pts):
            return [(round(sx + (vx * c + vy * s) * self.size), 
                     round(sy + (vx * s - vy * c) * self.size)) for vx, vy in pts]

        # Draw the physical hull segments
        for poly in (transform_points(self.HULL_SQUARE), transform_points(self.BOW_TRIANGLE)):
            pygame.gfxdraw.filled_polygon(surf, poly, self.color)
            pygame.gfxdraw.aapolygon(surf, poly, (0, 0, 0))

        # Render floating contextual text metrics
        if labels_on:
            # Base horizontal offset to place labels to the right of the vessel hull
            label_x = round(sx) + round(self.size * 0.4) + 2
            
            # 1. Primary Name Label
            lbl_name = label_cache.get(self.vessel.name, self.color)
            current_y = round(sy) - lbl_name.get_height() // 2
            surf.blit(lbl_name, (label_x, current_y))
            current_y += lbl_name.get_height() + 2  # Step down for any stacked telemetry

            # 2. Cargo Telemetry Label (Only shown if enabled and property exists on the subclass)
            if cargo_on and hasattr(self.vessel, 'cargo'):
                cargo_str = f"Cargo: {self.vessel.cargo:.1f}/{self.vessel.max_cargo:.0f} t"
                lbl_cargo = label_cache.get(cargo_str, (200, 200, 255))  # Slight light-blue tint for telemetry data
                surf.blit(lbl_cargo, (label_x, current_y))
                current_y += lbl_cargo.get_height() + 2

            # 3. Velocity Telemetry Label (Shown if enabled)
            if velocity_on:
                vel_str = f"Speed: {self.vessel.current_speed:.1f} km/h"
                lbl_vel = label_cache.get(vel_str, (200, 255, 200))  # Slight light-green tint for telemetry data
                surf.blit(lbl_vel, (label_x, current_y))
