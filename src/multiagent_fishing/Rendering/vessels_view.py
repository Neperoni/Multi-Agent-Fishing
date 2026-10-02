# =========================================================================
# VESSEL PRESENTATION LAYER (BoatView Wrapper)
# =========================================================================
class BoatView:
    """
    Renders a simulation vessel onto a Pygame surface using camera tracking coordinates.
    Decouples raw numerical coordinates from visual drawing operations.
    """
    # Define geometry primitives matching your original engine vectors
    HULL_SQUARE = [(-0.3, 0.2), (0.3, 0.2), (0.3, -0.5), (-0.3, -0.5)]
    BOW_TRIANGLE = [(-0.3, 0.2), (0.3, 0.2), (0.0, 0.6)]

    def __init__(self, vessel, color, scale_multiplier=1.0):
        """
        :param vessel: An instance of SimulationVessel (or its subclasses)
        :param color: The RGB color tuple used to fill the hull
        :param scale_multiplier: Relative size modifier for different vessel classes
        """
        self.vessel = vessel
        self.color = color
        self.mult = scale_multiplier

    def draw(self, surf, cam, labels_on, label_cache):
        """Renders the vessel and its label if it's within the visible viewport."""
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
        size = BOAT_SIZE_PX * self.mult

        def transform_points(pts):
            return [(round(sx + (vx * c + vy * s) * size), 
                     round(sy + (vx * s - vy * c) * size)) for vx, vy in pts]

        # Draw the physical hull segments
        for poly in (transform_points(self.HULL_SQUARE), transform_points(self.BOW_TRIANGLE)):
            pygame.gfxdraw.filled_polygon(surf, poly, self.color)
            pygame.gfxdraw.aapolygon(surf, poly, (0, 0, 0))

        # Render floating contextual text names from cache if labels are active
        if labels_on:
            lbl = label_cache.get(self.vessel.name, self.color)
            surf.blit(lbl, (round(sx) + 10, round(sy) - 10 - lbl.get_height()))
