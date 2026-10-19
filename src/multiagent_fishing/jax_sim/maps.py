import jax
import jax.numpy as jnp
from typing import NamedTuple


class GridMap(NamedTuple):
    """Regular terrain grid. Row index follows y, column index follows x."""
    height: jax.Array    # (H, W) elevation; > 0 means land
    biomass: jax.Array   # (H, W) fish density
    x0: float            # km, west edge of the grid
    y0: float            # km, south edge of the grid
    cell_km: float


def _cell_index(grid: GridMap, x, y):
    H, W = grid.height.shape
    col = jnp.clip(jnp.floor((x - grid.x0) / grid.cell_km).astype(jnp.int32), 0, W - 1)
    row = jnp.clip(jnp.floor((y - grid.y0) / grid.cell_km).astype(jnp.int32), 0, H - 1)
    return row, col


def sample_height(grid: GridMap, x, y):
    r, c = _cell_index(grid, x, y)
    return grid.height[r, c]


def sample_biomass(grid: GridMap, x, y):
    r, c = _cell_index(grid, x, y)
    return grid.biomass[r, c]


def in_bounds(grid: GridMap, x, y):
    H, W = grid.height.shape
    return ((x >= grid.x0) & (x < grid.x0 + W * grid.cell_km) &
            (y >= grid.y0) & (y < grid.y0 + H * grid.cell_km))


def sample_water_positions(key, grid: GridMap, n: int, n_candidates: int = 64):
    """Draws candidate points in the map box and keeps the first water cell per agent.
    Avoids a rejection loop, so it stays jittable."""
    H, W = grid.height.shape
    kx, ky = jax.random.split(key)
    xs = grid.x0 + jax.random.uniform(kx, (n_candidates, n)) * W * grid.cell_km
    ys = grid.y0 + jax.random.uniform(ky, (n_candidates, n)) * H * grid.cell_km
    wet = sample_height(grid, xs, ys) <= 0.0
    idx = jnp.argmax(wet, axis=0)           # first wet candidate per agent (0 if none)
    cols = jnp.arange(n)
    return xs[idx, cols], ys[idx, cols]