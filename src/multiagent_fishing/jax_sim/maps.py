import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List, Dict, Any
import numpy as np

from multiagent_fishing.world.map import Map

# ----------------------------------------------------------------------------
# Map Channel Layout Constants
# ----------------------------------------------------------------------------
class JaxLayers:
    """Layout of JaxMap.layers, shape (rows, cols, 7). Used as array indices."""
    HEIGHT: int = 0              # meters, >= 0 is land
    BIOMASS: int = 1             # density in [0, 1]
    SHORE_DIST: int = 2          # km to nearest land, 0 on land
    NEAREST_PORT_ID: int = 3     # index into JaxMap, -1 if no port is reachable
    NEAREST_PORT_DIST: int = 4   # km along water to that port
    NEAREST_PORT_DIR_X: int = 5  # unit vector toward that port (world x, or boat-forward)
    NEAREST_PORT_DIR_Y: int = 6  # world y, or boat-right
    LENGTH: int = 7

SEA_LEVEL: float = 0.0
OOB_HEIGHT: float = 1.0            # out-of-bounds points look like land

# ----------------------------------------------------------------------------
# Core JAX Immutable Map Container
# ----------------------------------------------------------------------------
class JaxMap(NamedTuple):
    """Immutable, device-allocated JAX memory container for the simulation.
    
    Houses pre-computed flow fields and layers. All methods are purely functional,
    stateless, and fully compatible with jax.jit and jax.vmap.
    """
    layers: jax.Array          # (rows, cols, 7) float32
    port_dist: jax.Array       # (n_ports, rows, cols) km along water
    port_flow: jax.Array       # (n_ports, 2, rows, cols) unit vectors (x, y)
    width_km: float
    height_km: float
    resolution_km: float       # cell size
    unreachable_km: float      # map diagonal used for "no path" caps
    ports_xy: jax.Array        # (n_ports, 2) pre-extracted port locations
    oob_row: jax.Array         # (7,) out-of-bounds default metrics

    @classmethod
    def from_numpy_map(cls, original_map: Map) -> "JaxMap":
        """Factory method that loads your original host Map class straight into device memory."""
        ports_list: List[Dict[str, Any]] = original_map.ports
        ports_xy: np.ndarray = np.array([[p["grid_x"], p["grid_y"]] for p in ports_list], dtype=np.float32)

        # Assemble the default out-of-bounds row matching historical specs
        oob: np.ndarray = np.zeros(JaxLayers.LENGTH, dtype=np.float32)
        oob[JaxLayers.HEIGHT] = OOB_HEIGHT
        oob[JaxLayers.NEAREST_PORT_ID] = -1.0
        oob[JaxLayers.NEAREST_PORT_DIST] = original_map.unreachable_km

        return cls(
            layers=jnp.array(original_map.layers, dtype=jnp.float32),
            port_dist=jnp.array(original_map.port_dist, dtype=jnp.float32),
            port_flow=jnp.array(original_map.port_flow, dtype=jnp.float32),
            width_km=float(original_map.width_km),
            height_km=float(original_map.height_km),
            resolution_km=float(original_map.resolution_km),
            unreachable_km=float(original_map.unreachable_km),
            ports_xy=jnp.array(ports_xy, dtype=jnp.float32),
            oob_row=jnp.array(oob, dtype=jnp.float32)
        )


# ----------------------------------------------------------------------------
# Pure JAX Geographic Getters (Replaces all original CPU query math)
# ----------------------------------------------------------------------------

def jax_is_in_bounds(map_data: JaxMap, x: jax.Array, y: jax.Array) -> jax.Array:
    """Returns a boolean array masking whether coordinates are within limits."""
    return (x >= 0.0) & (x <= map_data.width_km) & (y >= 0.0) & (y <= map_data.height_km)


def _jax_cell_index(map_data: JaxMap, x: jax.Array, y: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes internal matrix indices while safely clamping bounds to prevent GPU crashes."""
    rows_max: int
    cols_max: int
    rows_max, cols_max, _ = map_data.layers.shape
    inb: jax.Array = jax_is_in_bounds(map_data, x, y)
    
    # Safe indexing trick via clamping limits
    col: jax.Array = jnp.clip(jnp.floor(x / map_data.resolution_km).astype(jnp.int32), 0, cols_max - 1)
    row: jax.Array = jnp.clip(jnp.floor(y / map_data.resolution_km).astype(jnp.int32), 0, rows_max - 1)
    return row, col, inb


def _jax_to_ego(dx: jax.Array, dy: jax.Array, heading: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Rotates world vectors into the egocentric boat frame: (forward, right).
    
    Preserves your original +x = cos, +y = sin heading projection exactly.
    """
    c: jax.Array = jnp.cos(heading)
    s: jax.Array = jnp.sin(heading)
    fwd: jax.Array = dx * c + dy * s
    right: jax.Array = dx * s - dy * c
    return fwd, right


def jax_sample_point(map_data: JaxMap, x: jax.Array, y: jax.Array, heading: jax.Array = None) -> jax.Array:
    """Samples all tracking data channels at specified coordinates.
    
    If heading is supplied, automatically rotates port directions into egocentric frames.
    Accepts vectors / batches: shapes scale to (N, 7).
    """
    row: jax.Array
    col: jax.Array
    inb: jax.Array
    row, col, inb = _jax_cell_index(map_data, x, y)
    
    # Batch indexing lookups
    sampled_data: jax.Array = map_data.layers[row, col]  
    
    # Inject out-of-bounds row smoothly using jnp.where
    out: jax.Array = jnp.where(inb[..., None], sampled_data, map_data.oob_row)
    
    if heading is not None:
        fwd: jax.Array
        right: jax.Array
        fwd, right = _jax_to_ego(
            out[..., JaxLayers.NEAREST_PORT_DIR_X],
            out[..., JaxLayers.NEAREST_PORT_DIR_Y],
            heading
        )
        out = out.at[..., JaxLayers.NEAREST_PORT_DIR_X].set(fwd)
        out = out.at[..., JaxLayers.NEAREST_PORT_DIR_Y].set(right)
        
    return out


def jax_nav_to_specific_port(
    map_data: JaxMap, 
    port_id: jax.Array,  # Shape: (N,) -> Unique target ID per agent
    x: jax.Array,        # Shape: (N,)
    y: jax.Array,        # Shape: (N,)
    heading: jax.Array = None
) -> jax.Array:
    """Instantly extracts path navigation features for unique, arbitrary port targets.
    
    Uses direct JAX tuple indexing to prevent memory bandwidth bloat, compiling down 
    to a hardware-level gather instruction that extracts only the targeted cells.
    
    Output shape: (N, 3) -> [path_distance_km, dir_x, dir_y]
    """
    row: jax.Array
    col: jax.Array
    inb: jax.Array
    row, col, inb = _jax_cell_index(map_data, x, y)

    # Hardware parallel tuple query: map_data.port_dist is (n_ports, rows, cols)
    dist_raw: jax.Array = map_data.port_dist[port_id, row, col]
    dist: jax.Array = jnp.where(inb, dist_raw, map_data.unreachable_km)

    # map_data.port_flow shape is (n_ports, 2, rows, cols)
    flow_x: jax.Array = jnp.where(inb, map_data.port_flow[port_id, 0, row, col], 0.0)
    flow_y: jax.Array = jnp.where(inb, map_data.port_flow[port_id, 1, row, col], 0.0)

    if heading is not None:
        flow_x, flow_y = _jax_to_ego(flow_x, flow_y, heading)
        
    return jnp.stack([dist, flow_x, flow_y], axis=-1)


# ----------------------------------------------------------------------------
# Simulation Infrastructure Hooks (Resets & Spawns)
# ----------------------------------------------------------------------------

def jax_sample_port_positions(key: jax.Array, map_data: JaxMap, num_agents: int) -> Tuple[jax.Array, jax.Array]:
    """Instantly spawns boats uniformly across available ports.
    
    Bypasses rejection loops entirely, ensuring clean JAX traceability."""
    num_ports: int = map_data.ports_xy.shape[0]
    
    # Pick a port index uniformly at random for each agent
    port_indices: jax.Array = jax.random.randint(key, shape=(num_agents,), minval=0, maxval=num_ports)
    selected_ports: jax.Array = map_data.ports_xy[port_indices]  # Shape: (num_agents, 2)
    
    return selected_ports[:, 0], selected_ports[:, 1]
