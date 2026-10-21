from pathlib import Path
import jax
import jax.numpy as jnp
import multiagent_fishing as mf

from multiagent_fishing.jax_sim.game_store import GameStore
# Cleaned up map imports to remove the offline modification function
from multiagent_fishing.jax_sim.maps import JaxMap
from multiagent_fishing.jax_sim.env import FishingEnv, EnvConfig, POACHER, PATROL
from multiagent_fishing.jax_sim.rollout import make_rollout_fn

# Imported your dummy brain classes directly
from multiagent_fishing.brains.dummy_brain import DummyPoacherBrain, DummyPatrolBrain
import numpy as np


# =========================================================================
# CONFIG
# =========================================================================
FILE_PATH = "datasets/gebco_2026_n55.0_s35.0_w-135.0_e-118.0.nc"
PORTS_PATH = "datasets/ports.json"
RESOLUTION = 1.0  # km per grid cell
COASTAL_SUBSET = {"lon_min": -126.0, "lon_max": -122.0, "lat_min": 42.0, "lat_max": 50.0}
BIOMASS_SEED = 42

KM_PER_DEGREE = 111.32
SHIP_LABEL_THRESHOLD_KM = 4000.0  # Basically always on
PORT_LABEL_THRESHOLD_KM = 700.0   
CARGO_LABEL_THRESHOLD_KM = 350.0  # Cargo displays when fairly close
VELOCITY_LABEL_THRESHOLD_KM = 150.0 # Velocity displays only at extreme zoom

MAX_SCALE = 60.0            # max zoom, in screen pixels per km

BG_COLOR = (8, 10, 16)
FISHING_COLOR = (255, 204, 0)
PATROL_COLOR = (0, 255, 102)
PORT_COLOR = (255, 0, 127)

# --- Configurable Relative Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "../datasets/jax_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  

OUTPUT_FILENAME = "generation_001.npz"
FINAL_SAVE_PATH = str(OUTPUT_DIR / OUTPUT_FILENAME)


# =========================================================================
# EXECUTION WITH ASYNC PROGRESS TRACKING
# =========================================================================

# 1. Load data via your existing package MapGenerator
print("Loading data via package MapGenerator...")
generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
map_data = generator.generate_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)

# 2. Ship the pre-computed host datasets onto device arrays inside JaxMap directly
numpy_map = JaxMap.from_numpy_map(map_data)

# 3. Initialize the pure physical simulation environment passing your static map 
env_cfg = EnvConfig(num_poachers=8, num_patrols=3, max_steps=1000)
env = FishingEnv(cfg=env_cfg, map_data=numpy_map)

# 4. Instantiate your swappable Dummy Brain classes
nets = {
    POACHER: DummyPoacherBrain(),
    PATROL: DummyPatrolBrain()
}

# Dummy parameters dictionary map used to satisfy the functional loop setup
params = {
    POACHER: None,
    PATROL: None
}

key = jax.random.PRNGKey(0)

# 5. Build your decoupled, parallelized rollout simulation function
print("Compiling production execution rollout engine...")
rollout = make_rollout_fn(env, nets, num_envs=4096, horizon=128)

# 6. Execute the simulation loop via XLA on your hardware device
print("Running parallelized environment instances...")
traj = rollout(params, key)

# 7. Extract snapshots and package data tracking records into GameStore
store = GameStore.from_rollout(
    traj, numpy_map, env.cfg,
    meta={"seed": 0, "resolution_km": RESOLUTION},
)
store.save(FINAL_SAVE_PATH)
print(f"Saved {len(store.games)} games to {FINAL_SAVE_PATH}")
