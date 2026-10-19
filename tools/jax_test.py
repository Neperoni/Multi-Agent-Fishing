from pathlib import Path
import jax
import jax.numpy as jnp
import multiagent_fishing as mf  # Your existing package
from multiagent_fishing.jax_sim.batch_rollout import BatchRollout
from multiagent_fishing.jax_sim.maps import GridMap
from multiagent_fishing.jax_sim.env import FishingEnv, EnvConfig
from multiagent_fishing.jax_sim.policy import make_mlp_actor_critic, build_policies
from multiagent_fishing.jax_sim.rollout import make_rollout_fn
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
# Resolves path relative to THIS file's current directory (tools/)
CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "../datasets/jax_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Automatically creates the jax_runs folder if it doesn't exist

OUTPUT_FILENAME = "generation_001.npz"
FINAL_SAVE_PATH = str(OUTPUT_DIR / OUTPUT_FILENAME)


# =========================================================================
# EXECUTION
# =========================================================================
from tqdm import tqdm

# =========================================================================
# EXECUTION WITH ASYNC PROGRESS TRACKING
# =========================================================================

# 1. Load data via your existing package MapGenerator
print("Loading data via package MapGenerator...")
generator = mf.MapGenerator(FILE_PATH, ports_path=PORTS_PATH)
map_data = generator.generate_map(RESOLUTION, COASTAL_SUBSET, biomass_seed=BIOMASS_SEED)

# 2. Extract arrays from map_data.layers
jax_height = jnp.array(map_data.layers[..., 0])   
jax_biomass = jnp.array(map_data.layers[..., 1])  

# 3. Instantiate the JAX GridMap
grid = GridMap(
    height=jax_height, 
    biomass=jax_biomass, 
    x0=0.0, 
    y0=0.0, 
    cell_km=map_data.resolution_km  
)

# 4. Initialize the JAX vectorized simulation
env = FishingEnv(EnvConfig(num_poachers=8, num_patrols=3), grid)

key = jax.random.PRNGKey(0)
critic_mode = "local" 

# 5. Build policies and evaluate parallel environments
nets, params = build_policies(key, env.dims(critic_mode), make_mlp_actor_critic((256, 256)))
rollout = make_rollout_fn(env, nets, num_envs=4096, horizon=128, critic_mode=critic_mode)

# --- TRACKED EXECUTION PIPELINE ---

# Progress Bar 1: Running the Parallel Simulation
with tqdm(total=1, desc="Running JAX parallel rollouts", bar_format="{l_bar}{bar}| [Running on Device]") as pbar:
    traj = rollout(params, key)
    traj["rewards"]["poacher"].block_until_ready() # Force JAX to finish computation on GPU/TPU
    pbar.update(1)

# Progress Bar 2: Device-to-Host Data Transfer
with tqdm(total=1, desc="Retrieving data from device", bar_format="{l_bar}{bar}| [Copying to Host RAM]") as pbar:
    cpu_traj = jax.device_get(traj)
    pbar.update(1)

print("Finished! Trajectory shape:", cpu_traj["rewards"]["poacher"].shape)

# Progress Bar 3: Flattening and Structuring matrix layers
flat_data = {}
for key, value in tqdm(cpu_traj.items(), desc="Structuring matrix layers", unit="layer"):
    if isinstance(value, dict):
        for sub_key, sub_arr in value.items():
            flat_data[f"{key}_{sub_key}"] = sub_arr
    else:
        flat_data[key] = value

# 8. Wrap it in our manager and save it to disk safely
rollout_batch = BatchRollout(flat_data)
rollout_batch.save(FINAL_SAVE_PATH)
print(f"File successfully written to relative location: {FINAL_SAVE_PATH}")
