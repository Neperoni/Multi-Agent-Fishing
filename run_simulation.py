# run_simulation.py
from pathlib import Path
import numpy as np
from src.environment import MaritimeSimEnvironment

def main():
    # Initialize and configure the engine
    sim = MaritimeSimEnvironment()
    sim.load_map(Path("datasets/downscaled_500m.tif"))
    
    # Spawn the assets
    sim.spawn_poacher("Pirate_Alpha", x=50.0, y=120.0)
    sim.spawn_patrol("DFO_One", x=200.0, y=200.0)
    
    print("\nStarting simulation")
    
    # Run the clock frame-by-frame
    for frame in range(100):
        
        actions = {
            "Pirate_Alpha": (np.pi / 4, 0.5), # Heading angle, throttle
            "DFO_Comox": (0.0, 1.0)           # Heading angle, throttle
        }
        
        current_state = sim.step(actions=actions)
        
        vessel_data = current_state["vessels"]

if __name__ == "__main__":
    main()
