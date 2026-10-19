import numpy as np
import zipfile
import io

class BatchRollout:
    def __init__(self, trajectory_dict: dict):
        """
        Manages a collection of parallel JAX simulation trajectories.
        Expects a flat dictionary where nested keys have been joined with underscores 
        (e.g., 'rewards_poacher').
        """
        self.data = trajectory_dict
        
        # Pull out the target score metric based on poacher activities
        self.rewards = self.data.get("rewards_poacher") 
        
        if self.rewards is None:
            raise ValueError("Trajectory data must contain a 'rewards_poacher' array.")
            
        self.horizon, self.num_envs, self.num_agents = self.rewards.shape
        
        # Pre-compute total cumulative rewards earned per environment across all agents
        # Shape: (num_envs,)
        self.env_total_rewards = self.rewards.sum(axis=(0, 2))

    @classmethod
    def load(cls, filepath: str):
        """
        Surgically loads ONLY the tracking and reward arrays needed for playback.
        Bypasses massive network arrays (critic_obs, log_probs, values) to prevent 
        Python zip-bomb errors.
        """
        print(f"Loading targeted playback rollout data from {filepath}...")
        
        trajectory_dict = {}
        
        # Define the exact core keys needed by the viewer and metrics calculations
        required_keys = [
            "rewards_poacher", 
            "rewards_patrol", 
            "playback_poachers", 
            "playback_patrols"
        ]
        
        with zipfile.ZipFile(filepath, "r") as zf:
            all_files = zf.namelist()
            
            for key in required_keys:
                filename = f"{key}.npy"
                if filename in all_files:
                    with zf.open(filename) as f:
                        # Stream the specific bytes buffer isolated into memory
                        file_buffer = io.BytesIO(f.read())
                        trajectory_dict[key] = np.load(file_buffer, allow_pickle=True)
                else:
                    if key == "rewards_poacher":
                        raise ValueError(f"Required base rollout criteria metric '{key}' missing from log archive.")
                    else:
                        print(f"Warning: Optional playback key '{key}' missing from archive.")
                        
        print(f"Load complete. Extracted {len(trajectory_dict)} core telemetry arrays.")
        return cls(trajectory_dict)

    def save(self, filepath: str):
        """Helper function to save all rollout data to an .npz archive."""
        print(f"Saving rollout data to {filepath}...")
        np.savez_compressed(filepath, **self.data)
        print("Save complete.")

    def get_env_data(self, env_idx: int) -> dict:
        """Extracts the entire trajectory timeline specifically for one environment index."""
        single_env = {}
        for key, value in self.data.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                # Slices out all steps, only the selected env, and keeps trailing dims intact
                single_env[key] = value[:, env_idx, ...]
            else:
                single_env[key] = value
        return single_env

    def get_best_run(self) -> tuple[int, float, dict]:
        """Returns: (env_index, total_score, env_trajectory_data)"""
        best_idx = int(np.argmax(self.env_total_rewards))
        best_score = float(self.env_total_rewards[best_idx])
        return best_idx, best_score, self.get_env_data(best_idx)

    def get_representative_run(self) -> tuple[int, float, dict]:
        """Returns: (env_index, total_score, env_trajectory_data)"""
        median_score = np.median(self.env_total_rewards)
        rep_idx = int(np.argmin(np.abs(self.env_total_rewards - median_score)))
        rep_score = float(self.env_total_rewards[rep_idx])
        return rep_idx, rep_score, self.get_env_data(rep_idx)
