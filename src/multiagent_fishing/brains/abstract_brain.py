from abc import ABC, abstractmethod

class AbstractBrain(ABC):
    
    @staticmethod
    @abstractmethod
    def get_action(env, vessel_id, timestep, vessel_state):
        """
        Calculate and return the next action tuple for a vessel.
        Must be implemented by subclasses.
        """
        
        print("Brain not defined")
        pass
