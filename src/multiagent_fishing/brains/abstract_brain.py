from abc import ABC, abstractmethod
from multiagent_fishing.simulation.observation import Observation


class AbstractBrain(ABC):
    @abstractmethod
    def get_action(self, obs: Observation):
        """Return the next action tuple for the vessel described by `obs`."""