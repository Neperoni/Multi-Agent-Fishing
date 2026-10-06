from multiagent_fishing.brains.abstract_brain import AbstractBrain
from multiagent_fishing.simulation.observation import Observation


class DummyPoacherBrain(AbstractBrain):
    def get_action(self, obs: Observation):
        vessel = obs.vessel_state
        heading = vessel.current_heading

        # todo: check if there's actually fish

        fish_toggle = 1 if vessel.cargo < vessel.max_cargo else 0
        return (heading, 0.5, fish_toggle)


class DummyPatrolBrain(AbstractBrain):
    def get_action(self, obs: Observation):
        heading = obs.vessel_state.current_heading
        return (heading, 0.5)