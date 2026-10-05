from multiagent_fishing.brains.abstract_brain import AbstractBrain

class DummyPoacherBrain(AbstractBrain):
    @staticmethod
    def get_action(env, vessel_id, timestep, vessel_state):
        heading = vessel_state.current_heading
        
        #todo: check if theres actually fish
        
        fish_toggle = 1 if vessel_state.cargo < vessel_state.max_cargo else 0
        return (heading, 0.5, fish_toggle)

class DummyPatrolBrain(AbstractBrain):
    @staticmethod
    def get_action(env, vessel_id, timestep, vessel_state):
        heading = vessel_state.current_heading
        return (heading, 0.5)
