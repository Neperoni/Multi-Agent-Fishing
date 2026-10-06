from dataclasses import dataclass, field
from typing import Any

from .vessels import Role


@dataclass
class Observation:
    """Everything a brain can see when choosing an action at one timestep.

    To give brains new information, add a field here. No brain signatures change.
    """
    vessel_id: Any
    vessel_state: Any          # the vessel's own state object from env
    timestep: float            # dt for this decision
    role: Role
    env: Any                   # privileged: full global state, for traditional/god-mode brains
    params: dict = field(default_factory=dict)  # local parameters, e.g. detection range