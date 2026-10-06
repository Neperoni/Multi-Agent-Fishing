# multiagent_fishing/__init__.py

from . import simulation
from . import rendering
from . import brains

# Expose data infrastructure directly at the root level
from .world import Map, MapGenerator, Layers