"""
tribosim — Synthetic reciprocating friction test simulator
===========================================================
Modules
-------
genesis      : CoF evolution over cycles (run-in, piecewise drift, events, AR1 fluctuation)
wear         : Energy-based wear scar growth
kinematics   : Crank-slider, sinusoidal and trapezoidal displacement / velocity profiles (mm/s with f_test)
friction_loop: Tangential force loop (static + kinematic, Oden-Martins)
acquisition  : AR1 coloured-noise signal acquisition model
export       : CSV and shelve export/import helpers

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""

__author__ = "Iñigo Llavori Osa"
__version__ = "1.0.0"
__license__ = "MIT"

from .genesis import FrictionGenesis, GenesisEvent
from .wear import WearModel
from .kinematics import CrankSlider, Sinusoidal, Trapezoidal
from .friction_loop import FrictionLoop
from .acquisition import add_noise
from .export import Exporter, load_session

__all__ = [
    "FrictionGenesis",
    "GenesisEvent",
    "WearModel",
    "CrankSlider",
    "Sinusoidal",
    "Trapezoidal",
    "FrictionLoop",
    "add_noise",
    "Exporter",
    "load_session",
]
