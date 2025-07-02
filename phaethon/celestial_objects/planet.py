#
# Copyright 2024-2025 Fabian L. Seidler
#
# This file is part of Phaethon.
#
# Phaethon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or any later version.
#
# Phaethon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Phaethon.  If not, see <https://www.gnu.org/licenses/>.
#
"""
Managing the physical and orbital parameters of the planet and its host star.
"""

import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Type, TypeVar, Union, Dict, Optional

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

# pylint: disable = no-member
from astropy import units
import astropy.constants as const
from astropy.units.core import Unit as AstropyUnit
from astropy.units.quantity import Quantity as AstropyQuantity
from astropy.modeling.models import BlackBody

from helios.star_tool.functions import main_loop as startool_mainloop

#phaethon
from phaethon.celestial_objects.utils import verify_type


logger = logging.getLogger(__name__)


@dataclass
class Planet:
    """
    Class that holds data on planetary characteristics.
    """

    name: str
    mass: AstropyUnit
    radius: AstropyUnit
    bond_albedo: float
    dilution_factor: float
    internal_temperature: AstropyQuantity = 0.0 * units.Kelvin
    grav: Optional[AstropyQuantity] = None

    def __post_init__(self):
        self.mass = verify_type(self.mass, target_unit=units.M_earth)
        self.radius = verify_type(self.radius, target_unit=units.R_earth)
        self.grav = const.G * self.mass.to("kg") / (self.radius.to("m") ** 2)
        self.internal_temperature = verify_type(self.internal_temperature, target_unit=units.K)

    def get_info(self) -> dict:
        info_dict = self.__dict__.copy()
        for key, value in info_dict.items():
            if isinstance(value, (AstropyUnit, AstropyQuantity)):
                info_dict[key] = value.value

        return info_dict