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
Utilities for managing celestial objects.
"""

import logging
import warnings
from typing import Union

import numpy as np
import numpy.typing as npt

# pylint: disable = no-member
from astropy import units
from astropy.units.core import Unit as AstropyUnit
from astropy.units.quantity import Quantity as AstropyQuantity
from astropy.modeling.models import BlackBody


logger = logging.getLogger(__name__)


def black_body_spectral_exitance(
    wavl: npt.NDArray[AstropyUnit], temperature: float
) -> npt.NDArray[AstropyUnit]:
    """
    Radiant flux [erg/s] emitted by a black-body surface per unit area, per wavelength.
    """
    bb_planet = BlackBody(temperature=temperature * units.K)
    bb_emission = (
        bb_planet(wavl).to(
            "erg / (s cm3 sr)", equivalencies=units.spectral_density(wavl)
        )
        * np.pi
        * units.steradian
    )
    return bb_emission


def black_body_object_spectrum(
    wavl: npt.NDArray[AstropyUnit], temperature: float, radius: AstropyUnit
) -> npt.NDArray[AstropyUnit]:
    """
    Returns the spectral flux of a sphere, emitting as a black body of given temperature.
    """
    spectral_exitance = black_body_spectral_exitance(wavl=wavl, temperature=temperature)

    return 4 * np.pi * radius.to("cm") ** 2 * spectral_exitance


def verify_type(value: Union[float, int, AstropyQuantity], target_unit: AstropyUnit):
    """
    Verify and convert the type of the input value to an AstropyQuantity with the specified target
    unit.

    Parameters
    ----------
    value : Union[float, int, AstropyQuantity]
        The input value to be verified and converted. Can be a float, integer, or an
        AstropyQuantity.
    target_unit : AstropyUnit
        The target unit to which the input value should be converted if it is a float or integer.

    Returns
    -------
    AstropyQuantity
        The input value converted to an AstropyQuantity with the target unit.

    Raises
    ------
    TypeError
        If the input value is not of type float, int, or AstropyQuantity.
    """
    match value:
        case AstropyQuantity():
            return value.to(target_unit)
        case float():
            warnings.warn(f"No unit for `{value}`. Assuming {target_unit.name}.")
            return value * target_unit
        case int():
            warnings.warn(f"No unit for `{value}`. Assuming {target_unit.name}.")
            return value * target_unit
        case _:
            raise TypeError(
                "semimajor axis must be of unit `float` or `astropy.units.quantity`,"
                + f" not {type(value)}"
            )
