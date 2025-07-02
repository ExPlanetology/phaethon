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

# phaethon
from phaethon.celestial_objects.utils import verify_type
from phaethon.celestial_objects.planet import Planet
from phaethon.celestial_objects.star import Star

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass(frozen=True)
class PlanetarySystem:
    """
    A class to represent a planetary system with a star and a planet.

    This class encapsulates the essential properties of a planetary system, including the star,
    the planet, the semi-major axis of the planet's orbit, its orbital period, and the irradiation
    temperature at the position of the orbit.

    Attributes
    ----------
    star : Star
        The star of the planetary system.
    planet : Planet
        Data of the planet within the system.
    semimajor_axis : AstropyQuantity
        The semi-major axis of the planet's orbit.
    period : AstropyQuantity
        The orbital period of the planet.
    irrad_temp : float
        The irradiation temperature at the position of the planet's orbit.
    info : Dict[str, Union[float, Dict[str, Union[str, float]]]]
        Additional information about the orbit and energy budget, stored as a dictionary.
    """

    star: Star
    """ The star of the system."""
    planet: Planet
    """ Data of the planet. """
    semimajor_axis: AstropyQuantity
    """ Returns the semi-major axis of the planet. """
    period: AstropyQuantity
    """ Returns the period of the planet. """
    irrad_temp: float
    """ Irradiation temperature at position of the orbit. """
    info: Dict[str, Union[float, Dict[str, Union[str, float]]]]
    """ Information about orbit and energy budget, as a string."""

    @classmethod
    def build_from_semimajor_axis(
        cls: Type[T],
        *,
        semimajor_axis: Union[float, AstropyQuantity],
        planet: Planet,
        star: Star,
    ) -> T:
        """
        Constructs a PlanetarySystem instance using the semi-major axis of the planet's orbit.

        This class method creates a new instance of the PlanetarySystem class by specifying the
        semi-major axis of the planet's orbit, along with the planet and star objects. The semi-
        major axis can be provided as either a float or an AstropyQuantity.

        Parameters
        ----------
        semimajor_axis : Union[float, AstropyQuantity]
            The semi-major axis of the planet's orbit. If provided as a float, it will be converted
            to an AstropyQuantity with an assumed unit of astronomical units (AU).
        planet : Planet
            The planet object containing data about the planet.
        star : Star
            The star object representing the star of the planetary system.

        Returns
        -------
        PlanetarySystem
            An instance of the PlanetarySystem class initialized with the provided semi-major axis,
            planet, and star.
        """
        # semimajor axis to proper type
        _semimajor_axis: AstropyQuantity = verify_type(
            semimajor_axis, target_unit=units.AU
        )

        # calculate period (assume circular orbit)
        _period: AstropyQuantity = (
            2
            * np.pi
            * np.sqrt(_semimajor_axis.to("m") ** 3 / (const.G * star.mass.to("kg"))).to(
                "day"
            )
        )

        # irradiation temperature, i.e. temperature of anobject at the position of the orbit
        _irrad_temp: float = (
            (1.0 - planet.bond_albedo)
            * planet.dilution_factor
            * (star.radius.to("m") / _semimajor_axis.to("m")) ** 2
            * star.t_eff.to("K") ** 4
        ) ** (0.25)


        # info on planetary system
        info = {
            "star": star.get_info(),
            "planet": planet.get_info(),
            "orbit": {
                "period [days]": _period.to("day").value,
                "semi_major_axis [AU]": _semimajor_axis.to("AU").value,
            },
        }

        return cls(
            star=star,
            planet=planet,
            semimajor_axis=_semimajor_axis,
            period=_period,
            irrad_temp=_irrad_temp,
            info=info,
        )

    @classmethod
    def build_from_period(
        cls: Type[T],
        *,
        period: Union[float, int, AstropyQuantity],
        planet: Planet,
        star: Star,
    ) -> T:
        """
        Constructs a PlanetarySystem instance using the orbital period of the planet.

        This class method creates a new instance of the PlanetarySystem class by specifying the
        orbital period of the planet, along with the planet and star objects. The period can be
        provided as either a float or an AstropyQuantity.

        Parameters
        ----------
        period : Union[float, AstropyQuantity]
            The orbital period of the planet. If provided as a float, it will be converted to an
            AstropyQuantity with an assumed unit of days.
        planet : Planet
            The planet object containing data about the planet.
        star : Star
            The star object representing the star of the planetary system.

        Returns
        -------
        PlanetarySystem
            An instance of the PlanetarySystem class initialized with the provided orbital period,
            planet, and star.
        """
        # period axis to proper type
        _period: AstropyQuantity = verify_type(period, target_unit=units.day)

        # calculate semi-major axis (assume circular orbit)
        _semimajor_axis: AstropyQuantity = np.cbrt(
            const.G * star.mass.to("kg") * _period.to("s") ** 2 / (4.0 * np.pi)
        ).to("AU")

        # irradiation temperature, i.e. temperature of anobject at the position of the orbit
        _irrad_temp: float = (
            (1.0 - planet.bond_albedo)
            * planet.dilution_factor
            * (star.radius.to("m") / _semimajor_axis.to("m")) ** 2
            * star.t_eff.to("K") ** 4
        ) ** (0.25)

        # info on planetary system
        info = {
            "star": star.get_info(),
            "planet": planet.get_info(),
            "orbit": {
                "period [days]": _period.to("day").value,
                "semi_major_axis [AU]": _semimajor_axis.to("AU").value,
            },
        }

        return cls(
            star=star,
            planet=planet,
            semimajor_axis=_semimajor_axis,
            period=_period,
            irrad_temp=_irrad_temp,
            info=info,
        )

    @classmethod
    def build_from_irrad_temp(
        cls: Type[T],
        *,
        irrad_temp: Union[float, AstropyQuantity],
        planet: Planet,
        star: Star,
    ) -> T:
        """
        Constructs a PlanetarySystem instance using the irradiation temperature at the planet's
        orbit.

        This class method creates a new instance of the PlanetarySystem class by specifying the
        irradiation temperature at the position of the planet's orbit, along with the planet and
        star objects. The irradiation temperature can be provided as either a float or an
        AstropyQuantity.

        Parameters
        ----------
        irrad_temp : Union[float, AstropyQuantity]
            The irradiation temperature at the position of the planet's orbit. If provided as a
            float, it will be converted to an AstropyQuantity with an assumed unit of Kelvin (K).
        planet : Planet
            The planet object containing data about the planet.
        star : Star
            The star object representing the star of the planetary system.

        Returns
        -------
        PlanetarySystem
            An instance of the PlanetarySystem class initialized with the provided irradiation
            temperature, planet, and star.
        """

        # semimajor axis to proper type
        _irrad_temp: AstropyQuantity = verify_type(irrad_temp, target_unit=units.Kelvin)

        # calculate semi-major axis (assume circular orbit)
        _semimajor_axis: AstropyQuantity = (
            np.sqrt(planet.dilution_factor * (1.0 - planet.bond_albedo))
            * star.radius.to("m")
            * (star.t_eff.to("K") / _irrad_temp.to("K")) ** 2
        ).to("AU")

        # calculate period (assume circular orbit)
        _period: AstropyQuantity = (
            2
            * np.pi
            * np.sqrt(_semimajor_axis.to("m") ** 3 / (const.G * star.mass.to("kg"))).to(
                "day"
            )
        )

        # info on planetary system
        info = {
            "star": star.get_info(),
            "planet": planet.get_info(),
            "orbit": {
                "period [days]": _period.to("day").value,
                "semi_major_axis [AU]": _semimajor_axis.to("AU").value,
            },
        }

        return cls(
            star=star,
            planet=planet,
            semimajor_axis=_semimajor_axis,
            period=_period,
            irrad_temp=_irrad_temp,
            info=info,
        )
