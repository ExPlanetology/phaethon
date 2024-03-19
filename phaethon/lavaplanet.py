"""
Module for managing the parameters of the (lava-)planet
"""
from abc import ABC, abstractmethod
import numpy as np
from astropy.units.core import Unit as AstropyUnit
import astropy.constants as const

from phaethon.star import Star


class Orbit(ABC):
    """
    Relates orbital period <-> semi-major axis
    """

    @abstractmethod
    def get_period(self, star_mass: AstropyUnit) -> AstropyUnit:
        ...

    @abstractmethod
    def get_semimajor_axis(self, star_mass: AstropyUnit) -> AstropyUnit:
        ...


class CircularOrbitFromPeriod(Orbit):
    period: AstropyUnit

    def __init__(self, period: AstropyUnit):
        self.period = period.to("day")

    def get_period(self, star_mass: AstropyUnit) -> AstropyUnit:
        return self.period

    def get_semimajor_axis(self, star_mass: AstropyUnit) -> AstropyUnit:
        return np.cbrt(
            const.G * star_mass.to("kg") * self.period.to("s") ** 2 / (4.0 * np.pi)
        ).to("AU")


class CircularOrbitFromSemiMajorAxis(Orbit):
    semi_major_axis: AstropyUnit

    def __init__(self, semi_major_axis: AstropyUnit):
        self.semi_major_axis = semi_major_axis.to("AU")

    def get_period(self, star_mass: AstropyUnit) -> AstropyUnit:
        return (
            2
            * np.pi
            * np.sqrt(
                self.semi_major_axis.to("m") ** 3 / (const.G * star_mass.to("kg"))
            ).to("day")
        )

    def get_semimajor_axis(self, star_mass: AstropyUnit) -> AstropyUnit:
        return self.semi_major_axis


class LavaPlanet:
    def __init__(
        self,
        name: str,
        mass: AstropyUnit,
        radius: AstropyUnit,
        orbit: Orbit,
        t_int: AstropyUnit,
    ):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.grav = const.G * self.mass.to("kg") / (self.radius.to("m") ** 2)
        self._orbit = orbit

    def calc_temp(
        self, star: Star, bond_albedo: float, dilution_factor: float
    ) -> AstropyUnit:
        """
        Calculates the temperature of the planet.

        Parameters
        ----------
            star : phaethon.star.Star
                Class with stellar parameters.
            bond_albedo: float
                Bond Albedo.
            dilution_factor : float
                Dilution factor, dimensionless.
        Returns
        -------
            temperature : AstropyUnit
                Temperature of the planet.
        """
        semi_major_axis: AstropyUnit = self._orbit.get_semimajor_axis(
            star_mass=star.mass
        )
        return (
            (1.0 - bond_albedo)
            * dilution_factor
            * (star.radius.to("m") / semi_major_axis.to("m")) ** 2
            * star.t_eff.to("K")
        ) ** (0.25)

    def calc_semimajor_axis_from_temp(
        self,
        star: Star,
        t_planet: AstropyUnit,
        bond_albedo: float,
        dilution_factor: float,
    ) -> AstropyUnit:
        """
        Calculates the semimajor axis of the planet given its temperature.
        Updates the internal value (i.e., in self._orbit).

        Parameters
        ----------
            star : phaethon.star.Star
                Class with stellar parameters.
            t_planet : astropy.units.core.Unit
                Temperature of the planet
            bond_albedo: float
                Bond Albedo.
            dilution_factor : float
                Dilution factor, dimensionless.
        Returns
        -------
            temperature : AstropyUnit
                Temperature of the planet.
        """

        return (
            np.sqrt(dilution_factor * (1.0 - bond_albedo))
            * star.radius.to("m")
            * (star.t_eff.to("K") / t_planet.to("K")) ** 2
        ).to("AU")
