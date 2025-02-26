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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

import astropy.constants as const
import astropy.units as unit
import numpy as np
import numpy.typing as npt
from astropy import units
from astropy.units.core import Unit as AstropyUnit
from astropy.units.quantity import Quantity as AstropyQuantity
from astropy.modeling.models import BlackBody
from helios.star_tool.functions import main_loop as startool_mainloop
from scipy.interpolate import interp1d

# ================================================================================================
#   LOGGER
# ================================================================================================

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


# ================================================================================================
#   STAR
# ================================================================================================


class Star:
    """
    Class to store stellar parameters, as well as to load and process spectra
    """

    name: str
    mass: float
    radius: float
    t_eff: float
    distance: float
    metallicity: float
    logg: float
    file_or_blackbody: bool
    source_file: str
    spectral_emittance_fitfunc: Callable

    def __init__(
        self,
        name: str,
        mass: float,
        radius: float,
        t_eff: float,
        distance: float,
        metallicity: float,
    ) -> None:
        self.name = name
        self.mass = mass
        self.radius = radius
        self.t_eff = t_eff
        self.distance = distance
        self.metallicity = metallicity

        # derived params
        grav: float = const.G * self.mass.to("kg") / (self.radius.to("m") ** 2)
        self.logg = np.log10(grav.to("cm / s**2").value)

        # default init
        self.file_or_blackbody: str = "blackbody"
        self.source_file: str = None
        self.spectral_emittance_fitfunc: Callable = None

        # stellar spectrum
        self._orig_wavl: np.ndarray = np.zeros(0)
        self._orig_spectral_emittance: np.ndarray = np.zeros(0)

        self.wavl: np.ndarray = np.zeros(0)
        self.spectral_emittance: np.ndarray = np.zeros(0)

        self.path_in_h5 = "./"

    def get_info(self) -> dict:
        """Returns a dictionary containing information about the star."""
        _dict = self.__dict__.copy()
        info_dict = {}
        for key, value in _dict.items():

            # Do not store raw file path. TODO: because it might not be a string, JSON cannot
            # handle it (called by pipeline.info_dump); fix this?
            if key == "source_file":
                continue

            if isinstance(value, (AstropyUnit, AstropyQuantity)):
                info_dict[key] = value.value
            elif isinstance(value, (np.ndarray, Callable)):
                continue
            else:
                info_dict[key] = value

        return info_dict

    def get_spectrum_from_file(
        self,
        source_file: os.PathLike,
        outdir: os.PathLike,
        opac_file_for_lambdagrid: os.PathLike,
        plot_and_tweak: bool = False,
        skiprows: int = 0,
        w_conversion_factor: float = 1,
        flux_conversion_factor: float = 1,
    ):
        """
        Reads and processes a spectrum from a file such that HELIOS can read it.

        Parameters
        ----------
            source_file : os.PathLike
                File to read-in.
            outdir: os.PathLike
                Where to store the final stellar spectrum file.
            opac_file_for_lambdagrid: os.PathLike
                An opacity file for a future HELIOS run as reference to evaluate the star's
                spectrum on the right wavelength grid.
            plot_and_tweak: bool
                Make star_tool plot the final spectrum,
                allowing to manually tweak it (maybe necessary for phoenix,
                especially for mid-to-far infrared wavelengths)
            skiprows : int
                Number of header rows to skip.
            w_conversion_factor: float
                Wavelength conversion factor. The factor that is needed to convert input wave-
                lengths into cm.
            flux_conversion_factor : float
                Flux units conversion factor. The factor that is needed to convert input wave-
                lengths into erg / s / cm³.
        """
        if not outdir.endswith("/"):
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        self.source_file = source_file

        startool_params = {
            "data_format": "ascii",
            "source_file": self.source_file,
            "name": self.name,
            "w_conversion_factor": w_conversion_factor,
            "flux_conversion_factor": flux_conversion_factor,
            "temp": self.t_eff.value,
            "distance_from_Earth": self.distance / unit.parsec,
            "R_star": self.radius / const.R_sun,
        }
        startool_output_file = outdir + self.name + ".h5"
        _orig_wavl, _orig_spectral_emittance, wavl, converted_flux = startool_mainloop(
            startool_params,
            convert_to="r50_kdistr",
            opac_file_for_lambdagrid=opac_file_for_lambdagrid,
            output_file=startool_output_file,
            plot_and_tweak="no" if not plot_and_tweak else "yes",
            save_in_hdf5="yes",
            skiprows=skiprows,
        )

        self.file_or_blackbody = "file"
        self.path_to_h5 = startool_output_file
        self.path_in_h5 = "r50_kdistr/ascii/" + self.name

        self._orig_wavl = np.array(_orig_wavl) * 1e4  # from Angstroem to micron
        self.wavl = np.array(wavl) * 1e4
        self._orig_spectral_emittance = np.array(_orig_spectral_emittance) * 1e-13
        self.spectral_emittance = np.array(converted_flux) * 1e-13
        self.spectral_emittance_fitfunc = interp1d(
            self._orig_wavl,
            self._orig_spectral_emittance,
            bounds_error=False,
            fill_value=0.0,
        )

    def get_phoenix_spectrum(
        self,
        outdir: str,
        opac_file_for_lambdagrid: str,
        plot_and_tweak: bool = False,
    ) -> None:
        """
        Obtain spectrum from Phoenix.

        Parameters
        ----------
            outdir : str
                Where to place the output file
            opac_file_for_lambdagrid : str
                Reference for wavelength - must be same as planet.
            plot_and_tweak: bool
                Make star_tool plot the final spectrum, allowing to manually tweak it (maybe
                necessary for phoenix, especially for mid-to-far infrared wavelengths).
        """
        if not outdir.endswith("/"):
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        startool_params = {
            "data_format": "phoenix",
            "name": self.name,
            "temp": self.t_eff.value,
            "log_g": self.logg,
            "m": self.metallicity.value,
        }

        startool_output_file = outdir + self.name + ".h5"
        _orig_wavl, _orig_spectral_emittance, wavl, converted_flux = startool_mainloop(
            startool_params,
            convert_to="r50_kdistr",
            opac_file_for_lambdagrid=opac_file_for_lambdagrid,
            output_file=startool_output_file,
            plot_and_tweak="no" if not plot_and_tweak else "yes",
            save_in_hdf5="yes",
        )

        self.file_or_blackbody = "file"
        self.path_to_h5 = startool_output_file
        self.path_in_h5 = "r50_kdistr/ascii/" + self.name

        self._orig_wavl = np.array(_orig_wavl) * 1e4  # from Angstroem to micron
        self.wavl = np.array(wavl) * 1e4
        self._orig_spectral_emittance = np.array(_orig_spectral_emittance) * 1e-13
        self.spectral_emittance = np.array(converted_flux) * 1e-13
        self.spectral_emittance_fitfunc = interp1d(
            self._orig_wavl,
            self._orig_spectral_emittance,
            bounds_error=False,
            fill_value=0.0,
        )

    def get_muscles_spectrum(
        self,
        outdir: str,
        opac_file_for_lambdagrid: str,
        plot_and_tweak: bool = False,
    ) -> None:
        """
        Obtain spectrum from MUSCLES.

        Parameters
        ----------
            outdir : str
                Where to place the output file
            opac_file_for_lambdagrid : str
                Reference for wavelength - must be same as planet
            plot_and_tweak: bool
                Make star_tool plot the final spectrum,
                allowing to manually tweak it (maybe necessary for muscles,
                especially for mid-to-far infrared wavelengths)
        """
        if not outdir.endswith("/"):
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        startool_params = {
            "data_format": "phoenix",
            "name": self.name,
            "temp": self.t_eff.value,
            "log_g": self.logg,
            "m": self.metallicity.value,
        }

        startool_output_file = outdir + self.name + ".h5"
        _orig_wavl, _orig_spectral_emittance, wavl, converted_flux = startool_mainloop(
            startool_params,
            convert_to="r50_kdistr",
            opac_file_for_lambdagrid=opac_file_for_lambdagrid,
            output_file=startool_output_file,
            plot_and_tweak="no" if not plot_and_tweak else "yes",
            save_in_hdf5="yes",
        )

        self.file_or_blackbody = "file"
        self.path_to_h5 = startool_output_file
        self.path_in_h5 = "r50_kdistr/ascii/" + self.name

        self._orig_wavl = np.array(_orig_wavl) * 1e4  # from Angstroem to micron
        self.wavl = np.array(wavl) * 1e4
        self._orig_spectral_emittance = np.array(_orig_spectral_emittance) * 1e-13
        self.spectral_emittance = np.array(converted_flux) * 1e-13
        self.spectral_emittance_fitfunc = interp1d(
            self._orig_wavl,
            self._orig_spectral_emittance,
            bounds_error=False,
            fill_value=0.0,
        )


# ================================================================================================
#   ORBIT
# ================================================================================================


class Orbit(ABC):
    """
    Relates orbital period <-> semi-major axis.
    """

    @abstractmethod
    def get_period(self, star_mass: AstropyUnit) -> AstropyUnit:
        """
        Returns the period of a circular orbit of given semi-major axis and stellar mass.

        Parameters
        ----------
            star_mass: astropy.core.units.Quantity
                Stellar mass.

        Returns
        -------
            period : astropy.core.units.Quantity
                Orbital period of the planet.
        """

    @abstractmethod
    def get_semimajor_axis(self, star_mass: AstropyUnit) -> AstropyUnit:
        """
        Returns the semi-major axis of a circular orbit of given orbital period and stellar mass.

        Parameters
        ----------
            star_mass: astropy.core.units.Quantity
                Stellar mass.

        Returns
        -------
            semi_major_axis : astropy.core.units.Quantity
                Semi-major axis of the planet's orbit.
        """


class CircularOrbitFromPeriod(Orbit):
    """
    Properties of a circular orbit based on its period.

    Parameters
    ----------
        period : astropy.units.core.Unit
            Period of the planet.
    """

    period: AstropyUnit

    def __init__(self, period: AstropyUnit):
        self.period = period.to("day")

    def get_period(self, star_mass: AstropyUnit) -> AstropyUnit:
        """
        Returns the period of a circular orbit of given semi-major axis and stellar mass.

        Parameters
        ----------
            star_mass: astropy.core.units.Quantity
                Stellar mass.

        Returns
        -------
            period : astropy.core.units.Quantity
                Orbital period of the planet.
        """
        return self.period

    def get_semimajor_axis(self, star_mass: AstropyUnit) -> AstropyUnit:
        """
        Returns the semi-major axis of a circular orbit of given orbital period and stellar mass.

        Parameters
        ----------
            star_mass: astropy.core.units.Quantity
                Stellar mass.

        Returns
        -------
            semi_major_axis : astropy.core.units.Quantity
                Semi-major axis of the planet's orbit.
        """
        return np.cbrt(
            const.G * star_mass.to("kg") * self.period.to("s") ** 2 / (4.0 * np.pi)
        ).to("AU")


class CircularOrbitFromSemiMajorAxis(Orbit):
    """
    Properties of a circular orbit based on its semi-major axis.

    Parameters
    ----------
        semi_major_axis : astropy.units.core.Unit
            Semi-major axis of the planet's orbit.
    """

    semi_major_axis: AstropyUnit

    def __init__(self, semi_major_axis: AstropyUnit):
        self.semi_major_axis = semi_major_axis.to("AU")

    def get_period(self, star_mass: AstropyUnit) -> AstropyUnit:
        """
        Returns the period of a circular orbit of given semi-major axis and stellar mass.

        Parameters
        ----------
            star_mass: astropy.core.units.Quantity
                Stellar mass.

        Returns
        -------
            period : astropy.core.units.Quantity
                Orbital period of the planet.
        """
        return (
            2
            * np.pi
            * np.sqrt(
                self.semi_major_axis.to("m") ** 3 / (const.G * star_mass.to("kg"))
            ).to("day")
        )

    def get_semimajor_axis(self, star_mass: AstropyUnit) -> AstropyUnit:
        """
        Returns the semi-major axis of a circular orbit of given orbital period and stellar mass.

        Parameters
        ----------
            star_mass: astropy.core.units.Quantity
                Stellar mass.

        Returns
        -------
            semi_major_axis : astropy.core.units.Quantity
                Semi-major axis of the planet's orbit.
        """
        return self.semi_major_axis


# ================================================================================================
#   PLANET
# ================================================================================================


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
    internal_temperature: Union[None, AstropyUnit]
    grav: Union[None, AstropyUnit] = None
    temperature: Union[None, AstropyUnit] = None

    def __post_init__(self):
        self.grav = const.G * self.mass.to("kg") / (self.radius.to("m") ** 2)

    def get_info(self) -> dict:
        info_dict = self.__dict__.copy()
        for key, value in info_dict.items():
            if isinstance(value, (AstropyUnit, AstropyQuantity)):
                info_dict[key] = value.value

        return info_dict


class PlanetarySystem:
    """
    Stores and manipulates data on the planetary system.

    Parameters
    ----------
        star : Star
            Star-object. See `phaethon.celestial_objects.Star`.
        planet : Planet
            Planet-object. See `phaethon.celestial_objects.Planet`.
        orbit : Orbit
            Orbit-object. See `phaethon.celestial_objects.Orbit`.
    """

    star: Star
    planet: Planet
    orbit: Orbit

    def __init__(
        self,
        star: Star,
        planet: Planet,
        orbit: Orbit,
    ):
        self.star = star
        self.planet = planet
        self.orbit = orbit

        self.planet.temperature = self.calc_pl_temp()

    def calc_pl_temp(self) -> AstropyUnit:
        """
        Calculates the temperature of the planet.

        Parameters
        ----------
            bond_albedo: float
                Bond Albedo.
            dilution_factor : float
                Dilution factor, dimensionless.
        Returns
        -------
            temperature : AstropyUnit
                Temperature of the planet.
        """
        semi_major_axis: AstropyUnit = self.orbit.get_semimajor_axis(
            star_mass=self.star.mass
        )
        return (
            (1.0 - self.planet.bond_albedo)
            * self.planet.dilution_factor
            * (self.star.radius.to("m") / semi_major_axis.to("m")) ** 2
            * self.star.t_eff.to("K") ** 4
        ) ** (0.25)

    def get_period(self) -> AstropyUnit:
        """
        Returns the orbital period of the planet.

        Returns
        -------
            period : astropy.core.units.Quantity
                Orbital period of the planet.
        """
        return self.orbit.get_period(star_mass=self.star.mass)

    def get_semimajor_axis(self) -> AstropyUnit:
        """
        Returns the semi-major axis of the orbit.

        Returns
        -------
            semi_major_axis : astropy.core.units.Quantity
                Semi-major axis of the planet's orbit.
        """
        return self.orbit.get_semimajor_axis(star_mass=self.star.mass)

    def set_semimajor_axis_from_pl_temp(self, t_planet: AstropyUnit) -> None:
        """
        Calculates the semimajor axis of the planet given its temperature.
        Updates the internal value (i.e., in self.orbit).

        Parameters
        ----------
            t_planet : astropy.units.core.Unit
                Temperature of the planet
        Returns
        -------
            temperature : AstropyUnit
                Temperature of the planet.
        """

        new_semimajor_axis: AstropyUnit = (
            np.sqrt(self.planet.dilution_factor * (1.0 - self.planet.bond_albedo))
            * self.star.radius.to("m")
            * (self.star.t_eff.to("K") / t_planet.to("K")) ** 2
        ).to("AU")
        self.orbit = CircularOrbitFromSemiMajorAxis(semi_major_axis=new_semimajor_axis)
        self.planet.temperature = t_planet

    def get_info(self) -> dict:
        """
        Returns info dict

        Returns
        -------
            info : dict
                Dictionary containing information on the celestial objects in this system,
                i.e. the star, planet and its orbit.
        """
        return {
            "star": self.star.get_info(),
            "planet": self.planet.get_info(),
            "orbit": {
                "period": self.orbit.get_period(star_mass=self.star.mass).value,
                "semi_major_axis": self.orbit.get_semimajor_axis(
                    star_mass=self.star.mass
                ).value,
            },
        }
