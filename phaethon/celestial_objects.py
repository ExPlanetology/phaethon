"""
Module for managing the parameters of the (lava-)planet
"""
import os
from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass
import numpy as np
from astropy.units.core import Unit as AstropyUnit
from astropy.units.quantity import Quantity as AstropyQuantity
import astropy.units as unit
import astropy.constants as const
from scipy.interpolate import interp1d

from star_tool.functions import main_loop as startool_mainloop

# ================================================================================================
#   STAR
# ================================================================================================

class Star():
    """
    Class to store stellar parameters and load and process spectra
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
        grav: float = const.G * self.mass.to("kg") / (self.radius.to("m")**2)
        self.logg = np.log(grav.value)

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
        _dict = self.__dict__
        info_dict = {}
        for key, value in _dict.items():
            if isinstance(value, (AstropyUnit, AstropyQuantity)):
                info_dict[key] = value.value
            elif isinstance(value, (np.ndarray, Callable)):
                continue
            else:
                info_dict[key] = value

        return info_dict      

    def get_spectrum_from_file(
        self,
        source_file: str,
        outdir: str,
        opac_file_for_lambdagrid: str,
        plot_and_tweak: bool = False,
        skiprows: int = 0,
        w_conversion_factor=1,
        flux_conversion_factor=1,
    ):
        """
        Reads and processes a spectrum from a file
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
            self._orig_wavl, self._orig_spectral_emittance, bounds_error=False, fill_value=0.0
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
                Reference for wavelength - must be same as planet
            plot_and_tweak: bool
                Make star_tool plot the final spectrum,
                allowing to manually tweak it (maybe necessary for phoenix,
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
            self._orig_wavl, self._orig_spectral_emittance, bounds_error=False, fill_value=0.0
        )

# ================================================================================================
#   ORBIT
# ================================================================================================

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

# ================================================================================================
#   PLANET
# ================================================================================================

@dataclass
class Planet:
    name: str
    mass: AstropyUnit
    radius: AstropyUnit
    bond_albedo: float
    dilution_factor: float
    grav: float = np.nan
    temperature: float = np.nan

    def __post_init__(self):
        self.grav = const.G * self.mass.to("kg") / (self.radius.to("m") ** 2)

    def get_info(self) -> dict:
        info_dict = self.__dict__
        for key, value in info_dict.items():
            if isinstance(value, (AstropyUnit, AstropyQuantity)):
                info_dict[key] = value.value

        return info_dict


class PlanetarySystem:
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
        """ Returns info dict """
        return {
            "star":self.star.get_info(),
            "planet":self.planet.get_info(),
            "orbit":{
                "period":self.orbit.get_period(star_mass=self.star.mass).value,
                "semi_major_axis":self.orbit.get_semimajor_axis(star_mass=self.star.mass).value,
            }
        }
