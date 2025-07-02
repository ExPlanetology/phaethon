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
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

# pylint: disable = no-member
from astropy import units
import astropy.constants as const
from astropy.units.core import Unit as AstropyUnit
from astropy.units.quantity import Quantity as AstropyQuantity

from helios.star_tool.functions import main_loop as startool_mainloop

# phaethon


logger = logging.getLogger(__name__)


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
            "distance_from_Earth": self.distance / units.parsec,
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
