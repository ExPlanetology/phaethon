"""
Module to load and process stellar spectra
"""
from typing import Callable
import os
import sys
import numpy as np
import astropy.units as unit
import astropy.constants as const
from scipy.interpolate import interp1d

sys.path.append("/home/fabian/LavaWorlds/HELIOS")
from star_tool.functions import main_loop as startool_mainloop


class Star(object):
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
    spec_fit: Callable

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
        self.mass = mass * const.M_sun
        self.radius = radius * const.R_sun
        self.t_eff = t_eff * unit.K
        self.distance = distance * unit.pc
        self.metallicity = metallicity * unit.dex

        # derived params
        grav: float = const.G * self.mass / (self.radius**2)
        self.logg = np.log(grav.value)

        # default init
        self.file_or_blackbody = "blackbody"
        self.source_file = None
        self.spec_fit = None

    def get_spectrum_from_file(
        self,
        source_file: str,
        outdir: str,
        opac_file_for_lambdagrid: str,
        plot_and_tweak: bool = False,
        skiprows: int = 0,
        resolution: int=200,
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
        orig_lambda, orig_flux, new_lambda, converted_flux = startool_mainloop(
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

        self.orig_lambda = np.array(orig_lambda) * 1e4  # from Angstroem to micron
        self.new_lambda = np.array(new_lambda) * 1e4
        self.orig_flux = np.array(orig_flux) * 1e-13
        self.new_flux = np.array(converted_flux) * 1e-13
        self.spec_fit = interp1d(
            self.orig_lambda, self.orig_flux, bounds_error=False, fill_value=0.0
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
        orig_lambda, orig_flux, new_lambda, converted_flux = startool_mainloop(
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

        self.orig_lambda = np.array(orig_lambda) * 1e4  # from Angstroem to micron
        self.new_lambda = np.array(new_lambda) * 1e4
        self.orig_flux = np.array(orig_flux) * 1e-13
        self.new_flux = np.array(converted_flux) * 1e-13
        self.spec_fit = interp1d(
            self.orig_lambda, self.orig_flux, bounds_error=False, fill_value=0.0
        )


if __name__ == "__main__":
    star = Star(
        name="Phoenix",
        mass=1.0,
        radius=1.0,
        t_eff=5770.0,
        distance=10.0,
        metallicity=0.0,
    )
    # star.get_phoenix_spectrum(
    #     outdir="./",
    #     opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5",
    #     plot_and_tweak=True,
    # )
    star.get_spectrum_from_file(
        outdir="output/",
        source_file="/home/fabian/LavaWorlds/phaethon/phaethon/star_tool/sun_gueymard_2003_modified.txt",
        opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5",
        skiprows=9,
        plot_and_tweak=False,
        w_conversion_factor=1e-7,
        flux_conversion_factor=1e10,
    )
