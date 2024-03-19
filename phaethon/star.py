"""
Module to load and process stellar spectra
"""
from typing import Callable
import os
import numpy as np
import astropy.units as unit
import astropy.constants as const
from scipy.interpolate import interp1d

from star_tool.functions import main_loop as startool_mainloop

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
        self.mass = mass * unit.M_sun
        self.radius = radius * unit.R_sun
        self.t_eff = t_eff * unit.K
        self.distance = distance * unit.pc
        self.metallicity = metallicity * unit.dex

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
