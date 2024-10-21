"""
This module allows the automatic read-in of phaethon/HELIOS results, and offers some handy tools
for their post-processing.

Copyright 2024 Fabian L. Seidler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass, field
from typing import Union, Literal, Callable, List, Optional
import os
import json
import logging
import warnings

import h5py
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy.constants as sc
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy import units
from astropy.units.core import Unit as AstropyUnit
from astropy.units.quantity import Quantity as AstropyQuantity

from phaethon.fastchem_coupling import FastChemCoupler

logger = logging.getLogger(__name__)

# ================================================================================================
#   CLASSES
# ================================================================================================


@dataclass
class PhaethonResult:
    """
    Stores a Phaethon simulation, and offers some utilities for post-processing.
    """

    ## This class holds a lot of information, therefore:
    # pylint: disable=too-many-instance-attributes

    # basic
    path: os.PathLike
    star_basepath: os.PathLike = ""
    load_star: bool = True

    # metainfo
    planet_params: dict = field(init=False)
    star_params: dict = field(init=False)
    orbit_params: dict = field(init=False)
    vapour_engine_params: dict = field(init=False)

    # atmo structure
    temperature: ArrayLike = field(init=False)
    pressure: ArrayLike = field(init=False)
    altitude: ArrayLike = field(init=False)

    # atmo chemistry
    chem: pd.DataFrame = field(init=False)
    species: List[str] = field(init=False)
    cond: pd.DataFrame = field(init=False)
    condensates: List[str] = field(init=False)
    elem_condfrac: pd.DataFrame = field(init=False)

    # spectra
    wavl: ArrayLike = field(init=False)
    spectral_exitance_planet: ArrayLike = field(init=False)
    fpfs: ArrayLike = field(init=False)
    t_bright: ArrayLike = field(init=False)
    spectral_exitance_star: ArrayLike = field(init=False)
    star_wavl: ArrayLike = field(init=False)
    transmissivity: ArrayLike = field(init=False)
    integrated_transmissivity: ArrayLike = field(init=False)
    optical_depth: ArrayLike = field(init=False)
    _contrib: np.ndarray = field(init=False)
    contribution: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """
        Dataclass inits as empty, the post-init fills these with data from 'path'.
        """
        self.load_phaethon_output(self.path, self.star_basepath, self.load_star)

    def load_phaethon_output(
        self, path: os.PathLike, star_basepath: os.PathLike = "", load_star: bool = True
    ) -> None:
        """
        Load a phaethon result located at 'path'.
        """
        self.path = path
        self.star_basepath = star_basepath
        self.load_star = load_star

        # ------- load spectra ------#
        filename = self.path + r"HELIOS_iterative/TOA_flux_eclipse.dat"
        df = pd.read_csv(filename, skiprows=2, sep=r"\s+")
        self.wavl = df[r"cent_lambda[um]"].to_numpy() * units.micron
        self.spectral_exitance_planet = df[r"F_up_at_TOA"].to_numpy() * (
            units.erg / units.s / (units.cm**3)
        )
        self.fpfs = df[r"planet/star"].to_numpy()  # turn into ppm
        self.t_bright = self.brightness_temp() * units.K

        # ------ load pressure-temperature profile -------#
        filename = self.path + r"HELIOS_iterative/tp.dat"
        df = pd.read_csv(filename, skiprows=1, sep=r"\s+")
        self.temperature = df[r"temp.[K]"].to_numpy() * units.K
        self.pressure = df[r"press.[10^-6bar]"].to_numpy() / 1e6 * units.bar
        self.altitude = df[r"altitude[cm]"].to_numpy() * units.cm

        # ------ load chemistry -----#
        filename = self.path + r"chem_profile.dat"
        self.chem = pd.read_csv(filename, sep=r"\s+")
        self.chem = self.chem.rename(
            columns={
                r"#p(bar)": r"P(bar)",
                r"#P(bar)": r"P(bar)",
                r"T(k)": r"T(K)",
            },
            errors="ignore",
        )

        self.species = list(
            self.chem.drop(
                [r"T(K)", r"P(bar)", r"n_<tot>(cm-3)", r"n_g(cm-3)", r"m(u)"], axis=1
            ).keys()
        )

        # ------------ condensates ----------- #
        filename = self.path + r"cond_chem_profile.dat"
        if os.path.isfile(filename):
            self.cond = pd.read_csv(filename, sep=r"\s+")
            self.cond = self.cond.rename(
                columns={
                    r"#p(bar)": "P(bar)",
                }
            )
            _condensates = list(self.cond.drop([r"T(K)", r"P(bar)"], axis=1).keys())
            self.condensates = []
            _elem_names = []
            for cond in _condensates:
                if (
                    cond.endswith(r"(s,l)")
                    or cond.endswith(r"(s)")
                    or cond.endswith(r"(l)")
                ):
                    self.condensates.append(cond)
                else:
                    _elem_names.append(cond)
            self.elem_condfrac = self.cond[_elem_names]

        # ------------ planet params ------------#
        with open(self.path + r"metadata.json", encoding="utf-8") as f:
            params = json.load(f)

        self.planet_params = params[r"planet"]
        self.star_params = params[r"star"]
        self.orbit_params = params[r"orbit"]
        self.vapour_engine_params = params[r"vapour_engine"]

        # ----------- stellar spectrum ------------#
        self.spectral_exitance_star = None
        self.star_wavl = None
        if load_star:
            with h5py.File(star_basepath + self.star_params[r"path_to_h5"], "r") as f:
                self.spectral_exitance_star = np.array(
                    f[r"r50_kdistr"][r"ascii"][self.star_params[r"name"]]
                ) * (units.erg / units.s / (units.cm**3))
                self.star_wavl = np.array(f[r"r50_kdistr"][r"lambda"]) * units.cm

        # -------------- transmissivity ------------ #
        self.transmissivity = (
            pd.read_table(
                self.path + r"/HELIOS_iterative/transmission.dat",
                skiprows=1,
                sep=r"\s+",
                index_col=0,
            )
            .drop(
                [r"cent_lambda[um]", r"low_int_lambda[um]", r"delta_lambda[um]"], axis=1
            )
            .to_numpy()
        )

        self.integrated_transmissivity = np.flip(
            np.cumprod(np.flip(self.transmissivity), axis=1)
        )

        # -------------- optical depth --------------- #
        self.optical_depth = (
            pd.read_table(
                self.path + r"/HELIOS_iterative/optdepth.dat",
                skiprows=1,
                sep=r"\s+",
                index_col=0,
            )
            .drop(
                [r"cent_lambda[um]", r"low_int_lambda[um]", r"delta_lambda[um]"], axis=1
            )
            .to_numpy()
        )

        # ------------ contribution function -----------#
        self._contrib = (
            pd.read_table(
                self.path + r"/HELIOS_iterative/contribution.dat",
                skiprows=1,
                sep=r"\s+",
                index_col=0,
            )
            .drop(
                [r"cent_lambda[um]", r"low_int_lambda[um]", r"delta_lambda[um]"], axis=1
            )
            .to_numpy()
        )

        self.contribution = self._contrib / self._contrib.sum(axis=1)[:, None]
        self.contribution = np.cumsum(np.flip(self.contribution), axis=1)
        self.contribution = np.flip(self.contribution)

    def get_photospheric_pressurelevel(
        self, photosphere_level: float, smoothing_window_size: Union[int, None] = None
    ) -> ArrayLike:
        """
        Photospheric pressure level as funcition of wavelength, defined by the integrated
        transmissivity, i.e. the pressure where the atmosphere has absorbed a given fraction
        of all incoming light, per wavelength. This fraction is defined in the parameter
        'photosphere_level'.

        Parameters
        ----------
            photosphere_level : float
                Where to define the photosphere, fraction of incoming light that has to be
                absorbed to be defined as photosphere.
            smoothing_window_size : Union[int, None] (optional)
                Width of smooting window. If None, no smoothing is applied. Default is None.
        Returns
        -------
            photosphere : ArrayLike
                Array with pressures that define the photosphere for the given photospheric level.
                Same size as input array, but optionally smoothed.
        """

        # check input
        if photosphere_level < 0.0 or photosphere_level > 1.0:
            raise ValueError(
                f"'photosphere_level' is {photosphere_level}, but must be bounded by 0 and 1."
            )

        # find photosphere
        mask = np.argmin(
            abs(self.integrated_transmissivity - photosphere_level), axis=1
        )
        _photosphere: ArrayLike = self.pressure[1:][mask]

        # apply smoothing (optional)
        if smoothing_window_size is not None:
            return moving_average(_photosphere, smoothing_window_size)
        return _photosphere

    def brightness_temp(self) -> ArrayLike:
        """
        Parameters
        ----------
            wavl : numpy array
                Wavelength, in micron
            flux : numpy array
                Flux from planet, in erg s^-1 cm^-3
        Returns
        -------
            brightness_temp : ArrayLike
                Brightness temperature of the planet, in K.
        """
        wavl = self.wavl.copy().to("micron").value
        flux = self.spectral_exitance_planet.copy().value

        # to SI
        wavl *= 1e-6
        flux *= 1 / np.pi * 0.1

        return (
            sc.h
            * sc.c
            / (sc.k * wavl)
            / (np.log(1 + (2 * sc.h * sc.c**2) / (flux * wavl**5)))
        ) * units.K

    def calc_fpfs(
        self,
        pl_radius: Optional[Union[float, int, AstropyUnit]] = None,
        st_radius: Optional[Union[float, int, AstropyUnit]] = None,
    ) -> ArrayLike:
        """
        Rescales the secondary eclipse depth to a new planetary radius.

        Parameters
        ----------
            pl_radius : AstropyQuantity
                Radius of the planet.
            st_radius : AstropyQuantity
                Radius of the star.
        Returns
        -------
            fpfs : ArrayLike
                Rescaled secondary eclipse depth.
        """

        # planetary radius
        if pl_radius is None:
            _pl_radius_in_m: float = self.star_params["radius"] * units.R_earth.to("m")
        elif isinstance(pl_radius, AstropyQuantity):
            _pl_radius_in_m: float = pl_radius.to("m")
        elif isinstance(pl_radius, (int, float)):
            warnings.warn(r"'pl_radius' has no unit, assuming Earth radii")
            _pl_radius_in_m: float = float(pl_radius) * units.R_earth.to("m")
        else:
            raise TypeError(r"'pl_radius' must be None, float or an astropy Quantity.")

        # stellar radius
        if st_radius is None:
            _st_radius_in_m: float = self.star_params["radius"] * units.R_sun.to("m")
        elif isinstance(st_radius, AstropyQuantity):
            _st_radius_in_m: float = st_radius.to("m")
        elif isinstance(st_radius, (int, float)):
            warnings.warn(r"'st_radius' has no unit, assuming Solar radii")
            _st_radius_in_m: float = float(st_radius) * units.R_sun.to("m")
        else:
            raise TypeError(r"'pl_radius' must be None, float or an astropy Quantity.")

        # calculate new secondary eclipse depth
        fpfs_calculated: ArrayLike = (
            self.spectral_exitance_planet
            / self.spectral_exitance_star
            * (_pl_radius_in_m / _st_radius_in_m) ** 2
        ).value

        return fpfs_calculated

    def run_cond(
        self,
        cond_mode: Literal["none", "equilibrium", "rainout"],
        full_output: bool = False,
    ):
        """
        A postprocessing tool to run condensation along the P-T-profile, bottom-up.

        Parameters
        ----------
            cond_mode: Literal["none", "equilibrium", "rainout"]
                none:
                    No condensation.
                equilibrium:
                    Equilibrium condensation, conserves elemental comp. along column.
                rainout:
                    Removes elements from layer if they condense, runs from bottom
                    to top of the atmosphere.
            full_output: bool
                If True, returns Fastchem and output objects.
        Returns
        -------
            element_cond_degree : pd.DataFrame
                Fraction of a given element that has condensed.
            cond_number_density : pd.DataFrame
                Number densities of condensates [#/m^3]
        """

        class DummyGas:
            """
            Dummy for an empty gas, in order to rerun
            fastchem with the chemistry defined in the phaethon
            output folder. Otherwise, we would have to reload
            the chemistry and put it into a vapour object.
            """

            ## Because this is just a dummy class
            # pylint: disable=too-few-public-methods

            def to_fastchem(self, *args, **kwargs):
                """
                Dummy function, designed to not overwrite the
                fastchem input files already present in the phaethon
                folder.
                """

        fastchem_coupler = FastChemCoupler()
        fastchem, output_data = fastchem_coupler.run_fastchem(
            vapour=DummyGas(),
            pressures=self.pressure.to("bar").value,
            temperatures=self.temperature.to("K").value,
            outdir=self.path,
            cond_mode=cond_mode,
        )

        # element condensation degree, DataFrame
        elem_cond_degree_array: np.ndarray = np.array(output_data.element_cond_degree)
        elem_names = [
            fastchem.getElementSymbol(i) for i in range(elem_cond_degree_array.shape[1])
        ]
        element_cond_degree: pd.DataFrame = pd.DataFrame(
            data=elem_cond_degree_array, columns=elem_names
        )

        # condensates, DataFrame
        numden_cond_arr: np.ndarray = np.array(output_data.number_densities_cond)
        cond_names = [
            fastchem.getCondSpeciesSymbol(i) for i in range(numden_cond_arr.shape[1])
        ]
        cond_number_density: pd.DataFrame = pd.DataFrame(
            data=numden_cond_arr, columns=cond_names
        )

        if full_output:
            return (
                element_cond_degree,
                cond_number_density,
                fastchem,
                output_data,
            )
        return element_cond_degree, cond_number_density


# ================================================================================================
#   UTILITIES
# ================================================================================================


def integrate_flux_in_waveband(
    wavl: np.ndarray,
    spectral_exitance: np.ndarray,
    lambda_lower: AstropyQuantity,
    lambda_upper: AstropyQuantity,
    radius: Union[float, int, AstropyUnit],
    **kwargs,
) -> float:
    """
    Integrates the flux within a waveband.

    Parameters
    ----------
        wavl : np.ndarray
            An array of wavelengths (in units compatible with `lambda_lower` and
            `lambda_upper`) at which the spectral exitance is defined.

        spectral_exitance : np.ndarray
            An array of spectral exitance values corresponding to the wavelengths in
            `wavl`. The shape of this array should match that of `wavl`.

        radius : AstropyQuantity
            The radius of the emitting surface. It should be provided as an `AstropyQuantity`
            with appropriate length units (e.g., meters).

        lambda_lower : AstropyQuantity
            The lower bound of the wavelength range for integration. Should be an
            `AstropyQuantity` in units compatible with the wavelengths in `wavl`.

        lambda_upper : AstropyQuantity
            The upper bound of the wavelength range for integration. Should be an
            `AstropyQuantity` in units compatible with the wavelengths in `wavl`.

        **kwargs : dict, optional
            Additional keyword arguments. Currently present for compatibility reasons only.
    Returns
    -------
        flux : float
            The total integrated flux within the specified wavelength band, taking into
            account the radius of the emitting surface.
    """

    # planetary radius
    if isinstance(radius, AstropyQuantity):
        _radius_in_cm: float = pl_radius.to("cm")
    elif isinstance(radius, (int, float)):
        warnings.warn(r"'radius' has no unit, assuming Earth radii")
        _radius_in_cm: float = float(radius) * units.R_earth.to("cm")
    else:
        raise TypeError(r"'radius' must be None, float or an astropy Quantity.")

    # define units
    si_unit_wavl: str = r"cm"
    si_unit_exitance: str = r"erg / (s cm3)"

    # set correct units
    _wavl: np.ndarray = wavl.to(si_unit_wavl)
    _flux: np.ndarray = (
        spectral_exitance.to(si_unit_exitance) * 4 * np.pi * _radius_in_cm**2
    )

    # interpolate in waveband
    fit: Callable = interp1d(_wavl.value, _flux.value)

    # integrate flux within defined bounds
    integrated_flux, _ = quad(
        fit, lambda_lower.to(si_unit_wavl).value, lambda_upper.to(si_unit_wavl).value
    )

    return integrated_flux * units.erg / (units.s * units.cm**2)


def moving_average(arr: ArrayLike, window_size: int) -> ArrayLike:
    """
    Computes the moving average of the array 'arr'.

    Phaethon uses this utility to smooth the photospheric altitude, particularly for plots,
    as it might vary rapidly with wavelength.
    """
    return np.convolve(arr, np.ones(window_size), "same") / window_size