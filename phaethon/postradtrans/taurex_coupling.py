#
# Copyright 2025 Fabian L. Seidler
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
Classes and utilities to conveniently use Taurex3 on the atmospheric profiles computed
by HELIOS/phaethon.
"""

from typing import Union, Callable, Optional, List, Tuple, Literal
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from molmass import Formula
from astropy import units
from astropy.units import Quantity

from taurex.cache import OpacityCache, CIACache
from taurex.data.profiles.temperature.temparray import TemperatureArray
from taurex.planet import Planet
from taurex.model import TransmissionModel, EmissionModel, DirectImageModel
from taurex.chemistry import TaurexChemistry, ArrayGas
from taurex.contributions import (
    AbsorptionContribution,
    CIAContribution,
    RayleighContribution,
)
from taurex.stellar import BlackbodyStar
from taurex.binning import FluxBinner, SimpleBinner

from phaethon.analyse import PhaethonResult
from phaethon.utilities import formula_to_hill

# ================================================================================================
#   CLASSES
# ================================================================================================


class GasSpeciesNameTranslator:
    """
    Relates a species to its name in FastChem.

    Parameters
    ----------
        formula : str
            Formula of gas species.
    """

    formula: str
    fastchem_name: str
    atom_mass: str

    def __init__(self, formula: str):
        self.formula = formula
        self.fastchem_name = formula_to_hill(formula)
        __formula = Formula(formula)
        self.atom_mass = __formula.mass


class TaurexCoupler:
    """
    Makes using a Phaethon output as input for Taurex easy.

    Parameters
    ----------
        line_species : List[GasSpeciesNameTranslator
            List of `GasSpeciesNameTranslator` objects. See
            `phaethon.petitradtrans_coupling.GasSpeciesNameTranslator`.
        wlen_bords_micron : Tuple[float, float]
            Wavelengths between which the spectrum is evaluated.
        teff_star : float
            Effective temperature of the host star.
        gas_continuum_contributors : Optional[List[str]]
            List of names of continuum absorbers, e.g. ['H2-H2', 'H2-He']
        rayleigh_species : Optional[List[str]]
            List of gases that act as rayleigh scatterers, e.g. ['H2', 'H2O']
    """

    ## This class must store a lot
    # pylint: disable=too-many-instance-attributes

    phaethon_result: PhaethonResult

    line_species: list
    rayleigh_species: list
    p_profile: np.ndarray
    t_profile: np.ndarray
    mmw_profile: np.ndarray

    def __init__(
        self,
        opac_path: str,
        line_species: List[str],
        rayleigh_species: Optional[List[str]] = None,
    ) -> None:

        self.line_species = [
            GasSpeciesNameTranslator(species) for species in line_species
        ]
        if rayleigh_species is None:
            rayleigh_species = []
        self.rayleigh_species = [
            GasSpeciesNameTranslator(species) for species in rayleigh_species
        ]

        # taurex stuff
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(opac_path)
        # CIACache().set_cia_path("/path/to/cia")

    def set_atmo(
        self,
        phaethon_result: PhaethonResult,
        **radtrans_kwargs,
    ) -> None:
        """
        Set the atmospheric conditions from a PhaethonResult (temperature-pressure structure,
        mean molecular weight, speciation with altitude, etc.).

        Parameters
        ----------
            phaethon_result : PhaethonResult
                Result from a phaethon simulation.
            **radtrans_kwargs:
                Keyword arguments passed to petitRADTRANS.Radtrans() object during initialisation.
        """

        self.phaethon_result = phaethon_result

        # p-T profiles; increasing order (restriction py petitRADTRANS)
        self.p_profile = self.phaethon_result.pressure.to("Pa").value
        self.t_profile = self.phaethon_result.temperature.to("K").value

        # temperature profile
        temperature = TemperatureArray(
            tp_array=self.t_profile,
            p_points=self.p_profile,
        )

        # chemistry profiles; TODO: add Rayleigh and CIA species as well!
        chemistry = TaurexChemistry()
        for specimen in self.line_species:
            mix_ratio_array = self.phaethon_result.chem[
                specimen.fastchem_name
            ].to_list()
            chemistry.addGas(
                ArrayGas(
                    molecule_name=specimen.formula,
                    mix_ratio_array=mix_ratio_array,
                )
            )
        self.mmw_profile = self.phaethon_result.chem["m(u)"].to_numpy()[::-1]

        # planet
        planet = Planet(
            planet_radius=(phaethon_result.planet_params["radius"] * units.R_earth)
            .to("R_jup")
            .value,
            planet_mass=(phaethon_result.planet_params["mass"] * units.M_earth)
            .to("M_jup")
            .value,
        )

        # star
        star = BlackbodyStar(
            temperature=phaethon_result.star_params["t_eff"],
            radius=phaethon_result.star_params["radius"],
        )

        # init Taurex objects
        self._transmission_model = TransmissionModel(
            planet=planet,
            temperature_profile=temperature,
            chemistry=chemistry,
            star=star,
            atm_min_pressure=self.p_profile.min(),
            atm_max_pressure=self.p_profile.max(),
            nlayers=30,
        )

        self._emission_model = EmissionModel(
            planet=planet,
            temperature_profile=temperature,
            chemistry=chemistry,
            star=star,
            atm_min_pressure=self.p_profile.min(),
            atm_max_pressure=self.p_profile.max(),
            nlayers=30,
        )

        self._direct_image_model = DirectImageModel(
            planet=planet,
            temperature_profile=temperature,
            chemistry=chemistry,
            star=star,
            atm_min_pressure=self.p_profile.min(),
            atm_max_pressure=self.p_profile.max(),
            nlayers=30,
        )

        # Add contributions
        for model in [
            self._transmission_model,
            self._emission_model,
            self._direct_image_model,
        ]:
            model.add_contribution(AbsorptionContribution())
            # model.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"]))
            if len(self.rayleigh_species) > 0:
                model.add_contribution(RayleighContribution())
            model.build()

    def __eval_taurex_model(
        self,
        which: Literal["transmission", "emission", "direct-imaging"],
        wavlbounds_in_micron: Tuple[float, float],
        n_samples: int = 1000,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Evaluates a Taurex-Model.
        """
        wngrid = np.sort(
            10000
            / np.logspace(
                np.log10(min(wavlbounds_in_micron)),
                np.log10(max(wavlbounds_in_micron)),
                n_samples,
            )
        )
        bn = SimpleBinner(wngrid=wngrid)

        match which:
            case "transmission":
                _model = self._transmission_model
            case "emission":
                _model = self._emission_model
            case "direct-imaging":
                _model = self._direct_image_model
            case _:
                raise NotImplementedError()

        bin_wn, bin_rprs, _, _ = bn.bin_model(_model.model())

        bin_wavl = 10000 / bin_wn
        bin_wavl *= units.micron

        return bin_wavl, bin_rprs

    def calc_planet_flux(
        self, wavlbounds_in_micron: Tuple[float, float], n_samples: int = 1000
    ) -> Union[np.ndarray, np.ndarray]:
        """
        Calculate the transmission spectrum of a planet.

        Parameters
        ----------
            r_planet : float
                Radius of planet, in R_EARTH
            r_star: float
                Radius of star, in R_SUN
        Returns
        -------
            wavl_micron : np.ndarray
                Wavelengths, in micron.
            flux : np.ndarray
                Flux emitted by the planet.
        """
        return self.__eval_taurex_model(
            which="direct-imaging",
            wavlbounds_in_micron=wavlbounds_in_micron,
            n_samples=n_samples,
        )

    def calc_fpfs(
        self, wavlbounds_in_micron: Tuple[float, float], n_samples: int = 1000
    ):
        """
        Calculate the secondary eclipse depth of a planet.

        Parameters
        ----------
            r_planet : float
                Radius of planet, in R_EARTH.
            gravity : float
                Gravitational acceleration, in cgs.
            **calctransrad_kws:
                Keywords passed to petitRADTRANS.radtrans.calculate_transit_radii().

        Returns
        -------
            wavl_micron : np.ndarray
                Wavelength of spectrum, in micron.
            transm_rad : np.ndarray
                transmission radius, in R_earth
        """

        return self.__eval_taurex_model(
            which="emission",
            wavlbounds_in_micron=wavlbounds_in_micron,
            n_samples=n_samples,
        )

    def calc_transm(
        self, wavlbounds_in_micron: Tuple[float, float], n_samples: int = 1000
    ):
        """
        Calculate the transmission spectrum of a planet.

        Parameters
        ----------
            r_planet : float
                Radius of planet, in R_EARTH.
            gravity : float
                Gravitational acceleration, in cgs.
            **calctransrad_kws:
                Keywords passed to petitRADTRANS.radtrans.calculate_transit_radii().

        Returns
        -------
            wavl_micron : np.ndarray
                Wavelength of spectrum, in micron.
            transm_rad : np.ndarray
                transmission radius, in R_earth
        """

        return self.__eval_taurex_model(
            which="transmission",
            wavlbounds_in_micron=wavlbounds_in_micron,
            n_samples=n_samples,
        )

    def _visualize_mixing_ratios(self):
        """
        Plots the atmospheric mixing rations of active absorbers used by TauREx. Mostly for de-
        bugging purposes.
        """
        plt.figure()

        for x, gasname in enumerate(self._transmission_model.chemistry.activeGases):

            plt.plot(
                self._transmission_model.chemistry.activeGasMixProfile[x],
                self._transmission_model.pressureProfile / 1e5,
                label=gasname,
            )
        for x, gasname in enumerate(self._transmission_model.chemistry.inactiveGases):

            plt.plot(
                self._transmission_model.chemistry.inactiveGasMixProfile[x],
                self._transmission_model.pressureProfile / 1e5,
                label=gasname,
            )
        plt.gca().invert_yaxis()
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.show()
