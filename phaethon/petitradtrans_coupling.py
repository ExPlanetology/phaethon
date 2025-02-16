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
Classes and utilities to conveniently use petitRADTRANS on the atmospheric profiles computed
by HELIOS/phaethon.
"""

from typing import Union, Callable, Optional, List, Tuple, Dict
import numpy as np
from scipy.interpolate import interp1d
from molmass import Formula
from astropy import units
from astropy.units import Quantity

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable

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


class RadtransCoupler:
    """
    Makes using a Phaethon output as input for petitRADTRANS easy.

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

    radtrans: Radtrans
    phaethon_result: PhaethonResult

    line_species: list
    p_profile: np.ndarray
    t_profile: np.ndarray
    massfrac_profiles: dict
    mmw_profile: np.ndarray

    wlen_bords_micron: list
    gas_continuum_contributors: list
    rayleigh_species: list

    star_spec_fit: Callable

    def __init__(
        self,
        line_species: List[str],
        wlen_bords_micron: Tuple[float, float],
        teff_star: float,
        gas_continuum_contributors: Optional[List[str]] = None,
        rayleigh_species: Optional[List[str]] = None,
    ) -> None:

        self.wlen_bords_micron = wlen_bords_micron
        self.line_species = [GasSpeciesNameTranslator(species) for species in line_species]
        if rayleigh_species is None:
            rayleigh_species = []
        self.rayleigh_species = [GasSpeciesNameTranslator(species) for species in rayleigh_species]

        if gas_continuum_contributors is None:
            gas_continuum_contributors = []
        self.gas_continuum_contributors = gas_continuum_contributors

        self.radtrans = None

        # stellar spectrum
        star = PhoenixStarTable()
        stellar_spec, _ = star.compute_spectrum(teff_star)
        wlen_in_cm = stellar_spec[:, 0]
        flux_star_in_hz = stellar_spec[:, 1]
        flux_star_in_cm = flux_star_in_hz * cst.c / (wlen_in_cm**2)
        wlen_in_um = wlen_in_cm * 1e4
        self.star_spec_fit = interp1d(wlen_in_um, flux_star_in_cm)

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
        self.p_profile = self.phaethon_result.pressure.to("bar").value[::-1]
        self.t_profile = self.phaethon_result.temperature.to("K").value[::-1]

        # chemistry profiles, in massfracs (because pRT ...)
        self.massfrac_profiles = {}
        for specimen in self.line_species + self.rayleigh_species:
            self.massfrac_profiles[specimen.formula] = (
                self.phaethon_result.chem[specimen.fastchem_name]
                * specimen.atom_mass
                / self.phaethon_result.chem["m(u)"].to_numpy()[::-1]
            )
        self.mmw_profile = self.phaethon_result.chem["m(u)"].to_numpy()[::-1]

        # init Radtrans object
        self.radtrans = Radtrans(
            pressures=self.p_profile,
            line_species=[specimen.formula for specimen in self.line_species],
            wavelength_boundaries=self.wlen_bords_micron,
            gas_continuum_contributors=self.gas_continuum_contributors,
            rayleigh_species=(
                [specimen.formula for specimen in self.rayleigh_species]
                if len(self.rayleigh_species) > 0
                else None
            ),
            **radtrans_kwargs,
        )

    def calc_planet_flux(
        self, r_planet: float, reference_gravity: float = 981.0
    ) -> Union[np.ndarray, np.ndarray]:
        """
        Flux emitted by the planet.

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
        wavl_cm, planet_flux, _ = self.radtrans.calculate_flux(
            temperatures=self.t_profile,
            mass_fractions=self.massfrac_profiles,
            reference_gravity=reference_gravity,
            mean_molar_masses=self.mmw_profile,
            planet_radius=r_planet * cst.r_earth,
        )
        wavl_micron = wavl_cm * 1e4

        return wavl_micron, planet_flux

    def calc_fpfs(
        self,
        *,
        pl_radius: Optional[Union[float, int, Quantity]] = None,
        st_radius: Optional[Union[float, int, Quantity]] = None,
        grav_acc: Optional[Union[float, int, Quantity]] = None,
        use_phoenix: bool = False,
        **calcflux_kws,
    ) -> Union[np.ndarray, np.ndarray]:
        """
        Secondary occultation depth.

        Parameters
        ----------
            r_planet : float
                Radius of the planet, in R_EARTH
            r_star: float
                Radius of the star, in R_SUN
            grav_acc: float
                Gravitational acceleration at the planets surface (r_planet), in cm/s^2
            use_phoenix: bool
                If true, then a PHOENIX spectrum matching the stellar parameters is downladed and
                used. Otherwise, try to use the stellar spectrum from the HELIOS simualtion.
                Default is False.
            **calcflux_kws
                Keywords passed to petitRADTRANS.radtrans.calculate_flux().
        Returns
        -------
            wavl_micron : np.ndarray
                Wavelengths, in micron.
            fpfs : np.ndarray
                Planet-to-star flux ratio.
        """

        # check if atmosphere has been set
        if self.radtrans is None:
            raise ValueError("Atmosphere is undefined, please run 'set_atmo' first.")

        # planetary radius
        if pl_radius is None:
            _pl_radius_in_cm: float = self.phaethon_result.planet_params[
                "radius"
            ] * units.R_earth.to("cm")
        elif isinstance(pl_radius, Quantity):
            _pl_radius_in_cm: float = pl_radius.to("cm")
        elif isinstance(pl_radius, (int, float)):
            warnings.warn(r"'pl_radius' has no unit, assuming Earth radii")
            _pl_radius_in_cm: float = float(pl_radius) * units.R_earth.to("cm")
        else:
            raise TypeError(r"'pl_radius' must be None, float or an astropy Quantity.")

        # stellar radius
        if st_radius is None:
            _st_radius_in_cm: float = self.phaethon_result.star_params[
                "radius"
            ] * units.R_sun.to("cm")
        elif isinstance(st_radius, Quantity):
            _st_radius_in_cm: float = st_radius.to("cm")
        elif isinstance(st_radius, (int, float)):
            warnings.warn(r"'st_radius' has no unit, assuming Solar radii")
            _st_radius_in_cm: float = float(st_radius) * units.R_sun.to("cm")
        else:
            raise TypeError(r"'st_radius' must be None, float or an astropy Quantity.")

        # get emission spectrum of the planet
        wavl_cm, planet_spectral_emittance, _ = self.radtrans.calculate_flux(
            temperatures=self.t_profile,
            mass_fractions=self.massfrac_profiles,
            reference_gravity=(
                grav_acc
                if grav_acc is not None
                else self.phaethon_result.planet_params["grav"] * 100
            ),
            mean_molar_masses=self.mmw_profile,
            planet_radius=_pl_radius_in_cm,
            star_radius=_st_radius_in_cm,
            **calcflux_kws,
        )
        planet_flux = planet_spectral_emittance * 4.0 * np.pi * (_pl_radius_in_cm) ** 2
        wavl_micron = wavl_cm * 1e4

        # get stellar spectrum
        if use_phoenix:
            stellar_flux = (
                self.star_spec_fit(wavl_micron) * 4.0 * np.pi * (_st_radius_in_cm) ** 2
            )
        else:
            fit_stellar_flux = interp1d(
                self.phaethon_result.star_wavl.to("micron").value,
                self.phaethon_result.spectral_exitance_star,
            )
            stellar_flux = (
                fit_stellar_flux(wavl_micron) * 4.0 * np.pi * (_st_radius_in_cm) ** 2
            )

        # planet-to-star flux ratio (secondary eclipse depth)
        fpfs = planet_flux / stellar_flux

        return wavl_micron, fpfs

    def calc_transm(self, r_planet: float, gravity: float = 981.0, **calctransrad_kws):
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
        wavl, transm_rad, _ = self.radtrans.calculate_transit_radii(
            temperatures=self.t_profile,
            mass_fractions=self.massfrac_profiles,
            mean_molar_masses=self.mmw_profile,
            reference_gravity=gravity,
            planet_radius=r_planet * cst.r_earth,
            reference_pressure=self.p_profile[-1],
            **calctransrad_kws,
        )

        # scale to proper units
        wavl_micron = wavl * 1e4
        transm_rad /= cst.r_earth

        return wavl_micron, transm_rad
