#
# Copyright 2024-2026 Fabian L. Seidler
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

from io import StringIO
import sys
from contextlib import contextmanager
from typing import Union, Callable, Optional, List, Tuple, Self
from dataclasses import dataclass
import warnings
import logging

# astropy.units generates the 'no-member' error for the units.
# pylint: disable=no-member
from astropy import units
from astropy.units import Quantity, Unit, UnitConversionError
import astropy.constants as ac
from scipy.interpolate import interp1d
from molmass import Formula
import numpy as np
import numpy.typing as npt

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable

from phaethon.analyse import PhaethonResult
from phaethon.interfaces import PostRadtransProtocol
from phaethon.utilities import formula_to_hill

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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
    atom_mass: float

    @classmethod
    def new(cls, formula: str) -> Self:
        """
        Create a new species that relates a chemical species to names in petitRADTRANS and in
        FastChem.

        Parameters
        ----------
            formula : str
                Formula of gas species.
        """
        return cls(
            formula=formula,
            fastchem_name=formula_to_hill(formula),
            atom_mass=Formula(formula).mass,
        )


# pylint: disable=invalid-name
@contextmanager
def catch_petitRADTRANS_spam():
    """
    Context manager for absorbing warnings/prints raised by petitRADTRANS, which directly
    go to stdout. However, phaethon should redirect them to the logger instead.
    """
    out = StringIO()
    sys.stdout = out
    try:
        yield out.getvalue
    finally:
        sys.stdout = sys.__stdout__


def get_name_of_variable(var) -> Optional[str]:
    """
    Retrieves the name of a variable from `globals`.
    """
    var_metadata = [k for k, v in globals().items() if v is var]
    if len(var_metadata) == 0:
        return None
    name = var_metadata[0]
    return name


def to_astropy_unit(value: float | int | Quantity, target_unit: Unit) -> Quantity:
    """
    Makes sure that a value is a astropy.units.Quantity of correct type.
    """

    param_name: str = get_name_of_variable(value)

    # Correct type (python type)?
    if not isinstance(value, (float, int, Quantity)):
        if param_name is None:
            raise TypeError(
                "Anonymous variable to `to_astropy_unit` must be of type 'float', 'int' or \
                    'astropy.units.Quantity'."
            )
        raise TypeError(
            f"'{param_name}' must be of type 'float', 'int' or 'astropy.units.Quantity'."
        )

    # Correct unit ("type" in astropy.units)?
    if isinstance(value, Quantity):
        if value.unit.physical_type != target_unit.physical_type:
            if param_name is None:
                raise UnitConversionError(
                    f"Anonymous variable to `to_astropy_unit` is not of type \
                        '{target_unit.physical_type}'!"
                )
            raise UnitConversionError(
                f"'{param_name}' is not of type '{target_unit.physical_type}'!"
            )

    #
    if isinstance(value, (float, int)):
        return value * target_unit
    return value


class PetitRadtransCoupler(PostRadtransProtocol):
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
    __atmosphere_is_init: bool

    wlen_bords_micron: list
    gas_continuum_contributors: list
    rayleigh_species: list
    additional_outputs: dict

    star_spec_fit: Callable

    def __init__(
        self,
        line_species: List[str],
        wlen_bords_micron: Tuple[float, float],
        gas_continuum_contributors: Optional[List[str]] = None,
        rayleigh_species: Optional[List[str]] = None,
        **kwargs,
    ) -> None:

        self.wlen_bords_micron = wlen_bords_micron
        self.line_species = [
            GasSpeciesNameTranslator.new(species) for species in line_species
        ]
        if rayleigh_species is None:
            rayleigh_species = []
        self.rayleigh_species = [
            GasSpeciesNameTranslator.new(species) for species in rayleigh_species
        ]

        if gas_continuum_contributors is None:
            gas_continuum_contributors = []
        self.gas_continuum_contributors = gas_continuum_contributors

        self.additional_outputs = {}

        self.__atmosphere_is_init = False
        self.star_spec_fit = None

        # init Radtrans object with EMPTY pressure array
        with warnings.catch_warnings(), catch_petitRADTRANS_spam() as redirect_spam:
            # pRT warns that atmosphere is initialized with one layer at 1 bar, which is ignored
            # because the pressure structure is set later in `set_atmo`.
            warnings.simplefilter("ignore")

            self.radtrans = Radtrans(
                line_species=[specimen.formula for specimen in self.line_species],
                wavelength_boundaries=self.wlen_bords_micron,
                gas_continuum_contributors=self.gas_continuum_contributors,
                rayleigh_species=(
                    [specimen.formula for specimen in self.rayleigh_species]
                    if len(self.rayleigh_species) > 0
                    else None
                ),
                **kwargs,
            )

            # restore captured petitRADTRANS spam as log message
            captured = redirect_spam()
            for line in captured.splitlines():
                logger.info(line)

    def set_phoenix_star(self, t_eff: float | int):
        """
        Set the star & its spectrum from the PHOENIX database.

        NOTE: This function is called by 'set_atmo'. If you have a custom star, you must reset
        it manually after every call to 'set_atmo'.

        Parameters
        ----------
            teff_star : float
                Effective temperature of the host star.
        """
        with catch_petitRADTRANS_spam() as redirect_spam:
            star = PhoenixStarTable()
            stellar_spec, _ = star.compute_spectrum(t_eff)

            # restore captured petitRADTRANS spam as log message
            captured = redirect_spam()
            logger.info(captured)

        wlen_in_cm = stellar_spec[:, 0]
        flux_star_in_hz = stellar_spec[:, 1]
        flux_star_in_cm = flux_star_in_hz * cst.c / (wlen_in_cm**2)
        wlen_in_um = wlen_in_cm * 1e4
        self.star_spec_fit = interp1d(wlen_in_um, flux_star_in_cm)

    def set_atmo(
        self,
        phaethon_result: PhaethonResult,
        **kwargs,
    ) -> None:
        """
        Set the atmospheric conditions from a PhaethonResult (temperature-pressure structure,
        mean molecular weight, speciation with altitude, etc.).

        Parameters
        ----------
            phaethon_result : PhaethonResult
                Result from a phaethon simulation.
        """

        self.phaethon_result = phaethon_result

        # p-T profiles; increasing order (restriction by petitRADTRANS)
        self.p_profile = self.phaethon_result.pressure.to("bar").value[::-1]
        self.t_profile = self.phaethon_result.temperature.to("K").value[::-1]

        # chemistry profiles in massfracs (because pRT ...)
        self.massfrac_profiles = {}
        for specimen in self.line_species + self.rayleigh_species:
            self.massfrac_profiles[specimen.formula] = (
                self.phaethon_result.chem[specimen.fastchem_name]
                * specimen.atom_mass
                / self.phaethon_result.chem["m(u)"].to_numpy()[::-1]
            )
        self.mmw_profile = self.phaethon_result.chem["m(u)"].to_numpy()[::-1]

        # update pressure profile; unfortunately, this means we have to overwrite _pressure,
        # otherwise we are forced to reload line-lists whenever 'set_atmo' is called.
        # pylint: disable=protected-access
        self.radtrans._pressures = self.phaethon_result.pressure.to("dyne / cm2").value[
            ::-1
        ]

        # inform class that atmosphere is set
        self.__atmosphere_is_init = True

        # update star as well
        self.set_phoenix_star(t_eff=self.phaethon_result.star_params["t_eff"])

    def __is_atmosphere_init(self) -> None:
        """
        Checks if an atmosphere has been initialised, otherwise the output of petitRADTRANS
        either makes no sense, or fails with an indescriptive error warning.
        """
        if not self.__atmosphere_is_init:
            raise ValueError(
                "No atmosphere has been loaded. Please run 'set_atmo' first!"
            )

    def calc_planet_flux(
        self,
        *,
        pl_radius: Optional[Union[float, int, Quantity]] = None,
        pl_mass: Optional[Union[float, int, Quantity]] = None,
        **kwargs,
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

        self.__is_atmosphere_init()

        # planet radius
        _planet_radius: Quantity = to_astropy_unit(
            (
                pl_radius
                if pl_radius is not None
                else self.phaethon_result.planet_params["radius"] * units.R_earth
            ),
            target_unit=units.R_earth,
        )

        # planet mass
        _planet_mass: Quantity = to_astropy_unit(
            (
                pl_mass
                if pl_mass is not None
                else self.phaethon_result.planet_params["mass"] * units.M_earth
            ),
            target_unit=units.M_earth,
        )

        # "surface" gravity (i.e., gravitational acceleration at reference pressure)
        _reference_gravity = (ac.G * _planet_mass / (_planet_radius**2)).decompose()

        # planet spectral emittance
        wavl_cm, planet_spectral_emittance, self.additional_outputs = (
            self.radtrans.calculate_flux(
                temperatures=self.t_profile,
                mass_fractions=self.massfrac_profiles,
                mean_molar_masses=self.mmw_profile,
                reference_gravity=_reference_gravity.to("cm / s2").value,
                planet_radius=_planet_radius.to("cm").value,
                **kwargs,
            )
        )

        # spectral emittance (of one cm2 on the surface of the planet) to global emission
        planet_flux = (
            planet_spectral_emittance
            * 4.0
            * np.pi
            * (_planet_radius.to("cm").value) ** 2
        )

        # assign proper units
        wavl = (wavl_cm * units.cm).to("micron")
        planet_flux *= units.erg / (units.cm**2 * units.second * units.cm)

        return wavl, planet_flux

    def calc_fpfs(
        self,
        *,
        pl_radius: Optional[Union[float, int, Quantity]] = None,
        pl_mass: Optional[Union[float, int, Quantity]] = None,
        st_radius: Optional[Union[float, int, Quantity]] = None,
        use_phoenix: bool = False,
        **kwargs,
    ) -> Union[Quantity, np.ndarray]:
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
            **kwargs
                Keywords passed to `self.calc_planet_flux`.
        Returns
        -------
            wavl_micron : np.ndarray
                Wavelengths, in micron.
            fpfs : np.ndarray
                Planet-to-star flux ratio.
        """

        self.__is_atmosphere_init()

        # star radius
        _star_radius: Quantity = to_astropy_unit(
            (
                st_radius
                if st_radius is not None
                else self.phaethon_result.star_params["radius"] * units.R_sun
            ),
            target_unit=units.R_sun,
        )

        # planet emission
        wavl, planet_flux = self.calc_planet_flux(
            pl_radius=pl_radius, pl_mass=pl_mass, **kwargs
        )

        # get stellar spectrum
        if use_phoenix:
            stellar_flux = (
                (
                    self.star_spec_fit(wavl.to("micron").value)
                    * 4.0
                    * np.pi
                    * (_star_radius.to("cm").value) ** 2
                )
                * units.erg
                / (units.cm**2 * units.second * units.cm)
            )
        else:
            fit_stellar_flux = interp1d(
                self.phaethon_result.star_wavl.to("micron").value,
                self.phaethon_result.spectral_exitance_star,
            )
            stellar_flux = (
                (
                    fit_stellar_flux(wavl.to("micron").value)
                    * 4.0
                    * np.pi
                    * (_star_radius.to("cm").value) ** 2
                )
                * units.erg
                / (units.cm**2 * units.second * units.cm)
            )

        # planet-to-star flux ratio (secondary eclipse depth)
        fpfs = (planet_flux / stellar_flux).decompose()

        return wavl, fpfs

    def calc_transm_radius(
        self,
        *,
        pl_radius: Optional[Union[float, int, Quantity]] = None,
        pl_mass: Optional[Union[float, int, Quantity]] = None,
        reference_pressure: Optional[float, int, Quantity] = None,
        **kwargs,
    ) -> [Quantity, Quantity]:
        """
        Transmission radius as function of wavelength.

        Parameters
        ----------
            pl_radius : float
                Radius of planet. If not 'astropy.units.Quantity', assume Earth-radius.
            pl_mass : float
                Mass of planet. If not 'astropy.units.Quantity', assume Earth-mass.
            reference_pressure : float
                Pressure where the planet has radius `pl_radius`. If not 'astropy.units.Quantity',
                assume bar.
            **kwargs:
                Keywords passed to petitRADTRANS.radtrans.calculate_transit_radii().

        Returns
        -------
            wavl_micron : np.ndarray
                Wavelength of spectrum, in micron.
            transm_rad : np.ndarray
                transmission radius, in R_earth
        """

        self.__is_atmosphere_init()

        # reference pressure (where the radius of the planet is computed)
        _ref_pressure: Quantity = to_astropy_unit(
            (
                reference_pressure
                if reference_pressure is not None
                else self.p_profile[-1] * units.bar
            ),
            target_unit=units.bar,
        )

        # planet radius
        _planet_radius: Quantity = to_astropy_unit(
            (
                pl_radius
                if pl_radius is not None
                else self.phaethon_result.planet_params["radius"] * units.R_earth
            ),
            target_unit=units.R_earth,
        )

        # planet mass
        _planet_mass: Quantity = to_astropy_unit(
            (
                pl_mass
                if pl_mass is not None
                else self.phaethon_result.planet_params["mass"] * units.M_earth
            ),
            target_unit=units.M_earth,
        )

        # "surface" gravity (i.e., gravitational acceleration at reference pressure)
        _reference_gravity = (ac.G * _planet_mass / (_planet_radius**2)).decompose()

        # transmission calculation
        wavl_cm, transm_rad_in_cm, self.additional_outputs = (
            self.radtrans.calculate_transit_radii(
                temperatures=self.t_profile,
                mass_fractions=self.massfrac_profiles,
                mean_molar_masses=self.mmw_profile,
                reference_gravity=_reference_gravity.to("cm / s2").value,
                planet_radius=_planet_radius.to("cm").value,
                reference_pressure=_ref_pressure.to("bar").value,
                **kwargs,
            )
        )

        # apply correct units
        wavl: Quantity = (wavl_cm * units.cm).to("micron")
        transm_rad: Quantity = (transm_rad_in_cm * units.cm).to("R_earth")

        return wavl, transm_rad

    def calc_transm_depth(
        self,
        *,
        pl_radius: Optional[Union[float, int, Quantity]] = None,
        pl_mass: Optional[Union[float, int, Quantity]] = None,
        st_radius: Optional[Union[float, int, Quantity]] = None,
        reference_pressure: Optional[float, int, Quantity] = None,
        **kwargs,
    ) -> [Quantity, npt.ArrayLike]:
        r"""
        Transmission depth:

        ..math:
            \delta = \left( \frac{R_p[\lambda]}{R_s} \right)^2

        Effectively, the fractional area covered by the planet during transit, equivalent to the
        dimming.

        Parameters
        ----------
            pl_radius : float
                Radius of planet. If not 'astropy.units.Quantity', assume Earth-radius.
            pl_mass : float
                Mass of planet. If not 'astropy.units.Quantity', assume Earth-mass.
            reference_pressure : float
                Pressure where the planet has radius `pl_radius`. If not 'astropy.units.Quantity',
                assume bar.
            st_radius : float
                Radius of star. If not 'astropy.units.Quantity', assume Sun-radius.
            **kwargs:
                Keywords passed to petitRADTRANS.radtrans.calculate_transit_radii().

        Returns
        -------
            wavl_micron : np.ndarray
                Wavelength of spectrum, in micron.
            transm_rad : np.ndarray
                transmission radius, in R_earth
        """

        self.__is_atmosphere_init()

        # planet transmission radius
        wavl, transm_rad = self.calc_transm_radius(
            pl_radius=pl_radius,
            pl_mass=pl_mass,
            reference_pressure=reference_pressure,
            **kwargs,
        )

        # star radius
        _star_radius: Quantity = to_astropy_unit(
            (
                st_radius
                if st_radius is not None
                else self.phaethon_result.star_params["radius"] * units.R_sun
            ),
            target_unit=units.R_sun,
        )

        # calculate area frac
        transm_area_frac = (transm_rad / _star_radius) ** 2

        # make dimensionless (above quantity is 'earthRad2 / solRad2')
        transm_area_frac = transm_area_frac.decompose()

        return wavl, transm_area_frac
