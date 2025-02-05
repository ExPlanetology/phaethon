"""
A schematic implementation of how phaethon is supposed to work. Due to the absence of a FOSS
license for MAGMA, we cannot share the modified code used in our Study, Seidler et al. 2024. 
Hence, this code will not work.

If you desire to run the script, please obtain a copy of the MAGMA code and modify it according
to Seidler et al. 2024.
"""
import logging
import pandas as pd
from typing import Callable, Dict
from scipy.interpolate import interp1d
import astropy.units as unit

from phaethon import debug_file_logger
from phaethon.celestial_objects import (
    CircularOrbitFromPeriod,
    Planet,
    PlanetarySystem,
    Star,
)
from phaethon.outgassing import VapourEngine
from phaethon.gas_mixture import IdealGasMixture
from phaethon.pipeline import PhaethonPipeline
from phaethon.fastchem_coupling import FastChemCoupler

logger = debug_file_logger()

OPACITY_PATH: str = "/home/fabian/lavaworlds/opacities/lavaplanets/R200_0.1_200_pressurebroad/"

class VapourEngineExample(VapourEngine):
    """ A simple example to an outgassing routine. Only valid for a fixed oxygen fugacity, but
    allows for variable temperature (as it must, because the oxidation state of the melt is usually
    held constant during a simualtion while the temperature of the atmosphere-melt interface
    changes).
    """

    _df: pd.DataFrame
    """ Frame holding the log of partial pressures of species as function of temperature. """

    _logp_fits: Dict[str, Callable]
    """ Dicitonary holding `scipy.interpolate.interp1d` instances that relate temperature <-> 
    outgassing pressure. """

    vapour: IdealGasMixture
    """ Object holding the properties of the vapour. """

    def __init__(self) -> None:
        self._df = pd.read_csv("input/logP_TERRA_dIW-4.csv", index_col=0)
        
        self._logp_fits = {}
        temp_arr = self._df.columns.to_numpy(dtype=float)
        for species in self._df.index:
            self._logp_fits[species] = interp1d(temp_arr, self._df.loc[species].to_numpy())

    def get_info(self) -> dict:
        """ Returns information on state of the outgassing routine. """
        return {
            "dIW": -4.0,
            "composition_name": "TERRA"
        }

    def set_extra_params(self, params: dict) -> None:
        """ No extra parameters to set """
        pass

    def equilibriate_vapour(self, temperature: float) -> IdealGasMixture:
        """
        Reports the vapour composition as function of temperature.

        Params
        ------
            temperature : float
                Temperature of the melt.
        
        Returns
        -------
            vapour : IdealGasMixture
                Object holding the properties of an ideal gas mixture.
        """
        pbar = pd.Series()
        
        # evaluate species pressure
        for species, logp_fit in self._logp_fits.items():
            pbar[species] = 10**float(logp_fit(temperature))


        # store as vapour
        self.vapour = IdealGasMixture(p_bar=pbar)

        return self.vapour

if __name__=="__main__":
    star = Star(
        name="Sun",
        mass=1.0 * unit.M_sun,
        radius=1.0 * unit.R_sun,
        t_eff=5770.0 * unit.K,
        distance=10.0 * unit.pc,
        metallicity=0.0 * unit.dex,
    )
    star.get_spectrum_from_file(
        outdir="output/stellar_spectra/",
        source_file=f"input/sun_gueymard_2003_modified.txt",
        opac_file_for_lambdagrid=OPACITY_PATH + "SiO_opac_ip_kdistr.h5",
        skiprows=9,
        plot_and_tweak=False,
        w_conversion_factor=1e-7,
        flux_conversion_factor=1e10,
    )

    planet = Planet(
        name="55 Cnc e",
        mass=1.0 * unit.M_earth,
        radius=1.0 * unit.R_earth,
        bond_albedo=0.0,
        dilution_factor=2.0 / 3.0,
        internal_temperature=0 * unit.K,
    )

    planetary_system = PlanetarySystem(
        star=star, planet=planet, orbit=CircularOrbitFromPeriod(period=1 * unit.day)
    )
    planetary_system.set_semimajor_axis_from_pl_temp(t_planet=2500 * unit.K)

    pipeline = PhaethonPipeline(
        planetary_system=planetary_system,
        vapour_engine=VapourEngineExample(),
        fastchem_coupler=FastChemCoupler(),
        outdir="output/test/",
        # opac_species={"SiO", "MgO", "Mg", "Fe", }, # run two iterations
        opac_species={"SiO"},
        scatterers={},
        opacity_path=OPACITY_PATH,
        nlayer=38,
    )

    # You might need to adept the architecutre to your system.
    pipeline.run(cuda_kws={'arch':'sm_86'}, t_abstol=35)
