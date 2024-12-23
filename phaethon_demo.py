"""
A schematic implementation of how phaethon is supposed to work. Due to the absence of a FOSS
license for MAGMA, we cannot share the modified code used in our Study, Seidler et al. 2024. 
Hence, this code will not work.

If you desire to run the script, please obtain a copy of the MAGMA code and modify it according
to Seidler et al. 2024.
"""
import astropy.units as unit

from phaethon.celestial_objects import (
    CircularOrbitFromPeriod,
    Planet,
    PlanetarySystem,
    Star,
)
from phaethon.outgassing import VapourEngine
from phaethon.gas_mixture import IdealGasMixture
from phaethon.pipeline import PhaethonPipeline

class Melt:
    """
    Dummy class to manage access to MAGMA.
    """

    def __init__(self, buffer: str = "IW") -> None:
        self.buffer = buffer

    def set_chemistry(self, wt: dict = None, mol: dict = None) -> None:
        """
        Dummy function to set chemistry in terms of melt oxides.
        """
        raise NotImplementedError()

    def vaporise(self, temperature: float, dlogfO2: float) -> IdealGasMixture:
        """
        Dummy function to set chemistry in terms of melt oxides.
        """
        raise NotImplementedError()

class ModifiedMAGMA(VapourEngine):
    """ Wrapper to the MAGMA code """

    melt: Melt
    dlogfO2: float
    vapour: IdealGasMixture

    def __init__(
        self,
        buffer: str,
        dlogfO2: float,
        melt_wt_comp: dict = None,
        melt_mol_comp: dict = None,
    ):
        self.melt = Melt(buffer=buffer)
        self.dlogfO2 = dlogfO2

        if melt_mol_comp is not None and melt_wt_comp is None:
            self.melt.set_chemistry(mol=melt_mol_comp)
        elif melt_mol_comp is None and melt_wt_comp is not None:
            self.melt.set_chemistry(wt=melt_wt_comp)
        elif melt_mol_comp is not None and melt_wt_comp is not None:
            raise ValueError(
                "Provide either 'melt_mol_comp' or 'melt_wt_comp', not both."
            )
        else:
            pass

    def get_info(self) -> dict:
        raise NotImplementedError(
            "Absent license to MAGMA, implementation must not be shared."
        )

    def set_extra_params(self, params: dict) -> None:
        raise NotImplementedError(
            "Absent license to MAGMA, implementation must not be shared."
        )

    def equilibriate_vapour(self, temperature: float) -> IdealGasMixture:
        raise NotImplementedError(
            "Absent license to MAGMA, implementation must not be shared."
        )

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
    opac_file_for_lambdagrid="input/SiO_opac_ip_kdistr.h5",
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

BSE = {
    "SiO2": 45,
    "MgO": 37.8,
    "FeO": 8.05,
    "Al2O3": 4.45,
    "CaO": 3.55,
    "Na2O": 0.36,
    "TiO2": 0.201,
}
vapour_engine = ModifiedMAGMA(buffer="IW", dlogfO2=4, melt_wt_comp=BSE)

pipeline = PhaethonPipeline(
    planetary_system=planetary_system,
    vapour_engine=vapour_engine,
    outdir="output/test/",
    opac_species={"SiO"},
    scatterers={},
    opacity_path="input/",
    nlayer=38,
)

# You might need to adept the architecutre to your system.
pipeline.run(cuda_kws={'arch':'sm_86'})
