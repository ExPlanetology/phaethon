import numpy as np
import astropy.units as unit
import astropy.constants as const

from phaethon.celestial_objects import (
    Star,
    Planet,
    Orbit,
    CircularOrbitFromPeriod,
    CircularOrbitFromSemiMajorAxis,
    PlanetarySystem
)
from phaethon.outgassing import PureMineralVapourMuspell
from phaethon.fastchem_coupling import FastChemCoupler
from phaethon.pipeline import PhaethonRunner

star = Star(
    name="Sun",
    mass=1.0 * unit.M_sun,
    radius=1.0 * unit.R_sun,
    t_eff=5770.0 * unit.K,
    distance=10.0 * unit.pc,
    metallicity=0.0 * unit.dex,
)
base_path = "/home/fabian/LavaWorlds/phaethon"
star.get_spectrum_from_file(
    outdir="output/stellar_spectra/",
    source_file=f"{base_path}/phaethon/star_tool/sun_gueymard_2003_modified.txt",
    opac_file_for_lambdagrid=(
        f"{base_path}/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5"
    ),
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
    dilution_factor=1.0,
    internal_temperature=0 * unit.K,
)

planetary_system = PlanetarySystem(
    star=star,
    planet=planet,
    orbit=CircularOrbitFromPeriod(period=1*unit.day)
)
planetary_system.set_semimajor_axis_from_pl_temp(t_planet=2500 * unit.K)

vapour_engine = PureMineralVapourMuspell(buffer="IW", dlogfO2=1.5, melt_mol_comp={'SiO2':1.})

fastchem = FastChemCoupler()
# p_grid, t_grid = fastchem.get_grid(pressures=np.logspace(-8, 3, 120), temperatures=np.linspace(500, 6000, 120))
# fastchem.run_fastchem(vapour=vapour, pressures=p_grid, temperatures=t_grid, outdir="output/test", cond_mode="none")

runner = PhaethonRunner(
    planetary_system=planetary_system,
    vapour_engine=vapour_engine,
    outdir="output/test",
    # opac_species={"Si", "SiO", "SiO2", "O", "O2"},
    opac_species={"SiO"},
    scatterers={},
    opacity_path="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/",
)
# runner.info_dump()
# runner._equilibriate_surface(surface_temperature=runner.planetary_system.planet.temperature.value)
# runner._write_atmospecies_file()
# runner._helios_setup(standard_param_file="phaethon/data/standard_lavaplanet_params.dat")
# runner._loop_helios()
# runner._write_helios_output()

runner.run()