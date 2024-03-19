import sys

sys.path.append("/home/fabian/LavaWorlds/phaethon_dev/")
from astropy import units as unit

from phaethon.lavaplanet import (
    LavaPlanet,
    Orbit,
    CircularOrbitFromPeriod,
    CircularOrbitFromSemiMajorAxis,
)
from phaethon.star import Star


def test_calc_temperature_from_semimajoraxis():
    planet = LavaPlanet(
        name="55 Cnc e",
        mass=7.99 * unit.M_earth,
        radius=1.88 * unit.R_earth,
        orbit=CircularOrbitFromSemiMajorAxis(semi_major_axis=0.02 * unit.AU),
    )

    sunlike_star = Star(
    name="Sun",
    mass=1.0,
    radius=1.0,
    t_eff=5770.0,
    distance=10.0,
    metallicity=0.0,
    )
    
    sunlike_star.get_spectrum_from_file(
        outdir="output/stellar_spectra/",
        source_file="/home/fabian/LavaWorlds/phaethon/phaethon/star_tool/sun_gueymard_2003_modified.txt",
        opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5",
        skiprows=9,
        plot_and_tweak=False,
        w_conversion_factor=1e-7,
        flux_conversion_factor=1e10,
    )
    
    t_eq = planet.calc_temp(star=sunlike_star, bond_albedo=0.0, f=1.0)
    print("T_eq: ", t_eq)