import sys
sys.path.append("/home/fabian/LavaWorlds/phaethon_dev/")

from phaethon.star import Star

def test_star_phoenix():

    star = Star(
        name="Phoenix",
        mass=1.,
        radius=1.,
        t_eff=5770.,
        distance=10.,
        metallicity=0.0,
    )
    star.get_phoenix_spectrum(
        outdir="./",
        opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5"
    )