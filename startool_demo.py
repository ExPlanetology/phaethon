from phaethon.star import Star

star = Star(
    name="Phoenix",
    mass=1.0,
    radius=1.0,
    t_eff=5770.0,
    distance=10.0,
    metallicity=0.0,
)
# star.get_phoenix_spectrum(
#     outdir="output/stellar_spectra/",
#     opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5",
#     plot_and_tweak=True,
# )
star.get_spectrum_from_file(
    outdir="output/stellar_spectra/",
    source_file="/home/fabian/LavaWorlds/phaethon/phaethon/star_tool/sun_gueymard_2003_modified.txt",
    opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5",
    skiprows=9,
    plot_and_tweak=True,
    w_conversion_factor=1e-7,
    flux_conversion_factor=1e10,
)