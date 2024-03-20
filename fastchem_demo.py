import numpy as np

from phaethon.gas_mixture import IdealGasMixture
from phaethon.fastchem_coupling import FastChemCoupler

vap = IdealGasMixture({"SiO":1., "Mg":0.5, "Fe":0.25})

fastchem = FastChemCoupler(verbosity_level=2)
pressure, temperature = fastchem.get_grid(
    pressures=np.logspace(-8, 3, 150),
    temperatures=np.linspace(1000, 6000, 150),
)
fastchem.run_fastchem(
    vapour=vap,
    pressures=pressure,
    temperatures=temperature,
    outdir="output/",
    cond_mode="none",
)