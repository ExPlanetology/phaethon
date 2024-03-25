import astropy.units as unit
import matplotlib.pyplot as plt

from phaethon.analyse import PhaethonResult

result = PhaethonResult("output/test")

for pl_rad in [0.5, 1., 1.5, 2.]:
    fpfs_calc = result.calc_fpfs(pl_radius=pl_rad*unit.R_earth)
    plt.plot(result.wavl, fpfs_calc / 1e-6, label=pl_rad)
plt.legend(loc=0)
plt.semilogx()
plt.show()