import numpy as np
from phaethon.outgassing import VapourEngine, PureMineralVapourMuspell

def test_mineral_atmo():
    surface_exchange = PureMineralVapourMuspell(
        buffer="IW", dlogfO2=0.0, melt_wt_comp={"SiO2": 1.0}
    )
    vap = surface_exchange.equilibriate_vapour(
        surface_pressure=None, surface_temperature=2500.0
    )

    expected_pressures = np.array([-4.03298497, -4.38036993, -7.52843007, -1.48461503, -3.29656   ])
    np.allclose(vap.log_p, expected_pressures)