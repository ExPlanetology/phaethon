from phaethon.outgassing import VapourEngine, PureMineralVapourMuspell

surface_exchange = PureMineralVapourMuspell(
    buffer="IW", dlogfO2=0.0, melt_wt_comp={"SiO2": 1.0}
)
vap = surface_exchange.equilibriate_vapour(
    surface_pressure=None, surface_temperature=2500.0
)