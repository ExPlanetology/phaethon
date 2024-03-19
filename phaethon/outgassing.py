""" Module concerned with outgassing """
from abc import ABC, abstractmethod
from astropy.units.core import Unit as AstropyUnit

from phaethon.gas_mixture import IdealGasMixture
from muspell.melt import Melt


class VapourEngine(ABC):
    @abstractmethod
    def set_extra_params(self, params: dict) -> None:
        ...

    @abstractmethod
    def equilibriate_vapour(
        self, surface_pressure: AstropyUnit, surface_temperature: AstropyUnit
    ) -> IdealGasMixture:
        ...


class PureMineralVapourMuspell(VapourEngine):
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

    def set_extra_params(self, params: dict) -> None:
        """Set some extra parameters"""

        if "melt_wt_comp" in params:
            self.melt.set_chemistry(wt=params.get("melt_wt_comp"))
        elif "melt_mol_comp" in params:
            self.melt.set_chemistry(mol=params.get("melt_mol_comp"))

        if "buffer" in params:
            self.melt.buffer = params.get("buffer")

        if "dlogfO2" in params:
            self.dlogfO2 = params.get("dlogfO2")

    def equilibriate_vapour(
        self, surface_pressure: AstropyUnit, surface_temperature: AstropyUnit
    ) -> IdealGasMixture:
        """
        Calculates the vapour composition & pressure above the planets surface.
        """

        self.vapour = self.melt.vaporise(
            temperature=surface_temperature, dlogfO2=self.dlogfO2
        )
        return self.vapour
