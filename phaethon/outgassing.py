"""
Module concerned with outgassing.

Copyright 2024 Fabian L. Seidler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from astropy.units import Quantity
from tqdm import tqdm

from phaethon.gas_mixture import IdealGasMixture
from modified_magma import Melt

logger = logging.getLogger(__name__)


# ===============================================================================================
#   CLASSES
# ===============================================================================================


class VapourEngine(ABC):
    """
    Concerned with handling equilibration at atmosphere-magma-ocean interface.
    """

    @abstractmethod
    def get_info(self) -> dict:
        """Information and params on the particular vapour engine"""


    @abstractmethod
    def set_extra_params(self, params: dict) -> None:
        """ Modify params unique to implementation """


    @abstractmethod
    def equilibriate_vapour(self, surface_temperature: Quantity) -> IdealGasMixture:
        """ Equilibrate chemistry at the magma-ocean atmosphere interface """



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

    def equilibriate_vapour(self, surface_temperature: Quantity) -> IdealGasMixture:
        raise NotImplementedError(
            "Absent license to MAGMA, implementation must not be shared."
        )


# ===============================================================================================
#   FUNCTIONS
# ===============================================================================================


def run_temperature_series(
    vapour_engine: VapourEngine, temperature: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtain pressures along a temperature series.

    Parameters
    ----------
        vapour_engine : VapourEngine
            Engine performing the vaporisation
        temperature : np.ndarray
            Temperatures along which to run
    Returns
    -------
        df_log_p : pd.DataFrame
            Partial pressures of gas species.
        df_xelem : pd.DataFrame
            Elemental fractions in atmosphere.
    """

    df_log_p = pd.DataFrame()
    df_xelem = pd.DataFrame()

    for i in tqdm(range(len(temperature))):
        vapour = vapour_engine.equilibriate_vapour(
            surface_temperature=temperature[i]  # * unit.K
        )
        df_log_p[str(round(temperature[i], 2))] = vapour.log_p
        df_xelem[str(round(temperature[i], 2))] = vapour.elem_molfrac

    return df_log_p, df_xelem


def run_series_extra_params(
    vapour_engine: VapourEngine, temperature: Quantity, param_list: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtain pressures along a temperature series

    Parameters
    ----------
        vapour_engine : VapourEngine
            Engine performing the vaporisation
        temperature_series : np.ndarray
            Temperatures along which to run
    Returns
    -------
        df_log_p : pd.DataFrame
            Partial pressures of gas species.
        df_xelem : pd.DataFrame
            Elemental fractions in atmosphere.
    """

    df_log_p = pd.DataFrame()
    df_xelem = pd.DataFrame()

    for i in tqdm(range(len(param_list))):
        param_dict = param_list[i]
        vapour_engine.set_extra_params(param_dict)
        vapour = vapour_engine.equilibriate_vapour(surface_temperature=temperature)
        df_log_p[i] = vapour.log_p
        df_xelem[i] = vapour.elem_molfrac

    return df_log_p, df_xelem
