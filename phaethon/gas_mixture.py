# 
# Copyright 2024-2025 Fabian L. Seidler
#
# This file is part of Phaethon.
#
# Phaethon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or any later version.
#
# Phaethon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Phaethon.  If not, see <https://www.gnu.org/licenses/>.
#
"""
A tool to perform all calculations associated to an ideal gas (e.g. converting pressures to mixing
ratios, determining elemental fractions, producing a FastChem input file).
"""

import logging
from typing import Union

import numpy as np
import pandas as pd
import scipy.constants as sc
from molmass import Formula

logger = logging.getLogger(__name__)


def get_stoichiometry(formula: str) -> dict:
    """
    Splits any chemical formula into its stoichiometry.
    """
    stoichiometry: dict = {}
    i: int = 0
    while i < len(formula):
        if formula[i].isupper():
            element = formula[i]
            i += 1
            count = ""
            while i < len(formula) and formula[i].islower():
                element += formula[i]
                i += 1
            while i < len(formula) and formula[i].isdigit():
                count += formula[i]
                i += 1
            if count == "":
                count = "1"
            stoichiometry[element] = int(count)
        else:
            i += 1
    return stoichiometry


def get_compound_mass(formula: str) -> float:
    """
    Returns the molar mass of a chemical compound, e.g. SiO2

    Returns
    -------
        mass : float
            mass per molecule, in atomic mass units
    """
    stoich: dict = get_stoichiometry(formula)
    mass: float = 0.0
    for elem, stoich_coef in stoich.items():
        mass += stoich_coef * Formula(elem).mass

    return mass


class IdealGasMixture:
    """
    A class to perform all calculations associated to an ideal gas.

    Parameters
    ----------
        p_bar : Union[dict, pd.Series]
            Dictionary or pandas.Series that holds partial pressures of species in an ideal gas
            mixture. Keys/indices are species names, values are partial pressures, in bar.
    """

    # pylint: disable=too-many-instance-attributes

    _p_bar: pd.Series
    log_p: pd.Series
    p_total: pd.Series
    molfrac: pd.Series
    elem_molfrac: pd.Series
    mmw: float
    gas_stoich: pd.DataFrame
    mol_masses: pd.Series
    mixing_ratios: pd.Series

    def __init__(self, p_bar: Union[dict, pd.Series]):

        self._p_bar: pd.Series = pd.Series()
        self.log_p: pd.Series = pd.Series()
        self.p_total: float = np.nan
        self.molfrac: pd.Series = pd.Series()
        self.elem_molfrac: pd.Series = pd.Series()
        self.mmw: float = np.nan
        self.gas_stoich: pd.DataFrame = pd.DataFrame()
        self.mol_masses: pd.Series = pd.Series()
        self.mixing_ratios: pd.Series = pd.Series()

        self.p_bar: pd.Series = pd.Series(p_bar)

    @property
    def p_bar(self) -> pd.Series:
        """Return total pressure of gas mixture"""
        return self._p_bar

    @p_bar.setter
    def p_bar(self, p_dict: dict) -> None:
        """
        Build gas mixture from partial pressure dict.

        Parameters
        ----------
            p_dict : dict
                Dictionary containint the partial pressures of gas species, in bar.
        """
        self._p_bar = pd.Series(p_dict)
        self.log_p = np.log10(self._p_bar)
        self.p_total = self._p_bar.sum()
        self.gas_stoich = self._calc_stoich(self._p_bar)
        self.mol_masses = self._calc_molmasses(self._p_bar)
        self.elem_molfrac = self._calc_elem_molfrac(self.gas_stoich, self._p_bar)
        self.molfrac = self.p_bar / np.sum(self.p_bar)
        self.mmw = np.dot(self.mol_masses, self.molfrac)
        self.mixing_ratios = self._calc_volume_mixing_ratios(self.molfrac)

    def __add__(self, other):
        """
        Add two GasMixtures together.
        """
        assert isinstance(other, IdealGasMixture)
        return IdealGasMixture(self.p_bar.add(other.p_bar, fill_value=0.0))

    @staticmethod
    def _calc_stoich(p_bar: pd.Series) -> pd.DataFrame:
        """
        Calculate the stoichiometric matrix, containing all gas species.

        Parameters
        ----------
            p_bar : pd.Series
                Pressures of gas species, in bar.
        Returns
        -------
            gas_stoich : pd.DataFrame
                Stoichiometric matrix.
        """
        gas_stoich = pd.DataFrame()

        for i in range(len(p_bar)):
            species_formula = p_bar.index[i]
            species_stoich = pd.Series(get_stoichiometry(species_formula))
            gas_stoich = pd.concat([gas_stoich, species_stoich], axis=1)

        gas_stoich = gas_stoich.T
        gas_stoich.index = p_bar.index
        gas_stoich = gas_stoich.replace(np.nan, 0.0)

        return gas_stoich

    @staticmethod
    def _calc_molmasses(p_bar: pd.Series) -> pd.Series:
        """
        Calculate the molmasses of gas species.

        Parameters
        ----------
            p_bar : pd.Series
                Pressures of gas species, in bar.
        Returns
        -------
            mol_masses : pd.Series
                Series with molar masses of gas species.
        """
        mol_masses = {}

        for i in range(len(p_bar)):
            species_formula = p_bar.index[i]
            species_molmass = get_compound_mass(species_formula)
            mol_masses = mol_masses | {species_formula: species_molmass}

        return pd.Series(mol_masses)

    @staticmethod
    def _calc_elem_molfrac(gas_stoich: pd.DataFrame, p_bar: pd.Series) -> pd.Series:
        """
        Calculate the molar fractions of elements.

        Parameters
        ----------
            gas_stoich : pd.DataFrame
                Stoichiometric matrix.
            p_bar : pd.Series
                Pressures of gas species, in bar.
        Returns
        -------
            elem_molfrac : pd.Series
                Fractions of elements in the gas mixture.
        """
        n_gaselem_dimless = np.dot(gas_stoich.T, p_bar)

        elem_molfrac = pd.Series(
            data=n_gaselem_dimless / np.sum(n_gaselem_dimless),
            index=gas_stoich.columns,
        )

        return elem_molfrac

    @staticmethod
    def _calc_volume_mixing_ratios(molfracs: pd.Series) -> pd.Series:
        """
        Calculate the volume mixing ratios of gas species.

        Parameters
        ----------
            molfracs : pd.Series
                Molar fractions of gas species in mixture.
        Returns
        -------
            mixing_ratios : pd.Series
                Volume mixing ratios of gas species.
        """
        mixing_ratios = molfracs.copy()
        for i in range(len(mixing_ratios)):
            mixing_ratios.iloc[i] /= 1.0 - mixing_ratios.iloc[i]

        return mixing_ratios

    def to_fastchem(self, outfile: str, reference_element: str) -> None:
        """
        Write a fastchem input file.

        Parameters
        ----------
            outfile : str
                Name of the output file.
            reference_element : str
                reference element for normalisation
        """

        assert reference_element in self.elem_molfrac.index

        # molar ratio of elements to reference element
        xi = self.elem_molfrac.copy()
        for elem in xi.index:
            if elem == reference_element:
                xi[elem] = 1.0
            else:
                xi[elem] = (
                    self.elem_molfrac[elem] / self.elem_molfrac[reference_element]
                )

        # normalize to 10¹² reference element atoms
        log_n = np.log10(xi) + 12.0

        # write file
        with open(outfile, "w", encoding="utf-8") as f:
            f.write("# This is the header\n")
            f.write("e-    0.0\n")
            for elem in self.elem_molfrac.index:
                if elem in log_n:
                    value = log_n[elem]
                else:
                    value = -np.inf
                if np.isfinite(value):
                    f.write(elem + "    " + str(value) + "\n")
                else:
                    f.write(elem + "    " + str(0.0) + "\n")

    def col_dens(self, grav: float) -> pd.Series:
        """
        Calculates the column densities of the gas species

        Parameters
        ----------
            grav : float
                Gravitational acceleration, in m/s^2.
        Returns
        -------
            col_dens : pd.Series
                Column density of gas species.
        """
        return self.p_bar / (self.mol_masses * sc.Avogadro * sc.u * grav)
