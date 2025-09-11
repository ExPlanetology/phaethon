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

from dataclasses import dataclass
import logging
from typing import Union, Optional, List, Type, TypeVar

import numpy as np
import pandas as pd
import scipy.constants as sc
from molmass import Formula

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
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

    gas_species_names: List[str]
    p_bar: pd.Series
    log_p: pd.Series
    moles: pd.Series
    p_total: pd.Series
    molfrac: pd.Series
    massfrac: pd.Series
    elem_molfrac: pd.Series
    mmw: float
    gas_stoich: pd.DataFrame
    mol_masses: pd.Series
    mixing_ratios: pd.Series

    @classmethod
    def new_from_pressure(
        cls: Type[T],
        p_bar: Union[dict, pd.Series],
        temperature: Optional[float] = np.nan,
        volume: Optional[float] = np.nan,
    ) -> None:
        """
        Build gas mixture from partial pressure dict/series.

        Parameters
        ----------
            p_dict : Union[dict, pd.Series]
                Dictionary or Series containing the partial pressures of gas species, in bar.
        """
        p_bar = pd.Series(p_bar).sort_index()
        gas_species_names: List[str] = p_bar.sort_index().index.to_list()
        molfrac = p_bar / np.sum(p_bar)
        gas_stoich = cls._calc_stoich(gas_species_names)
        mol_masses = cls._calc_molmasses(gas_species_names)
        elem_molfrac = cls._calc_elem_molfrac(gas_stoich, p_bar) # for ideal gas, moles ~ p_bar
        mixing_ratios = cls._calc_volume_mixing_ratios(molfrac)
        massfrac = molfrac * mol_masses
        massfrac /= massfrac.sum()

        moles = p_bar * volume / (sc.R * temperature)

        return cls(
            gas_species_names=gas_species_names,
            p_bar=pd.Series(p_bar),
            log_p=np.log10(p_bar),
            p_total=p_bar.sum(),
            moles=moles,
            gas_stoich=gas_stoich,
            mol_masses=mol_masses,
            elem_molfrac=elem_molfrac,
            molfrac=molfrac,
            massfrac=massfrac,
            mmw=np.dot(mol_masses, molfrac),
            mixing_ratios=mixing_ratios,
        )

    @classmethod
    def new_from_moles(
        cls: Type[T],
        moles: Union[dict, pd.Series],
        temperature: Optional[float] = np.nan,
        volume: Optional[float] = np.nan,
    ) -> None:
        """
        Build gas mixture from a dict/series of moles of input gases.

        Parameters
        ----------
            p_dict : Union[dict, pd.Series]
                Dictionary or Series containing the partial pressures of gas species, in bar.
            temperature : float
                Temperature, in K.
            volume : float
                Volume of the gas, in m³.
        """

        moles = pd.Series(moles).sort_index()

        gas_species_names: List[str] = moles.sort_index().index.to_list()

        # evaluate ideal gas law
        p_bar = moles * sc.R * temperature / volume

        molfrac = moles / moles.sum()
        gas_stoich = cls._calc_stoich(gas_species_names)
        mol_masses = cls._calc_molmasses(gas_species_names)
        elem_molfrac = cls._calc_elem_molfrac(gas_stoich, moles)
        mixing_ratios = cls._calc_volume_mixing_ratios(molfrac)
        massfrac = molfrac * mol_masses
        massfrac /= massfrac.sum()

        return cls(
            gas_species_names=gas_species_names,
            p_bar=p_bar,
            log_p=np.log10(p_bar),
            p_total=p_bar.sum(),
            moles=moles,
            gas_stoich=gas_stoich,
            mol_masses=mol_masses,
            elem_molfrac=elem_molfrac,
            molfrac=molfrac,
            massfrac=massfrac,
            mmw=np.dot(mol_masses, molfrac),
            mixing_ratios=mixing_ratios,
        )

    @classmethod
    def new_from_fastchem_inputfile(
        cls: Type[T],
        input_file: str,
        reference_element: str,
        temperature: Optional[float] = np.nan,
        volume: Optional[float] = np.nan,
        total_moles: float = 1.,
    ) -> None:
        """
        Build gas mixture from a FastChem input file.

        Parameters
        ----------
            input_file : str
                Name of the input file.
            reference_element : str
                Reference element for normalization.

        Returns
        -------
            IdealGas
                An instance of the IdealGas class.
        """
        elem_molfrac = {}

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('#') or line.startswith('e-'):
                continue
            parts = line.split()
            if len(parts) == 2:
                elem, log_n = parts
                log_n = float(log_n)
                # Convert log_n back to molar ratio
                xi = 10 ** (log_n - 12.0)
                elem_molfrac[elem] = xi

        # Normalize the molar fractions
        if reference_element in elem_molfrac:
            reference_value = elem_molfrac[reference_element]
            for elem in elem_molfrac:
                elem_molfrac[elem] /= reference_value
        else:
            raise ValueError(f"Reference element {reference_element} not found in the input file.")

        moles = pd.Series(elem_molfrac) * total_moles

        gas_species_names: List[str] = moles.sort_index().index.to_list()

        # evaluate ideal gas law
        p_bar = moles * sc.R * temperature / volume

        molfrac = moles / moles.sum()
        gas_stoich = cls._calc_stoich(gas_species_names)
        mol_masses = cls._calc_molmasses(gas_species_names)
        elem_molfrac = cls._calc_elem_molfrac(gas_stoich, moles)
        mixing_ratios = cls._calc_volume_mixing_ratios(molfrac)
        massfrac = molfrac * mol_masses
        massfrac /= massfrac.sum()

        return cls(
            gas_species_names=gas_species_names,
            p_bar=p_bar,
            log_p=np.log10(p_bar),
            p_total=p_bar.sum(),
            moles=moles,
            gas_stoich=gas_stoich,
            mol_masses=mol_masses,
            elem_molfrac=elem_molfrac,
            molfrac=molfrac,
            massfrac=massfrac,
            mmw=np.dot(mol_masses, molfrac),
            mixing_ratios=mixing_ratios,
        )

    def __add__(self, other):
        """
        Add two GasMixtures together.
        """
        assert isinstance(other, IdealGasMixture)

        # if both were initalised by pressure
        if not self.p_bar.isna().any() and not other.p_bar.isna().any():
            return IdealGasMixture.new_from_pressure(
                self.p_bar.add(other.p_bar, fill_value=0.0)
            )
        if not self.moles.isna().any() and not other.moles.isna().any():
            return IdealGasMixture.new_from_moles(
                self.moles.add(other.moles, fill_value=0.0)
            )
        raise ValueError(
            "Both `GasMixture` objects must be initialized either by pressure or by moles to "
            + "perform addition."
        )

    @staticmethod
    def _calc_stoich(gas_species_names: List[str]) -> pd.DataFrame:
        """
        Calculate the stoichiometric of all gas species, order them in a matrix.

        Parameters
        ----------
            gas_species_names: List[str]
                List of species names, present in gas.
        Returns
        -------
            gas_stoich : pd.DataFrame
                Stoichiometric matrix.
        """
        gas_stoich = pd.DataFrame()

        for species_formula in gas_species_names:
            if species_formula == "e-":
                species_stoich = pd.Series({"e-": 1.0})
            else:
                species_stoich = pd.Series(
                    Formula(species_formula).composition().dataframe().Count.to_dict()
                )
            gas_stoich = pd.concat([gas_stoich, species_stoich], axis=1)

        gas_stoich = gas_stoich.T
        gas_stoich.index = gas_species_names
        gas_stoich = gas_stoich.replace(np.nan, 0.0)
        gas_stoich.sort_index(inplace=True)

        return gas_stoich

    @staticmethod
    def _calc_molmasses(gas_species_names: List[str]) -> pd.Series:
        """
        Calculate the molmasses of all gas species, order them in a pandas Series.

        Parameters
        ----------
            gas_species_names: List[str]
                List of species names, present in gas.
        Returns
        -------
            mol_masses : pd.Series
                Series with molar masses of gas species.
        """
        mol_masses = {}

        for species_formula in gas_species_names:
            if species_formula == "e-":
                mol_masses = mol_masses | {species_formula: 5.48579909e-4}
            else:
                species_molmass = Formula(species_formula).mass
                mol_masses = mol_masses | {species_formula: species_molmass}

        return pd.Series(mol_masses)

    @staticmethod
    def _calc_elem_molfrac(gas_stoich: pd.DataFrame, gas_moles: pd.Series) -> pd.Series:
        """
        Calculate the molar fractions of elements.

        Parameters
        ----------
            gas_stoich : pd.DataFrame
                Stoichiometric matrix.
            gas_moles : pd.Series
                Moles of gas species, in mole.
        Returns
        -------
            elem_molfrac : pd.Series
                Fractions of elements in the gas mixture.
        """
        n_gaselem_dimless = np.dot(gas_stoich.sort_index().T, gas_moles.sort_index())

        elem_molfrac = pd.Series(
            data=n_gaselem_dimless / np.sum(n_gaselem_dimless),
            index=gas_stoich.columns,
        )
        elem_molfrac.sort_index()

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
