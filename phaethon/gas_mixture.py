import pandas as pd
import numpy as np
import scipy.constants as sc
from abc import ABC
from typing import Union

from muspell.chemistry import ChemSys, get_compound_mass, get_stoichiometry

class IdealGasMixture():

    def __init__(self, p_bar: Union[dict, pd.Series]):

        self._p_bar: pd.Series = pd.Series()
        self.log_p: pd.Series = pd.Series()
        self.p_total: float = np.nan
        self.molfrac: pd.Series = pd.Series()
        self.elem_molfrac: pd.Series = pd.Series()
        self.mmw: float = np.nan
        self.gas_stoech: pd.DataFrame = pd.DataFrame()
        self.mol_masses: pd.Series = pd.Series()
        self.mixing_ratios: pd.Series = pd.Series()

        self.p_bar: pd.Series = pd.Series(p_bar)

    @property
    def p_bar(self) -> pd.Series:
        return self._p_bar

    @p_bar.setter
    def p_bar(self, p_dict: dict):
        self._p_bar = pd.Series(p_dict)
        self.log_p = np.log10(self._p_bar)
        self.p_total = self._p_bar.sum()
        self.gas_stoech = self._calc_stoech(self._p_bar)
        self.mol_masses = self._calc_molmasses(self._p_bar)
        self.elem_molfrac = self._calc_elem_molfrac(self.gas_stoech, self._p_bar)
        self.molfrac = self.p_bar / np.sum(self.p_bar)
        self.mmw = np.dot(self.mol_masses, self.molfrac)
        self.mixing_ratios = self._calc_volume_mixing_ratios(self.molfrac)

    def __add__(self, other):
        assert(isinstance(other, IdealGasMixture))
        return IdealGasMixture(
            self.p_bar.add(other.p_bar, fill_value=0.)
        )

    @staticmethod
    def _calc_stoech(p_bar: pd.Series) -> pd.DataFrame:
        gas_stoech = pd.DataFrame()

        for i in range(len(p_bar)):
            species_formula = p_bar.index[i]
            species_stoech = pd.Series(
                get_stoichiometry(species_formula)
            )
            gas_stoech = pd.concat([gas_stoech, species_stoech], axis=1)

        gas_stoech = gas_stoech.T
        gas_stoech.index = p_bar.index
        gas_stoech = gas_stoech.replace(np.nan, 0.)

        return gas_stoech

    @staticmethod
    def _calc_molmasses(p_bar: pd.Series) -> pd.Series:
        mol_masses = {}

        for i in range(len(p_bar)):
            species_formula = p_bar.index[i]
            species_molmass = get_compound_mass(species_formula)
            mol_masses = mol_masses | {species_formula:species_molmass}

        return pd.Series(mol_masses)

    @staticmethod
    def _calc_elem_molfrac(gas_stoech: pd.DataFrame, p_bar: pd.Series) -> pd.Series:
        # compute molar fraction
        N_gaselem_dimless = np.dot(gas_stoech.T, p_bar)

        elem_molfrac = pd.Series(
            data=N_gaselem_dimless / np.sum(N_gaselem_dimless),
            index=gas_stoech.columns,
        )

        return elem_molfrac

    @staticmethod
    def _calc_volume_mixing_ratios(molfracs: pd.Series) -> pd.Series:
        mixing_ratios = molfracs.copy()
        for i in range(len(mixing_ratios)):
            mixing_ratios.iloc[i] /= 1. - mixing_ratios.iloc[i]

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

        assert(reference_element in self.elem_molfrac.index)

        #   molar ratio of elements to reference element
        xi = self.elem_molfrac.copy()
        for elem in xi.index:
            if elem == reference_element:
                xi[elem] = 1.
            else:
                xi[elem] = self.elem_molfrac[elem] / self.elem_molfrac[reference_element]

        #   normalize to 10¹² reference element atoms
        logN = np.log10(xi) + 12.

        with open(outfile, 'w') as f:
            f.write('# This is the header\n')
            f.write('e-    0.0\n')
            for elem in self.elem_molfrac.index:
                if elem in logN:
                    value = logN[elem]
                else:
                    value = -np.inf
                if np.isfinite(value):
                    f.write(elem+'    '+str(value)+'\n')
                else:
                    f.write(elem+'    '+str(0.0)+'\n')

    def col_dens(self, grav: float) -> pd.Series:
        """
        Calculates the column densities of the gas species
        """
        return self.p_bar / (self.mol_masses * sc.Avogadro * sc.u * grav )

