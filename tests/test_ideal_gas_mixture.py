#
# Copyright 2025 Fabian L. Seidler
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
Unit tests for the ideal gas mixture.

The ideal gas mixture is used for convenient conversion between HELIOS and FastChem.
"""
from typing import List
import unittest
import pandas as pd

from phaethon.gas_mixture import IdealGasMixture


class TestIdealGasMixture(unittest.TestCase):
    """
    Collection of test cases for the `IdealGasMixture` class.
    """

    def setUp(self):
        self.p_bar = pd.Series({"H2": 0.5, "O2": 0.5})
        self.moles = pd.Series({"H2": 1.0, "O2": 1.0})

    def test_new_from_pressure(self):
        """
        Test if the initialisation by pressure works.
        """
        gas_mixture = IdealGasMixture.new_from_pressure(self.p_bar)
        self.assertIsInstance(gas_mixture, IdealGasMixture)
        self.assertEqual(gas_mixture.p_total, 1.0)
        self.assertTrue(gas_mixture.molfrac.equals(pd.Series({"H2": 0.5, "O2": 0.5})))

    def test_new_from_moles(self):
        """
        Test if the initialisation by total moles works.
        """
        gas_mixture = IdealGasMixture.new_from_moles(self.moles)
        self.assertIsInstance(gas_mixture, IdealGasMixture)
        self.assertTrue(gas_mixture.molfrac.equals(pd.Series({"H2": 0.5, "O2": 0.5})))

    def test_add(self):
        """
        Test if the addition of gases works.
        """
        gas_mixture1 = IdealGasMixture.new_from_pressure(self.p_bar)
        gas_mixture2 = IdealGasMixture.new_from_pressure(self.p_bar)
        combined_gas_mixture = gas_mixture1 + gas_mixture2
        self.assertIsInstance(combined_gas_mixture, IdealGasMixture)
        self.assertEqual(combined_gas_mixture.p_total, 2.0)
        self.assertTrue(
            combined_gas_mixture.molfrac.equals(pd.Series({"H2": 0.5, "O2": 0.5}))
        )

    def test_calc_stoich(self):
        """
        Test if the stoichiometry of gas species is calculated properly.
        """
        gas_species: List[str] = ["H2", "O2"]
        #pylint: disable = protected-access
        gas_stoich = IdealGasMixture._calc_stoich(gas_species)
        self.assertIsInstance(gas_stoich, pd.DataFrame)
        self.assertEqual(gas_stoich.loc["H2", "H"], 2.0)
        self.assertEqual(gas_stoich.loc["O2", "O"], 2.0)

    def test_calc_molmasses(self):
        """
        Check if the molar masses (g/mol) of each gas species is calculated properly.
        """
        gas_species: List[str] = ["H2", "O2"]
        #pylint: disable = protected-access
        mol_masses = IdealGasMixture._calc_molmasses(gas_species)
        self.assertIsInstance(mol_masses, pd.Series)
        self.assertAlmostEqual(mol_masses["H2"], 2.01588, places=3)
        self.assertAlmostEqual(mol_masses["O2"], 31.9988, places=3)

    def test_calc_elem_molfrac(self):
        """
        Test if molar fraction of elements in bulk gas is computed properly.
        """
        molfrac = pd.Series({"H2": 1.0, "O2": 0.5})
        gas_species = molfrac.index.to_list()
        #pylint: disable = protected-access
        gas_stoich = IdealGasMixture._calc_stoich(gas_species)
        elem_molfrac = IdealGasMixture._calc_elem_molfrac(gas_stoich, molfrac)
        self.assertIsInstance(elem_molfrac, pd.Series)
        self.assertAlmostEqual(elem_molfrac["H"], 0.666666, places=3)
        self.assertAlmostEqual(elem_molfrac["O"], 0.333333, places=3)

    def test_calc_volume_mixing_ratios(self):
        """
        Check if volume mixing ratios are computed properly.
        """
        molfracs = pd.Series({"H2": 0.5, "O2": 0.5})
        #pylint: disable = protected-access
        mixing_ratios = IdealGasMixture._calc_volume_mixing_ratios(molfracs)
        self.assertIsInstance(mixing_ratios, pd.Series)
        self.assertAlmostEqual(mixing_ratios["H2"], 1.0, places=3)
        self.assertAlmostEqual(mixing_ratios["O2"], 1.0, places=3)

    def test_col_dens(self):
        """
        Is the column density computed properly?
        """
        gas_mixture = IdealGasMixture.new_from_pressure(self.p_bar)
        col_dens = gas_mixture.col_dens(grav=9.81)
        self.assertIsInstance(col_dens, pd.Series)
        self.assertTrue((col_dens > 0).all())


if __name__ == "__main__":
    unittest.main()
