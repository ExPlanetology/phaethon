#
# Copyright 2026 Fabian L. Seidler
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
Unit tests for fastchem coupling.
"""
import unittest
from phaethon import IdealGasMixture, FastChemCoupler


class TestFastChemCoupler(unittest.TestCase):
    """
    Collection of test cases for the `FastChemCoupler` module.
    """

    def setUp(self):
        self.gas_comp = IdealGasMixture.new_from_pressure({"O": 2, "Si": 1})
        self.fc = FastChemCoupler(ref_elem="O")

    def test_gas_equilibrium(self):
        """
        Test if a equilibrium gas composition is computed correctly.
        """
        gas, _, _ = self.fc.pointcalc(self.gas_comp, pressure=1, temperature=2500)
        self.assertIsInstance(gas, IdealGasMixture)
        self.assertAlmostEqual(gas.p_total, 1.0, places=5)
        # self.assertTrue(gas.elem_molfrac.equals(pd.Series({"O": 2/3, "Si": 1/3})))

if __name__ == "__main__":
    unittest.main()
