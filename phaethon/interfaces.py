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
Interface types for variable objects, namely outgassing and post-processing radiative transfer
(with e.g. petitRADTRANS).
"""

import logging
from abc import abstractmethod
from typing import Protocol


from phaethon.gas_mixture import IdealGasMixture

logger = logging.getLogger(__name__)


class OutgassingProtocol(Protocol):
    """
    Equilibrates the atmosphere-magma-ocean system, computing the pressure and composition of the
    overlying vapour.
    """

    @abstractmethod
    def get_info(self) -> dict:
        """Information and params on the particular outgassing protocol"""

    @abstractmethod
    def equilibriate(self, temperature: float) -> IdealGasMixture:
        """Equilibrate chemistry at the magma-ocean atmosphere interface"""

class PostRadtransProtocol(Protocol):
    """
    Protocol for radiative transfer calculations after the P-T structure of the atmosphere has been
    equilibrated (e.g., with HELIOS). 
    """

    @abstractmethod
    def calc_transm(self):
        """
        Calculates the transmission spectrum.
        """
