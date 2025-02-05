"""
Module to couple FastChem 3.0 to phaethon.

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

from importlib import resources
import logging
import os
from typing import Literal, Tuple, Union, Optional
import numpy as np

# pylint: disable=c-extension-no-member
import pyfastchem

from phaethon.gas_mixture import IdealGasMixture
from phaethon.save_output import (
    saveChemistryOutput,
    saveCondOutput,
    saveMonitorOutput,
)

logger = logging.getLogger(__name__)

STANDARD_FASTCHEM_GAS_EQCONST = resources.files("phaethon.data") / "FastChem_logK.dat"
STANDARD_FASTCHEM_COND_EQCONST = (
    resources.files("phaethon.data") / "FastChem_logK_condensates.dat"
)


class FastChemCoupler:
    """
    Provides easy and automated access to FastChem. This class has utilities to create the look-up
    chemistry tables for HELIOS, but also to find atmospheric abundaces along the P-T-profile
    after radiative equilibrium has been achieved.
    """

    def __init__(
        self,
        path_to_eqconst: Union[str, os.PathLike] = STANDARD_FASTCHEM_GAS_EQCONST,
        path_to_condconst: Union[str, os.PathLike] = STANDARD_FASTCHEM_COND_EQCONST,
        verbosity_level: Literal[0, 1, 2, 3, 4] = 0,
    ) -> None:
        """
        Initialize the FastChemCoupler.

        Parameters
        ----------
            path_to_eqconst : Union[str, PathLike (optional)
                Path to the equilibrium constants data file. Defaults to the standard FastChem
                equilibrium constants data file.
            verbosity_level : Literal[0, 1, 2] (optional)
                Verbosity level. Defaults to 1.
        """
        self.path_to_eqconst = path_to_eqconst
        self.path_to_condconst = path_to_condconst
        self.verbosity_level = verbosity_level

    def get_grid(
        self, pressures: np.ndarray, temperatures: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a 2D grid of pressure-temperature pairs from given arrays of pressures and
        temperatures.

        Parameters
        ----------
            pressures : np.ndarray
                1D array containing pressure values. Order is arbitrary, as HELIOS - the target
                of the output files - will interpolate.
            temperatures : np.ndarray
                1D array containing temperature values. Order is arbitrary, as HELIOS - the target
                of the output files - will interpolate.

        Returns
        -------
            p_grid : np.ndarray
                Flattened 1D array representing the grid of pressure values.
            t_grid : np.ndarray
                Flattened 1D array representing the grid of temperature values.
        """
        p_grid, t_grid = np.meshgrid(pressures, temperatures)
        return p_grid.flatten(), t_grid.flatten()

    def run_fastchem(
        self,
        vapour: IdealGasMixture,
        pressures: np.ndarray,
        temperatures: np.ndarray,
        outdir: str,
        *,
        outfile_name: str = "chem.dat",
        cond_mode: Optional[Literal["equilibrium", "rainout"]] = None,
        monitor: bool = False,
        ref_elem: str = "O",
    ) -> Tuple[pyfastchem.FastChem, pyfastchem.FastChemOutput]:
        """
        Perform gas speciation calculation using FastChem.

        Parameters
        ----------
        vapour : IdealGasMixture
            Ideal gas mixture object representing the gas composition.
        pressures : numpy.ndarray
            Array of pressure values.
        temperatures : numpy.ndarray
            Array of temperature values.
        outdir : str
            Directory where FastChem saves the output files.
        outfile_name : str, optional
            Name of the output file. Default is "chem.dat".
        monitor : bool, optional
            Whether FastChem should produce a monitor file to check for convergence. Default is
            False.
        ref_elem : str, optional
            Reference element to which the elemental abundances are scaled. Default is 'O'
            (oxygen), because it will be always present for atmospheres in contact with a magma
            ocean.

        Returns
        -------
        Optional[str]
            If successful, returns None. Otherwise, returns an error message as a string.
        """

        # pylint: disable=too-many-arguments

        # --------- Is 'outdir' ok? ------#
        if not outdir.endswith("/"):
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # pass chemistry to fastchem; places a FastChem input file in outdir; needs a reference
        # element to scale the abundances
        vapour.to_fastchem(
            outfile=outdir + "input_chem.dat", reference_element=ref_elem
        )

        # create a FastChem object
        fastchem = pyfastchem.FastChem(
            outdir + "input_chem.dat",
            str(self.path_to_eqconst),
            str(self.path_to_condconst),
            self.verbosity_level,
        )

        # create the input and output structures for FastChem
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()

        input_data.temperature = temperatures
        input_data.pressure = pressures

        if cond_mode is None:
            input_data.equilibrium_condensation = False
            input_data.rainout_condensation = False
        elif cond_mode == "equilibrium":
            print("    cond_mode: equilibrium")
            input_data.equilibrium_condensation = True
            input_data.rainout_condensation = False
        elif cond_mode == "rainout":
            print("    cond_mode: rainout")
            input_data.equilibrium_condensation = False
            input_data.rainout_condensation = True
        else:
            raise NotImplementedError(
                f"cond_mode '{cond_mode}' is not implemented. Valid options are None, "
                + "'equilibrium' and 'rainout'."
            )

        # run FastChem on the entire p-T structure
        fastchem.calcDensities(input_data, output_data)

        # save the monitor output to a file
        if monitor:
            saveMonitorOutput(
                outdir + "monitor.dat",
                temperatures,
                pressures,
                output_data.element_conserved,
                output_data.fastchem_flag,
                output_data.nb_chemistry_iterations,
                output_data.total_element_density,
                output_data.mean_molecular_weight,
                fastchem,
            )

        # save gas speciation file
        saveChemistryOutput(
            outdir + outfile_name,
            temperatures,
            pressures,
            output_data.total_element_density,
            output_data.mean_molecular_weight,
            output_data.number_densities,
            fastchem,
        )

        # save condensation file
        saveCondOutput(
            outdir + "cond_" + outfile_name,
            temperatures,
            pressures,
            output_data.element_cond_degree,
            output_data.number_densities_cond,
            fastchem,
            output_species=None,
            additional_columns=None,
            additional_columns_desc=None,
        )

        return fastchem, output_data
