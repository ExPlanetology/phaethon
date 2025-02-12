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
import tempfile
import logging
import os
from enum import Enum
from typing import Literal, Tuple, Union, Optional, Dict
import numpy as np
import pandas as pd

# pylint: disable=c-extension-no-member
import pyfastchem

from phaethon.gas_mixture import IdealGasMixture
from phaethon.save_output import (
    saveChemistryOutput,
    saveCondOutput,
    saveMonitorOutput,
)

logger = logging.getLogger(__name__)

# ================================================================================================
#   CONSTANTS
# ================================================================================================

STANDARD_FASTCHEM_GAS_EQCONST = resources.files("phaethon.data") / "FastChem_logK.dat"
STANDARD_FASTCHEM_COND_EQCONST = (
    resources.files("phaethon.data") / "FastChem_logK_condensates.dat"
)

# ================================================================================================
#   ENUMS
# ================================================================================================

class CondensationMode(Enum):
    """
    Encapsualtes the descrete condensation modes allowed in FastChem
    """

    # The enum values don't mean much except for explanation :)
    NO_COND = "no condensation"
    EQ_COND = "equilibrium condensation"
    RAINOUT = "rain-out of condensates"

# ================================================================================================
#   CLASSES
# ================================================================================================

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
        ref_elem: str = "O",
        cond_mode: CondensationMode = CondensationMode.NO_COND,
    ) -> None:
        """
        Initialize the FastChemCoupler.

        Parameters
        ----------
            path_to_eqconst : Union[str, PathLike (optional)
                Path to the equilibrium constants data file. Defaults to the standard FastChem
                equilibrium constants data file.
            verbosity_level : Literal[0, 1, 2, 3, 4] (optional)
                Verbosity level. Defaults to 0 (silent).
            ref_elem : str (optional)
                Reference element to which the elemental abundances are scaled. Default is 'O'
                (oxygen), because it will be always present for atmospheres in contact with a magma
                ocean.
            cond_mode : str
                Condensation mode, allowed are:
                    "none"
                        --> no condensation
                    "equilibrium"
                        --> equilbrium condensation, i.e. elemental composition is
                            conserved in every atmospheric layer.
                    "rainout":
                        --> if condensates form, they precipitate and remove parts of
                            the elemental abundances. Currently, its use is discouraged,
                            as it does not work properly with the current FastChem <-> HELIOS
                            coupling.
        """
        self.path_to_eqconst = path_to_eqconst
        self.path_to_condconst = path_to_condconst
        self.verbosity_level = verbosity_level
        self.ref_elem = ref_elem
        self.cond_mode = cond_mode

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
        *,
        outdir: str,
        outfile_name: str = "chem.dat",
        monitor: bool = False,
        cond_mode: Optional[CondensationMode] = None,
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

        Returns
        -------
        Optional[str]
            If successful, returns None. Otherwise, returns an error message as a string.
        """

        # pylint: disable=too-many-arguments

        # is 'outdir' ok?
        if not outdir.endswith("/"):
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # pass chemistry to fastchem; places a FastChem input file in outdir; needs a reference
        # element to scale the abundances
        vapour.to_fastchem(
            outfile=outdir + "input_chem.dat", reference_element=self.ref_elem
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

        # condensation mode
        if cond_mode is None:
            cond_mode = self.cond_mode

        if not isinstance(cond_mode, CondensationMode):
                raise TypeError("'cond_mode' must be of type 'phaethon.CondensationMode'!")

        match cond_mode:
            case CondensationMode.NO_COND:
                input_data.equilibrium_condensation = False
                input_data.rainout_condensation = False
            case CondensationMode.EQ_COND:
                input_data.equilibrium_condensation = True
                input_data.rainout_condensation = False
            case CondensationMode.RAINOUT:
                input_data.equilibrium_condensation = False
                input_data.rainout_condensation = True

        # run FastChem on the entire P-T structure
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

    def pointcalc(
        self,
        vapour: Union[Dict[str, float], pd.Series, IdealGasMixture],
        pressure: float,
        temperature: float,
        *,
        verbosity_level: Literal[0, 1, 2, 3, 4] = 0,
        ref_elem: str = "O",
        cond_mode: Optional[CondensationMode] = None,
    ) -> IdealGasMixture:
        """
        Equilibrates a gas composition at a singe pressure and temperature and returns its result
        as an IdealGasMixture.
        """

        # build vapour
        if isinstance(vapour, IdealGasMixture):
            starting_gas: IdealGasMixture = vapour
        else:
            starting_gas: IdealGasMixture = IdealGasMixture(p_bar=vapour)

        # Since FastChem operates with physical in- and output files, we have to generate temporary
        # files for on-the-fly calculations.
        with tempfile.TemporaryDirectory() as tmpdir:

            # pass chemistry to fastchem; places a FastChem input file in outdir; needs a reference
            # element to scale the abundances
            starting_gas.to_fastchem(
                outfile=tmpdir + "input_chem.dat", reference_element=self.ref_elem
            )

            # create a FastChem object
            fastchem = pyfastchem.FastChem(
                tmpdir + "input_chem.dat",
                str(self.path_to_eqconst),
                str(self.path_to_condconst),
                self.verbosity_level,
            )

            # create the input and output structures for FastChem
            input_data = pyfastchem.FastChemInput()
            output_data = pyfastchem.FastChemOutput()

            input_data.temperature = np.array([temperature], dtype=float)
            input_data.pressure = np.array([pressure], dtype=float)

            # condensation mode
            if cond_mode is None:
                cond_mode = self.cond_mode

            if not isinstance(cond_mode, CondensationMode):
                raise TypeError("'cond_mode' must be of type 'phaethon.CondensationMode'!")

            match cond_mode:
                case CondensationMode.NO_COND:
                    input_data.equilibrium_condensation = False
                    input_data.rainout_condensation = False
                case CondensationMode.EQ_COND:
                    input_data.equilibrium_condensation = True
                    input_data.rainout_condensation = False
                case CondensationMode.RAINOUT:
                    input_data.equilibrium_condensation = False
                    input_data.rainout_condensation = True

            # run FastChem on the entire P-T structure
            fastchem.calcDensities(input_data, output_data)

            # save gas speciation file
            saveChemistryOutput(
                tmpdir + "gas_chem.dat",
                np.array(input_data.temperature),
                np.array(input_data.pressure),
                output_data.total_element_density,
                output_data.mean_molecular_weight,
                output_data.number_densities,
                fastchem,
            )

            # # TODO: Maybe also read condensation?
            # # save condensation file
            # saveCondOutput(
            #     tmpdir + "cond_chem.dat",
            #     temperatures,
            #     pressures,
            #     output_data.element_cond_degree,
            #     output_data.number_densities_cond,
            #     fastchem,
            #     output_species=None,
            #     additional_columns=None,
            #     additional_columns_desc=None,
            # )

            # read gas molfractions from FastChem output
            full_frame: pd.DataFrame = pd.read_csv(tmpdir + "gas_chem.dat", sep=r"\s+")
            full_frame.rename(
                columns={
                    r"#p(bar)": r"P(bar)",
                    r"#P(bar)": r"P(bar)",
                    r"T(k)": r"T(K)",
                },
                errors="ignore",
                inplace=True,
            )

            species: List[str] = list(
                full_frame.drop(
                    [r"T(K)", r"P(bar)", r"n_<tot>(cm-3)", r"n_g(cm-3)", r"m(u)"], axis=1
                ).keys()
            )

            gas_frame: pd.DataFrame = full_frame[species]

        return IdealGasMixture(p_bar=gas_frame.iloc[0] * pressure)