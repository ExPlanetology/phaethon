import os
from typing import Union, Literal, Tuple
import pkg_resources
import numpy as np
import pyfastchem

from phaethon.gas_mixture import IdealGasMixture
from phaethon.save_output import (
    saveChemistryOutput,
    saveMonitorOutput,
    saveChemistryOutputPandas,
    saveMonitorOutputPandas,
    saveCondOutput,
)

STANDARD_FASTCHEM_GAS_EQCONST = pkg_resources.resource_filename(
    __name__, "data/FastChem_logK.dat"
)

STANDARD_FASTCHEM_COND_EQCONST = pkg_resources.resource_filename(
    __name__, "data/FastChem_logK_condensates.dat"
)

class FastChemCoupler:
    def __init__(
        self,
        path_to_eqconst: Union[str, os.PathLike] = STANDARD_FASTCHEM_GAS_EQCONST,
        path_to_condconst: Union[str, os.PathLike] = STANDARD_FASTCHEM_COND_EQCONST,
        verbosity_level: Literal[0, 1, 2, 3, 4] = 2,
    ) -> None:
        """
        Initialize the FastChemCoupler.

        Parameters:
            path_to_eqconst (Union[str, PathLike], optional): Path to the equilibrium constants data file.
                Defaults to the standard FastChem equilibrium constants data file.
            verbosity_level (Literal[0, 1, 2], optional): Verbosity level. Defaults to 1.
        """
        self.path_to_eqconst = path_to_eqconst
        self.path_to_condconst = path_to_condconst
        self.verbosity_level = verbosity_level

    def get_grid(
        self, pressures: np.ndarray, temperatures: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a 2D grid of pressure-temperature pairs from given arrays of pressures and temperatures.

        Parameters:
            pressures (np.ndarray): 1D array containing pressure values.
            temperatures (np.ndarray): 1D array containing temperature values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D arrays:
                - p_grid: Flattened 1D array representing the grid of pressure values.
                - t_grid: Flattened 1D array representing the grid of temperature values.
        """
        p_grid, t_grid = np.meshgrid(pressures, temperatures)
        return p_grid.flatten(), t_grid.flatten()

    def run_fastchem(
        self,
        vapour: IdealGasMixture,
        pressures: np.ndarray,
        temperatures: np.ndarray,
        outdir: str,
        outfile_name: str,
        cond_mode: Literal["none", "equilibrium", "rainout"] = "none",
        monitor: bool = False,
    ):
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
            Whether FastChem should produce a monitor file to check for convergence. Default is False.

        Returns
        -------
        Union[None, str]
            If successful, returns None. Otherwise, returns an error message as a string.
        """

        assert cond_mode in ["none", "equilibrium", "rainout"]

        # --------- Is 'outdir' ok? ------#
        if not outdir.endswith("/"):
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # --------- pass chemistry to fastchem ------------#
        vapour.to_fastchem(
            outfile=outdir + "input_chem.dat", reference_element="O"
        )

        # create a FastChem object
        fastchem = pyfastchem.FastChem(
            outdir + "input_chem.dat",
            self.path_to_eqconst,
            self.path_to_condconst,
            self.verbosity_level,
        )

        # create the input and output structures for FastChem
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()

        input_data.temperature = temperatures
        input_data.pressure = pressures

        if cond_mode == "equilibrium":
            input_data.equilibrium_condensation = True
            input_data.rainout_condensation = False
        elif cond_mode == "rainout":
            input_data.equilibrium_condensation = False
            input_data.rainout_condensation = True
        else:
            input_data.equilibrium_condensation = False
            input_data.rainout_condensation = False

        # run FastChem on the entire p-T structure
        fastchem_flag = fastchem.calcDensities(input_data, output_data)

        # save the monitor output to a file
        if monitor:
            saveMonitorOutput(
                outdir + "monitor.dat",
                temperature,
                pressure,
                output_data.element_conserved,
                output_data.fastchem_flag,
                output_data.nb_chemistry_iterations,
                output_data.total_element_density,
                output_data.mean_molecular_weight,
                fastchem,
            )

        # this would save the output of all species
        saveChemistryOutput(
            outdir + outfile_name,
            temperatures,
            pressures,
            output_data.total_element_density,
            output_data.mean_molecular_weight,
            output_data.number_densities,
            fastchem,
        )

        # if cond_mode is not "none":
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
