import pyfastchem
from pyfastchem.save_output import (
    saveChemistryOutput,
    saveMonitorOutput,
    saveChemistryOutputPandas,
    saveMonitorOutputPandas,
)

class FastChemCoupler():

    def __init__(self, path_to_eqconst: str) -> None:
        self.path_to_eqconst = path_to_eqconst
    
    def run_grid(mol_elem_frac, outdir="./", monitor=True, verbose=True):
        """
        Gas speciation calculation with FastChem. Generates a "phase diagram"
        of gas speciation over a range of pressures & temperatures.

        Parameters
        ----------
            mol_elem_frac : dict
                dictionary with element name and molar fraction, i.e. {'Si':0.5, 'O':0.5}
            outdir : str
                directory where FastChem drops the output
            monitor : bool (optional)
                wherever FastChem should produce a monitor file (to check for convergence)
            verbose : bool (optional)
                wherever FastChem should comment on its success

        Returns
        -------
            All output is stored in "outdir"
        """

        # --------- Is 'outdir' ok? ------#
        if outdir == "":
            outdir = "./"
        elif outdir[-1] != "/":
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # --------- pass chemistry to fastchem ------------#
        write_fastchem_input(mol_elem_frac, outdir)

        #   read atmospheric P-T-grid
        df = pd.read_csv(
            "phaethon/FastChem/pt_grid.dat",
            names=["P", "T"],
            header=None,
            delim_whitespace=True,
        )
        temperature = df["T"].to_numpy(float)
        pressure = df["P"].to_numpy(float)

        # create a FastChem object
        fastchem = pyfastchem.FastChem(
            outdir + "input_chem.dat", "phaethon/FastChem/logK.dat", 1
        )

        # create the input and output structures for FastChem
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()

        input_data.temperature = temperature
        input_data.pressure = pressure

        # run FastChem on the entire p-T structure
        fastchem_flag = fastchem.calcDensities(input_data, output_data)
        if verbose:
            print("FastChem reports:", pyfastchem.FASTCHEM_MSG[fastchem_flag])

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
            outdir + "chem.dat",
            temperature,
            pressure,
            output_data.total_element_density,
            output_data.mean_molecular_weight,
            output_data.number_densities,
            fastchem,
        )


    def run_along_pt_profile(
        pressure, temperature, outdir="./", monitor=True, verbose=True
    ):
        """
        Gas speciation calculation with FastChem along a specified 
        pressure-temperature path.

        Parameters
        ----------
            mol_elem_frac : dict
                dictionary with element name and molar fraction, i.e. {'Si':0.5, 'O':0.5}
            outdir : str
                directory where FastChem drops the output
            monitor : bool (optional)
                wherever FastChem should produce a monitor file (to check for convergence)
            verbose : bool (optional)
                wherever FastChem should comment on its success

        Returns
        -------
            All output is stored in "outdir"
        """

        # --------- Is 'outdir' ok? ------#
        if outdir == "":
            outdir = "./"
        elif outdir[-1] != "/":
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # --------- run FastChem ------------#
        fastchem = pyfastchem.FastChem(
            outdir + "input_chem.dat", "phaethon/FastChem/logK.dat", 1
        )

        # create the input and output structures for FastChem
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()

        input_data.temperature = temperature
        input_data.pressure = pressure

        # run FastChem on the entire p-T structure
        fastchem_flag = fastchem.calcDensities(input_data, output_data)
        if verbose:
            print("FastChem reports:", pyfastchem.FASTCHEM_MSG[fastchem_flag])

        # save the monitor output to a file
        if monitor:
            saveMonitorOutput(
                outdir + "monitor_profile.dat",
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
            outdir + "chem_profile.dat",
            temperature,
            pressure,
            output_data.total_element_density,
            output_data.mean_molecular_weight,
            output_data.number_densities,
            fastchem,
        )