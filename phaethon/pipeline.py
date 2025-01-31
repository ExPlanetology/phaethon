"""
The main pipeline that streamlines the computation of the structure of an outgassed atmosphere.

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

import importlib
import json
import logging
import os
import time
from typing import Tuple, Optional, Literal, List
from io import TextIOWrapper
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import bayes_opt

# ----- HELIOS imports -----#
from helios import additional_heating as add_heat
from helios import clouds
from helios import computation as comp
from helios import host_functions as hsfunc
from helios import quantities as quant
from helios import read
from helios import realtime_plotting as rt_plot
from helios import write

# ----- PHAETHON imports -----#
from phaethon.celestial_objects import PlanetarySystem
from phaethon.fastchem_coupling import FastChemCoupler
from phaethon.gas_mixture import IdealGasMixture
from phaethon.outgassing import VapourEngine
from phaethon.logger import file_logger


# ================================================================================================
#   CONSTANTS
# ================================================================================================

DEFAULT_PARAM_FILE = (
    importlib.resources.files("phaethon.data") / "standard_lavaplanet_params.dat"
)


def closest_smaller_value(x: float, arr: ArrayLike):
    # Filter out values greater than or equal to x
    smaller_values = [value for value in arr if value < x]
    
    # If there are no values smaller than x, return minimum of array
    if not smaller_values:
        return min(arr)
    
    # Return the maximum of the smaller values (which will be the closest)
    return max(smaller_values)

def closest_larger_value(x: float, arr: ArrayLike):
    # Filter out values greater than or equal to x
    smaller_values = [value for value in arr if value > x]
    
    # If there are no values smaller than x, return minimum of array
    if not smaller_values:
        return max(arr)
    
    # Return the maximum of the smaller values (which will be the closest)
    return min(smaller_values)

# ================================================================================================
#   CLASSES
# ================================================================================================


class PhaethonPipeline:
    """
    Combines and stores all modules for a self-consistent simulation of an outgassed atmosphere.
    """

    planetary_system: PlanetarySystem
    vapour_engine: VapourEngine
    fastchem_coupler: FastChemCoupler

    opacity_path: str
    opac_species: set
    scatterers: set
    atmo: IdealGasMixture

    p_toa: float
    p_boa: float
    t_boa: float
    t_melt: float

    # HELIOS objects
    _reader: read.Read
    _keeper: quant.Store
    _computer: comp.Compute
    _writer: write.Write
    _plotter: rt_plot.Plot
    _fogger: clouds.Cloud

    def __init__(
        self,
        planetary_system: PlanetarySystem,
        vapour_engine: VapourEngine,
        outdir: str,
        opac_species: set,
        scatterers: set,
        opacity_path: str,
        path_to_eqconst: Optional[str] = None,
        p_toa: float = 1e-8,
        p_grid_fastchem: np.ndarray = np.logspace(-8, 3, 100),
        t_grid_fastchem: np.ndarray = np.linspace(500, 6000, 100),
        nlayer: int = 50,
    ) -> None:
        """
        Init the phaethon pipeline.

        Parameters
        ----------
        planetary_system : PlanetarySystem
            The planetary system to be modeled.
        vapour_engine : VapourEngine
            The engine responsible for handling vaporization/outgassing processes.
        outdir : str
            The directory where output files will be stored.
        opac_species : set
            A set of species to be considered for opacity calculations.
        scatterers : set
            A set of scatterer species included in the model.
        opacity_path : str
            Path to the opacity data files.
        path_to_eqconst : Optional[str], optional
            Path to the equilibrium constant data files. Default is None.
        p_toa : float, optional
            Total atmospheric pressure at the top of the atmosphere (in bar). Default is 1e-8.
        p_grid_fastchem : np.ndarray, optional
            Pressure grid for fast chemistry calculations. Default is a logarithmic space from 1e-8
            to 1e3.
        t_grid_fastchem : np.ndarray, optional
            Temperature grid for fast chemistry calculations. Default is a linear space from 500 to
            6000 K.
        nlayer : int, optional
            Number of layers in the model. Default is 50.
        """

        if not outdir.endswith("/"):
            outdir += "/"
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

        self.planetary_system = planetary_system
        self.vapour_engine = vapour_engine
        self.fastchem_coupler = (
            FastChemCoupler(path_to_eqconst=path_to_eqconst)
            if path_to_eqconst is not None
            else FastChemCoupler()
        )
        self.opacity_path = opacity_path
        self.opac_species = opac_species
        self.scatterers = scatterers
        self.atmo = IdealGasMixture({})
        self.p_toa = p_toa
        self.p_boa = None
        self.t_boa = self.planetary_system.planet.temperature.value
        self.t_melt = None
        self._p_grid_fastchem, self._t_grid_fastchem = self.fastchem_coupler.get_grid(
            pressures=p_grid_fastchem, temperatures=t_grid_fastchem
        )

        # part of advanced options
        self.nlayer = nlayer  # No. of atmospheric layers

        # HELIOS objects
        self._reader = None
        self._keeper = None
        self._computer = None
        self._writer = None
        self._plotter = None
        self._fogger = None

    def info_dump(self) -> None:
        """
        Dumps all info into a JSON-file in `self.outdir`.
        """

        metadata = self.planetary_system.get_info()

        # info on outgassing routine
        vapour_engine_dict = {
            "vapour_engine": {
                "type": str(type(self.vapour_engine)),
                "report": self.vapour_engine.get_info(),
            }
        }
        metadata.update(vapour_engine_dict)

        # info on atmospheric parameters
        atmo_dict = {
            "atmosphere": {
                "opac_species": list(self.opac_species),
                "scatterers": list(self.scatterers),
                "p_bar": self.atmo.p_total,
                "log_p": (
                    np.log10(self.atmo.p_total) if not self.atmo.log_p.empty else -1e90
                ),
                "t_boa:": self.t_boa,
                "elem_molfrac": (
                    self.atmo.elem_molfrac.to_dict()
                    if not self.atmo.log_p.empty
                    else {}
                ),
            }
        }
        metadata.update(atmo_dict)

        # write json file
        with open(
            self.outdir + "metadata.json", "w", encoding="utf-8"
        ) as metadata_file:
            json.dump(metadata, metadata_file)

    def run(
        self,
        t_abstol: float = 35.0,
        param_file: str = DEFAULT_PARAM_FILE,  # TODO: implement correct path type in type hint!
        cond_mode: Optional[Literal["equilibrium", "rainout"]] = None,
        cuda_kws: Optional[dict] = None,
        logfile_name: str = "phaethon.log"
    ) -> None:
        """
        Equilibrates an atmosphere with the underlying (magma) ocean and solves the
        radiative transfer problem.

        Parameters
        ----------
            t_abstol : float
                ΔT allowed between t_melt and t_boa, in K.
            param_file : str
                File containing the HELIOS parameters.
            cond_mode : Optional[Literal['equilibrium', 'rainout']]
                Condensation mode, allowed values:
                    None: no condensation
                    "equilibrium": equilibrium condensation
                    "rainout": if condensates form, they precipitate
            cuda_kws : dict
                Dictionary containing additional args (e.g., compiler flags) for nvcc, the CUDA
                compiler. Called when compiling the HELIOS kernels on-the-fly.
        """

        # init logger
        logger = file_logger(logfile=self.outdir + logfile_name)

        # Write an initial metadata file
        self.info_dump()

        # Generate the file which lists opacity & scattering species and store it in output dir.
        # Read by HELIOS
        self._write_opacspecies_file()

        # Initialise HELIOS. If necessary, pass additional compiler flags to nvcc (CUDA).
        if cuda_kws is None:
            cuda_kws = {}
        self._helios_setup(param_file=param_file, cuda_kws=cuda_kws)

        # Inital temperature of the melt, based on the irradiation temperature
        self.planetary_system.calc_pl_temp()
        self.t_melt = self.planetary_system.planet.temperature.value

        # run the loop
        start: float = time.time()
        iteration_counter: int = 1

        # main loop
        logger.info(f"Starting temperature (melt): {self.t_melt} K")
        self.tmelt_trace: List[float] = []
        self.delta_tmelt_trace: List[float] = []
        optimizer: Optional[bayes_opt.BayesianOptimization] = None
        while True:

            # log track of iteration
            logger.info(f"Entering iteration No. {iteration_counter}")

            # run full radiative transfer forward model
            delta_t_melt = self._single_forward_iteration(
                t_melt=self.t_melt, cond_mode=cond_mode
            )

            # append info for bayesian optimizer (and for convergence plots)
            self.tmelt_trace.append(self.t_melt)
            self.delta_tmelt_trace.append(delta_t_melt)

            # log temperature data
            logger.info(f"t_melt: {round(self.t_melt, 2)} K")
            logger.info(f"t_boa: {round(self.t_boa, 2)} K")
            logger.info(f"-> ΔT: {round(delta_t_melt, 2)} K")

            # Check convergence conditions
            if delta_t_melt <= t_abstol:
                break

            # Prepare for the next iteration
            iteration_counter += 1

            if optimizer is None:
                if self.t_boa < min(self.tmelt_trace) or self.t_boa > max(self.tmelt_trace):
                    self.t_melt = self.t_boa
                else:
                    # _search_range: Tuple[float] = (
                    #     max(1., closest_smaller_value(x=self.t_boa, arr=self.tmelt_trace) - 3*t_abstol),
                    #     closest_larger_value(x=self.t_boa, arr=self.tmelt_trace) + 3 * t_abstol,
                    # )
                    _search_range: Tuple[float] = (
                        min(self.tmelt_trace) - 1.5 * t_abstol,
                        max(self.tmelt_trace) + 1.5 * t_abstol,
                    )

                    logger.info(f"Initialising Bayesian optimizer with range {_search_range}")

                    optimizer = bayes_opt.BayesianOptimization(
                        f=None,
                        pbounds={"t_melt":_search_range},
                        # bounds_transformer = SequentialDomainReductionTransformer(minimum_window=t_abstol * 2)
                        acquisition_function=bayes_opt.acquisition.ExpectedImprovement(xi=0.0)
                    )
                    
                    # warm start for the optimizer by passing it the previous points
                    for _tmelt, _delta_t in zip(self.tmelt_trace, self.delta_tmelt_trace):
                        optimizer.register(
                            params={'t_melt':_tmelt},
                            target=-_delta_t, # negative because the optimizer maximises f
                        )

                    # suggest the next t_melt
                    next_point = optimizer.suggest()
                    logger.info(f"   suggested new point: {next_point}")
                    self.t_melt = next_point["t_melt"]
            else:
                optimizer.register(
                    params={'t_melt':self.t_melt},
                    target=-delta_t_melt,   # negative because the optimizer maximises f
                )
                next_point = optimizer.suggest()
                logger.info(f"   suggested new point: {next_point}")
                self.t_melt = next_point["t_melt"]


        # store output and metadata
        self._write_helios_output()
        self.info_dump()

        # final fastchem run, generate atmospheric abunance profile
        self._final_fastchem_run(cond_mode=cond_mode)

        end: float = time.time()
        logger.info(f"Finished. Duration: {(end - start) / 60.} min")

    # ============================================================================================
    # SEMI-PRIVATE METHODS
    # ============================================================================================

    def _final_fastchem_run(self, cond_mode: str) -> None:
        """
        Performs a final FastChem run along the computed P-T-profile in order to obtain the
        atmospheric speciations as function of pressure/altitude. Allows for using various
        condensation modes.

        Parameters
        ----------
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
        df = pd.read_csv(self.outdir + "HELIOS_iterative/tp.dat", header=1, sep=r"\s+")
        self.fastchem_coupler.run_fastchem(
            vapour=self.atmo,
            pressures=df["press.[10^-6bar]"].to_numpy(float) * 1e-6,
            temperatures=df["temp.[K]"].to_numpy(float),
            outdir=self.outdir,
            outfile_name="chem_profile.dat",  # DO NOT CHANGE - PhaethonResult searches it!
            cond_mode=cond_mode,
        )

    def _single_forward_iteration(
        self,
        t_melt: float,
        cond_mode: Optional[Literal["equilibrium", "rainout"]] = None,
    ) -> float:
        """
        Performs a single forward iteration: vapour -> fastchem -> helios

        Parameters
        ----------
            t_melt : float
                Melt temperature, in K.
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
        Returns
        -------
            delta_t : float
                ΔT := |T_melt - T_boa|, difference between resulting melt temperature and
                bottom-of-atmosphere (BOA) temperature.
        """

        # Vapour composition & pressure
        self._equilibriate_atmo_and_ocean(temperature=t_melt)
        self.p_boa = self.atmo.p_total  # bar
        if self.atmo.log_p.empty:
            raise ValueError("'VapourEngine' returned an empty result!")
        if self.p_toa > self.atmo.p_total:
            raise ValueError(
                f"The vapour pressure ({self.atmo.p_total} bar) is smaller than the TOA pressure"
                + f" ({self.p_toa} bar)!"
            )
        self._keeper.p_boa = float(self.atmo.p_total) / 1e-6  # weird HELIOS scaling

        # FastChem, generates look-up table for HELIOS in `outdir`. Direclty deposits files there.
        self.fastchem_coupler.run_fastchem(
            vapour=self.atmo,
            pressures=self._p_grid_fastchem,
            temperatures=self._t_grid_fastchem,
            outdir=self.outdir,
            outfile_name="chem.dat",  # DO NOT CHANGE - HELIOS searches for "chem.dat"
            cond_mode=cond_mode,
        )

        # solve radiative transfer
        self._reader.read_species_mixing_ratios(self._keeper)
        self._solve_rad_trans()
        self.t_boa = self._keeper.T_lay[self._keeper.nlayer]

        # convergence criterion
        delta_t = abs(self.t_boa - self.t_melt)

        return delta_t

    def _write_opacspecies_file(self):
        """
        Generates atmospheric species files. Necessary for HELIOS to know which species to include,
        which are scatterers, and where to find their abundances (in FastChem output files).
        """

        outfile = self.outdir + "/species_iterative.dat"

        with open(outfile, "w", encoding="utf-8") as species_dat:
            species_dat.write("species     absorbing  scattering  mixing_ratio\n\n")

            for species in self.opac_species.union(self.scatterers):
                outstr = species + " " * (13 - len(species))

                # species does absorb
                if species in self.opac_species:
                    outstr += "yes" + " " * 9
                else:
                    outstr += "no" + " " * 10

                if species in self.scatterers:
                    outstr += "yes" + " " * 9
                else:
                    outstr += "no" + " " * 10

                outstr += "FastChem"
                outstr += "\n\n"

                species_dat.write(outstr)

    def _equilibriate_atmo_and_ocean(
        self,
        temperature: float,
    ) -> None:
        """
        Chemically equilibriates the atmosphere-melt interface, and computes the vapour
        composition.

        Parameters
        ----------
            temperature : float
                Temperature of the planet's "surface", i.e. the (lava-)ocean, in K.

        """

        self.atmo = self.vapour_engine.equilibriate_vapour(temperature=temperature)
        self.t_boa = temperature

    def _helios_setup(
        self,
        param_file: str,
        run_type: Literal["iterative", "post-processing"] = "iterative",
        opacity_mixing: Literal["premixed", "on-the-fly"] = "on-the-fly",
        cuda_kws: dict = None,
    ) -> None:
        """
        Sets up the HELIOS model parameters and reads the necessary input files.

        Parameters
        ----------
        param_file : str
            The path to the parameter file.
        run_type : str, optional
            The type of run to perform. Can be 'iterative' or 'post-processing'. Default is
            'iterative'.
        opacity_mixing : str, optional
            The method for opacity mixing. Can be 'on-the-fly' or 'premixed'. Default is
            'on-the-fly'.
        cuda_kws : dict, optional
            Additional keywords for CUDA configuration.
        """

        if cuda_kws is None:
            cuda_kws = {}

        # Initialize components as instance attributes
        self._reader = read.Read()
        self._keeper = quant.Store()
        self._computer = comp.Compute(cuda_kws=cuda_kws)
        self._writer = write.Write()
        self._plotter = rt_plot.Plot()
        self._fogger = clouds.Cloud()

        # Read input files and preliminary calculations
        self._reader.read_param_file_and_command_line(
            param_file, self._keeper, self._fogger
        )
        self._reader.output_path = self.outdir

        self._keeper.run_type = run_type
        self._keeper.nlayer = np.int32(self.nlayer)
        self._keeper.ninterface = np.int32(self.nlayer + 1)

        # Set run type automation
        self._configure_run_type(self._keeper)

        # Set species file based on run type
        self._set_species_file(self._reader, self._keeper)

        # Specify opacity path
        self._reader.opacity_path = self.opacity_path

        # Pass pressure and convert to HELIOS units (10⁻⁶ bar)
        self._keeper.p_toa = self.p_toa / 1e-6
        self._reader.fastchem_path = self.outdir

        # Handle opacity mixing
        self._handle_opacity_mixing(self._reader, self._keeper, opacity_mixing)

        # Read kappa table or use constant kappa
        self._reader.read_kappa_table_or_use_constant_kappa(self._keeper)
        self._reader.read_or_fill_surf_albedo_array(self._keeper)
        self._keeper.dimensions()

        # Read stellar spectrum
        self._configure_stellar_spectrum(self._reader)

        # Read planetary and orbital parameters
        self._configure_planetary_parameters(self._keeper)

    def _configure_run_type(self, keeper: quant.Store) -> None:
        """
        Configures the run type settings in the keeper object.

        Parameters
        ----------
            keeper : quant.Store
                A HELIOS object storing all runtime parameters.

        """
        if keeper.run_type == "iterative":
            keeper.singlewalk = np.int32(0)
            keeper.iso = np.int32(0)
            keeper.energy_correction = np.int32(1)
            keeper.name = "HELIOS_iterative"
        elif keeper.run_type == "post-processing":
            keeper.singlewalk = np.int32(1)
            keeper.iso = np.int32(1)
            keeper.energy_correction = np.int32(0)
            keeper.name = "HELIOS_postproc"
        else:
            raise ValueError(f"Invalid run type: {keeper.run_type}")

    def _set_species_file(self, reader: read.Read, keeper: quant.Store) -> None:
        """
        Sets the species file path based on the run type (iterative vs. post-processing).

        Parameters
        ----------
            reader : read.Read
                A HELIOS object for reading-in the run-parameters.
            keeper : quant.Store
                A HELIOS object storing all runtime parameters.

        """
        species_file_path = f"{self.outdir}/species_iterative.dat"
        reader.species_file = species_file_path

        if keeper.run_type == "post-processing":
            reader.temp_path = f"{self.outdir}/HELIOS_iterative/tp.dat"
            reader.temp_format = "helios"

    def _handle_opacity_mixing(
        self,
        reader: read.Read,
        keeper: quant.Store,
        opacity_mixing: Literal["premixed", "on-the-fly"],
    ) -> None:
        """
        Handles the opacity mixing method.

        Parameters
        ----------
            reader : read.Read
                A HELIOS object for reading-in the run-parameters.
            keeper : quant.Store
                A HELIOS object storing all runtime parameters.
            opacity_mixing : str
                Method of opacity mising.

        """
        keeper.opacity_mixing = opacity_mixing
        if keeper.opacity_mixing == "premixed":
            reader.load_premixed_opacity_table(keeper)
        elif keeper.opacity_mixing == "on-the-fly":
            reader.read_species_file(keeper)
            reader.read_species_opacities(keeper)
            reader.read_species_scat_cross_sections(keeper)
        else:
            raise NotImplementedError(
                f"Opacity mixing method '{opacity_mixing}' is not implemented."
            )

    def _configure_stellar_spectrum(self, reader: read.Read) -> None:
        """
        Configures the stellar spectrum based on the planetary system.

        Parameters
        ----------
            reader : read.Read
                A HELIOS object for reading-in the run-parameters.

        """
        reader.stellar_model = self.planetary_system.star.file_or_blackbody

        if reader.stellar_model == "file":
            reader.stellar_path = self.planetary_system.star.path_to_h5
            reader.stellar_data_set = self.planetary_system.star.path_in_h5

        reader.read_star(self._keeper)

    def _configure_planetary_parameters(self, keeper: quant.Store) -> None:
        """
        Reads and sets the planetary and orbital parameters.

        Parameters
        ----------
            keeper : quant.Store
                A HELIOS object storing all runtime parameters.

        """
        keeper.g = keeper.fl_prec(
            self.planetary_system.planet.grav.to("cm / s^2").value
        )
        keeper.a = keeper.fl_prec(
            self.planetary_system.orbit.get_semimajor_axis(
                self.planetary_system.star.mass
            )
            .to("cm")
            .value
        )
        keeper.R_planet = keeper.fl_prec(
            self.planetary_system.planet.radius.to("cm").value
        )
        keeper.R_star = keeper.fl_prec(self.planetary_system.star.radius.to("cm").value)
        keeper.T_star = keeper.fl_prec(self.planetary_system.star.t_eff.to("K").value)
        keeper.T_intern = float(
            self.planetary_system.planet.internal_temperature.to("K").value
        )
        keeper.f_factor = np.float64(self.planetary_system.planet.dilution_factor)

    def _solve_rad_trans(self):
        """
        Solves the radiative transfer problem for a specified atmosphere using HELIOS
        """
        hsfunc.set_up_numerical_parameters(self._keeper)
        hsfunc.construct_grid(self._keeper)
        # hsfunc.initial_temp(keeper, reader, init_T_profile) TODO: implement!
        hsfunc.initial_temp(self._keeper, self._reader)

        if self._keeper.approx_f == 1 and self._keeper.planet_type == "rocky":
            hsfunc.approx_f_from_formula(self._keeper, self._reader)

        hsfunc.calc_F_intern(self._keeper)
        add_heat.load_heating_terms_or_not(self._keeper)

        self._fogger.cloud_pre_processing(self._keeper)

        # create, convert and copy arrays to be used in the GPU computations
        self._keeper.create_zero_arrays()
        self._keeper.convert_input_list_to_array()
        self._keeper.copy_host_to_device()
        self._keeper.allocate_on_device()

        # ---------- conduct core computations on the GPU --------------#
        self._computer.construct_planck_table(self._keeper)
        self._computer.correct_incident_energy(self._keeper)
        self._computer.radiation_loop(
            self._keeper, self._writer, self._reader, self._plotter
        )
        self._computer.convection_loop(
            self._keeper, self._writer, self._reader, self._plotter
        )

        self._computer.integrate_optdepth_transmission(self._keeper)
        self._computer.calculate_contribution_function(self._keeper)
        if self._keeper.convection == 1:
            self._computer.interpolate_entropy(self._keeper)
            self._computer.interpolate_phase_state(self._keeper)
        self._computer.calculate_mean_opacities(self._keeper)
        self._computer.integrate_beamflux(self._keeper)

        # ---------------- BACK TO HOST --------------#
        # copy everything from the GPU back to host and write output quantities to files
        self._keeper.copy_device_to_host()
        hsfunc.calculate_conv_flux(self._keeper)
        hsfunc.calc_F_ratio(self._keeper)

    def _write_helios_output(self):
        """
        Write all outputs in HELIOS-style. Target is HELIOS_iterative folder in 'outdir'.
        """
        os.makedirs(self.outdir + "HELIOS_iterative", exist_ok=True)

        self._writer.write_colmass_mu_cp_entropy(self._keeper, self._reader)
        self._writer.write_integrated_flux(self._keeper, self._reader)
        self._writer.write_downward_spectral_flux(self._keeper, self._reader)
        self._writer.write_upward_spectral_flux(self._keeper, self._reader)
        self._writer.write_TOA_flux_eclipse_depth(self._keeper, self._reader)
        self._writer.write_direct_spectral_beam_flux(self._keeper, self._reader)
        self._writer.write_planck_interface(self._keeper, self._reader)
        self._writer.write_planck_center(self._keeper, self._reader)
        self._writer.write_tp(self._keeper, self._reader)
        self._writer.write_tp_cut(self._keeper, self._reader)
        self._writer.write_opacities(self._keeper, self._reader)
        self._writer.write_cloud_mixing_ratio(self._keeper, self._reader)
        self._writer.write_cloud_opacities(self._keeper, self._reader)
        self._writer.write_Rayleigh_cross_sections(self._keeper, self._reader)
        self._writer.write_cloud_scat_cross_sections(self._keeper, self._reader)
        self._writer.write_g_0(self._keeper, self._reader)
        self._writer.write_transmission(self._keeper, self._reader)
        self._writer.write_opt_depth(self._keeper, self._reader)
        self._writer.write_cloud_opt_depth(self._keeper, self._reader)
        self._writer.write_trans_weight_function(self._keeper, self._reader)
        self._writer.write_contribution_function(self._keeper, self._reader)
        self._writer.write_mean_extinction(self._keeper, self._reader)
        self._writer.write_flux_ratio_only(self._keeper, self._reader)
        self._writer.write_phase_state(self._keeper, self._reader)
        self._writer.write_surface_albedo(self._keeper, self._reader)
        self._writer.write_criterion_warning_file(self._keeper, self._reader)

        if self._keeper.coupling == 1:
            self._writer.write_tp_for_coupling(self._keeper, self._reader)
            hsfunc.calculate_coupling_convergence(self._keeper, self._reader)

        if self._keeper.approx_f == 1:
            hsfunc.calc_tau_lw_sw(self._keeper, self._reader)

        hsfunc.success_message(self._keeper)
