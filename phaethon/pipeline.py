#
# Copyright 2024-2026 Fabian L. Seidler
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
Main pipeline, streamlines the computation of the structure of an outgassed atmosphere.
"""

import importlib
import traceback
import sys
import json
import logging
import os
import time
from typing import Optional, Literal, Dict, Union
import numpy as np
import pandas as pd
from copy import deepcopy

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
from phaethon.analyse import PhaethonResult
from phaethon.fastchem.coupling import FastChemCoupler
from phaethon.gas_mixture import IdealGasMixture
from phaethon.interfaces import (
    OutgassingProtocol,
    IteratorProtocol,
    PostRadtransProtocol,
)
from phaethon.iterator import MeltTemperatureIterator, SingleIteration
from phaethon.logger import file_logger

DEFAULT_CONFIG_FILE = (
    importlib.resources.files("phaethon.data") / "default_config.dat"
)
"""
Default HELIOS config. NOTE: Many of the default parameters will be overwritten by the pipeline
below (mostly params related to planet, star and opacity species).
"""


class PhaethonPipeline:
    """
    Combines and stores all modules for a self-consistent simulation of an outgassed atmosphere.
    """

    planetary_system: PlanetarySystem
    outgassing: OutgassingProtocol
    fastchem_coupler: FastChemCoupler

    opacity_path: str
    opac_species: set
    scatterers: set
    atmo: IdealGasMixture

    p_toa: float
    p_boa: Optional[float]  # not set at init; changes during run.
    t_boa: Optional[float]  # not set at init; changes during run.
    t_melt: float  # set at init, but changes during run.

    # solver
    iterator: IteratorProtocol

    # HELIOS objects; optional since they are initially none, and only set by `_helios_setup`
    _reader: Optional[read.Read]
    _keeper: Optional[quant.Store]
    _computer: Optional[comp.Compute]
    _writer: Optional[write.Write]
    _plotter: Optional[rt_plot.Plot]
    _fogger: Optional[clouds.Cloud]

    # post-radtrans routine
    postradtrans: Optional[PostRadtransProtocol]

    def __init__(
        self,
        planetary_system: PlanetarySystem,
        outgassing: OutgassingProtocol,
        fastchem_coupler: FastChemCoupler,
        outdir: str,
        opac_species: set,
        scatterers: set,
        opacity_path: str,
        p_toa: float = 1e-8,
        iterator: IteratorProtocol = MeltTemperatureIterator(
            delta_temp_abstol=10,
            max_iter=15,
            tmelt_limits=(10.0, 10000.0),
        ),
        postradtrans: Optional[PostRadtransProtocol] = None,
    ) -> None:
        """
        Init the phaethon pipeline.

        Parameters
        ----------
        planetary_system : PlanetarySystem
            The planetary system to be modeled.
        outgassing : OutgassingProtocol
            Outgassing/vaporization processes; supplies the bulk elemental chemistry of the
            atmosphere.
        fastchem_coupler : FastChemCoupler
            Interface to FastChem.
        outdir : str
            The directory where output files will be stored.
        opac_species : set
            A set of species to be considered for opacity calculations.
        scatterers : set
            A set of scatterer species included in the model.
        opacity_path : str
            Path to the opacity data files.
        p_toa : float, optional
            Total atmospheric pressure at the top of the atmosphere (in bar). Default is 1e-8.
        iterator : IteratorProtocol
            Iterator that converges atmosphere and melt. Default is `MeltTemperatureIterator`.
        postradtrans : Optional[PostRadtransProtocol]
            A post-radtrans protocol to compute transmission (and emission spectra) from the final
            (fully equilibrated) atmospheric profile provided by HELIOS. Default is None.
        """

        if not outdir.endswith("/"):
            outdir += "/"
        self.outdir = outdir

        self.planetary_system = planetary_system
        self.outgassing = outgassing
        self.fastchem_coupler = fastchem_coupler
        self.opacity_path = opacity_path
        self.opac_species = opac_species
        self.scatterers = scatterers
        self.atmo = IdealGasMixture.new_from_pressure({})
        self.p_toa = p_toa
        self.p_boa = None
        self.t_boa = None
        self.t_melt = self.planetary_system.irrad_temp.to("K").value

        # solver
        self.iterator = iterator

        # HELIOS objects
        self._reader = None
        self._keeper = None
        self._computer = None
        self._writer = None
        self._plotter = None
        self._fogger = None

        # post-radtrans routine (e.g., petitRADTRANS)
        self.postradtrans = postradtrans

    def info_dump(self) -> None:
        """
        Dumps all info into a JSON-file in `self.outdir`.
        """

        # info on planet, star, and orbit
        metadata: Dict[str, Union[float, str, int]] = self.planetary_system.info

        # info on outgassing routine
        outgassing_dict = {
            "outgassing": {
                "type": str(type(self.outgassing)),
                "report": self.outgassing.get_info(),
            }
        }
        metadata.update(outgassing_dict)

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
        config_file: os.PathLike = DEFAULT_CONFIG_FILE,
        *,
        t_melt_init: Optional[float] = None,
        nvcc_kws: Optional[dict] = None,
        logfile_name: str = "phaethon.log",
    ) -> None:
        """
        Equilibrates an atmosphere with the underlying (magma) ocean and solves the radiative
        transfer problem.

        Parameters
        ----------
            config_file : os.PathLike
                File containing the HELIOS parameters.
            t_abstol : float (optional)
                ΔT allowed between t_melt and t_boa, in K.
            t_melt_init : Optional[float] (optional)
                Optional starting temperature of the melt, in K.
            nvcc_kws : dict  (optional)
                Dictionary containing additional args (e.g., compiler flags) for nvcc, the CUDA
                compiler.
            logfile_name : str (optional)
                Name of the logfile, placed in the output directory. Default is "phaethon.log".
        """

        if nvcc_kws is None:
            nvcc_kws = {}

        # create outdir
        os.makedirs(self.outdir, exist_ok=True)

        # init logger
        logger = file_logger(logfile=self.outdir + logfile_name)

        # Write an initial metadata file
        self.info_dump()

        # Generate the file which lists opacity & scattering species; req. by HELIOS
        self._write_opacspecies_file()

        # Inital temperature of the melt, based on the irradiation temperature or constant input.
        # Do this every time `self.run()` is called, because `self.t_melt` gets overwritte during
        # each iteration.
        self.t_melt = (
            self.planetary_system.irrad_temp.to("K").value
            if t_melt_init is None
            else t_melt_init
        )
        if self.t_melt <= 0:
            logger.error(f"`t_melt` must be > 0, is {self.t_melt} K")
            raise ValueError(f"`t_melt` must be > 0, is {self.t_melt} K")

        # initialise HELIOS
        self._helios_setup(config_file=config_file, nvcc_kws=nvcc_kws)

        # run the loop
        start: float = time.time()
        logger.info(f"Starting temperature (melt): {self.t_melt} K")

        try:
            # solve for convergence
            self.iterator.iterate(pipeline=self, logger=logger)

            # store output and metadata
            self._write_helios_output()
            self.info_dump()

            # final fastchem run, generate atmospheric abundance profiles
            self._final_fastchem_run()

            end: float = time.time()
            logger.info(f"Finished. Duration: {(end - start) / 60.} min")

        # log error, just in case
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_details = traceback.format_exception(
                exc_type, exc_value, exc_traceback
            )
            logger.error("".join(traceback_details))
            raise Exception(
                f'An error occured: "{exc_value}". For more information, consult the '
                + f"logfile at {self.outdir + logfile_name}"
            )

        # clear cached helios data, because they can occupy large amounts of memory
        finally:
            self._wipe_helios_memory(nvcc_kws=nvcc_kws)
            logger.info("HELIOS memory wiped")

        # run postradtrans on fully equilibrated atmosphere
        self.run_post_radtrans(logger)

    def run_post_radtrans(self, logger: Optional[logging.Logger] = None):
        """
        Executes a radiative transfer code (petitRADTRANS, TauREx) using the equilibrated
        P-T-structure obtained by HELIOS.

        Parameters
        ----------
            logger : Optional[logging.Logger]
                external logger; default is `None`
        """
        if self.postradtrans is not None:

            if logger is not None:
                logger.info("Running post-radtrans routine...")

            phaethon_result = PhaethonResult(
                self.outdir, load_star=False
            )  # TODO: fix load_star so it can be always andloaded
            self.postradtrans.set_atmo(phaethon_result)

            # calc spectra; wavl is the same for all, so only store it once
            wavl, fpfs = self.postradtrans.calc_fpfs(use_phoenix=True)
            _, transm_radius = self.postradtrans.calc_transm_radius()
            _, transm_depth = self.postradtrans.calc_transm_depth()
            _, planet_flux = self.postradtrans.calc_planet_flux()

            # save as pandas data frame
            df = pd.DataFrame()
            df["wavl"] = wavl.to("micron").value
            df["transm_radius"] = transm_radius.to("R_earth").value
            df["transm_depth"] = transm_depth.value * 1e6  # dimensionless; to ppm
            df["fpfs"] = fpfs.value * 1e6  # dimensionless; to ppm
            df["planet_flux"] = planet_flux.to("erg / (s * cm3)").value
            with open(self.outdir + "postradtrans.csv", "w") as postrad_outfile:
                postrad_outfile.write(
                    f"# Transmission and emission spectra computed by the post-radtrans routine.\n"
                )
                postrad_outfile.write(f"# wavl [µm]\n")
                postrad_outfile.write(f"# transm_radius [earth radii]\n")
                postrad_outfile.write(f"# transm_depth [ppm]\n")
                postrad_outfile.write(f"# fpfs [ppm]\n")
                postrad_outfile.write(f"# planet_flux [erg / (s * cm3)]\n")
                df.to_csv(postrad_outfile, index=None)

        else:
            if logger is not None:
                logger.info("No postradtrans routine specified, skipping.")

    def single_run(
        self,
        *,
        config_file: os.PathLike = DEFAULT_CONFIG_FILE,
        t_melt: float,
        nvcc_kws: Optional[dict] = None,
        logfile_name: str = "phaethon.log",
    ) -> None:
        """
        Runs a single forward cycle, outgassing -> fastchem -> HELIOS.

        Parameters
        ----------
            config_file : os.PathLike
                File containing the HELIOS parameters.
            t_melt_init : Optional[float] (optional)
                Optional starting temperature of the melt, in K.
            nvcc_kws : dict  (optional)
                Dictionary containing additional args (e.g., compiler flags) for nvcc, the CUDA
                compiler.
            logfile_name : str (optional)
                Name of the logfile, placed in the output directory. Default is "phaethon.log".
        """

        # store iterator so that it can be reset after
        old_iterator = deepcopy(self.iterator)

        # set new iterator
        self.iterator = SingleIteration()
        self.run(
            config_file, t_melt_init=t_melt, nvcc_kws=nvcc_kws, logfile_name=logfile_name
        )

        # reset iterator
        self.iterator = old_iterator

    # ============================================================================================
    # SEMI-PRIVATE METHODS
    # ============================================================================================

    def _single_forward_iteration(self, t_melt: float) -> None:
        """
        Performs a single forward iteration: outgassing -> fastchem -> helios

        Parameters
        ----------
            t_melt : float
                Melt temperature, in K.
        """

        # equilibriate atmosphere-melt interface, compute atmosphere compositon
        self.atmo = self.outgassing.equilibriate(temperature=t_melt)
        self.p_boa = self.atmo.p_total  # bar

        # check result of outgassing
        if self.atmo.log_p.empty:
            raise ValueError("'VapourEngine' returned an empty result!")
        if self.p_toa > self.p_boa:
            raise ValueError(
                f"The vapour pressure ({self.atmo.p_total} bar) is smaller than the TOA pressure"
                + f" ({self.p_toa} bar)!"
            )

        # chemistry look-up tables with FastChem
        p_grid, t_grid = self.fastchem_coupler.get_grid(
            pressures=np.logspace(np.log10(self.p_toa), np.log10(self.p_boa), 100),
            temperatures=np.linspace(1000, 6000, 100),
        )
        self.fastchem_coupler.run_fastchem(
            vapour=self.atmo,
            pressures=p_grid,
            temperatures=t_grid,
            outdir=self.outdir,
            outfile_name="chem.dat",  # DO NOT CHANGE - HELIOS searches for "chem.dat"
        )

        # solve radiative transfer
        self._solve_rad_trans(p_boa=self.p_boa)

        # store bottom-of-atmosphere temperature
        self.t_boa = self._keeper.T_lay[self._keeper.nlayer]

    def _final_fastchem_run(self) -> None:
        """
        Performs a final FastChem run along the computed P-T-profile in order to obtain the
        atmospheric speciations as function of pressure/altitude. Allows for using various
        condensation modes.
        """
        df = pd.read_csv(self.outdir + "HELIOS_iterative/tp.dat", header=1, sep=r"\s+")
        self.fastchem_coupler.run_fastchem(
            vapour=self.atmo,
            pressures=df["press.[10^-6bar]"].to_numpy(float) * 1e-6,
            temperatures=df["temp.[K]"].to_numpy(float),
            outdir=self.outdir,
            outfile_name="chem_profile.dat",  # DO NOT CHANGE - PhaethonResult searches it!
        )

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

    def _wipe_helios_memory(self, nvcc_kws: Dict[str, str]) -> None:
        """
        Clears the cached data from HELIOS in order to free memory; is called at the end of the
        pipeline. Otherwise, HELIOS might crash after subsequent calls to the pipeline.

        Parameters
        ----------
        nvcc_kws : dict, optional
            Additional keywords for CUDA configuration. Needed during reset of the HELIOS
            computation module.
        """

        if nvcc_kws is None:
            nvcc_kws = {}

        self._reader = read.Read()
        self._keeper = quant.Store()
        self._computer = comp.Compute(nvcc_kws=nvcc_kws)
        self._writer = write.Write()
        self._plotter = rt_plot.Plot()
        self._fogger = clouds.Cloud()

    def _helios_setup(
        self,
        config_file: str,
        run_type: Literal["iterative", "post-processing"] = "iterative",
        nvcc_kws: dict = None,
    ) -> None:
        """
        Configures the HELIOS model from a config file and compiler arguments.

        Parameters
        ----------
        config_file : str
            The path to the parameter file.
        run_type : str, optional
            The type of run to perform. Can be 'iterative' or 'post-processing'. Default is
            'iterative'.
        nvcc_kws : dict, optional
            Additional keywords for CUDA configuration.
        """

        if nvcc_kws is None:
            nvcc_kws = {}

        # Initialize components as instance attributes
        self._reader = read.Read()
        self._keeper = quant.Store()
        self._computer = comp.Compute(nvcc_kws=nvcc_kws)
        self._writer = write.Write()
        self._plotter = rt_plot.Plot()
        self._fogger = clouds.Cloud()

        # Read input files and preliminary calculations
        self._reader.read_param_file_and_command_line(
            config_file, self._keeper, self._fogger
        )
        self._reader.output_path = self.outdir

        # Set run type automation
        self._keeper.run_type = run_type
        self._configure_run_type(self._keeper)

        # Set species file based on run type
        self._set_species_file(self._reader, self._keeper)

        # Specify opacity path
        self._reader.opacity_path = self.opacity_path

        # Pressure & chemistry; convert to HELIOS units (10⁻⁶ bar)
        self._keeper.p_toa = self.p_toa / 1e-6
        self._reader.fastchem_path = self.outdir

        # Opacities; strictly use on-the-fly mixing, as phaethon is not intended to be used with
        # premixed opacities; otherwise, use "reader.load_premixed_opacity_table(keeper)"
        self._keeper.opacity_mixing = "on-the-fly"
        self._reader.read_species_file(self._keeper)
        self._reader.read_species_opacities(self._keeper)
        self._reader.read_species_scat_cross_sections(self._keeper)

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
        keeper.a = keeper.fl_prec(self.planetary_system.semimajor_axis.to("cm").value)
        keeper.R_planet = keeper.fl_prec(
            self.planetary_system.planet.radius.to("cm").value
        )
        keeper.R_star = keeper.fl_prec(self.planetary_system.star.radius.to("cm").value)
        keeper.T_star = keeper.fl_prec(self.planetary_system.star.t_eff.to("K").value)
        keeper.T_intern = float(
            self.planetary_system.planet.internal_temperature.to("K").value
        )
        keeper.f_factor = np.float64(self.planetary_system.planet.dilution_factor)

    def _solve_rad_trans(self, p_boa: float):
        """
        Solves the radiative transfer problem for a specified atmosphere using HELIOS
        """

        # inform HELIOS about the change in atmospheric pressure
        self._keeper.p_boa = float(p_boa) / 1e-6  # weird HELIOS scaling (dyne)

        # read mixing ratios of gas species from fastchem file
        self._reader.read_species_mixing_ratios(self._keeper)

        # ???
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
