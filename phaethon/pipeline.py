from typing import Union
import numpy as np
import pandas as pd
import os
import json

# ----- HELIOS imports -----#
from helios import read
from helios import quantities as quant
from helios import host_functions as hsfunc
from helios import write
from helios import computation as comp
from helios import realtime_plotting as rt_plot
from helios import clouds
from helios import additional_heating as add_heat

# ----- PHAETHON imports -----#
from phaethon.celestial_objects import PlanetarySystem
from phaethon.outgassing import VapourEngine
from phaethon.fastchem_coupling import FastChemCoupler
from phaethon.gas_mixture import IdealGasMixture

# ============== Constants ================#
Mearth = 5.974e24
Rearth = 6.371e6
Rsun = 6.969e8
Msun = 1.989e30
Rjup = 6.9911e7


# ================================================================================================
#   CLASSES
# ================================================================================================


class PhaethonRunner:
    planetary_system: PlanetarySystem
    vapour_engine: VapourEngine
    fastchem_coupler: FastChemCoupler

    opacity_path: str
    opac_species: set
    scatterers: set
    atmo: Union[None, IdealGasMixture]

    p_toa: float
    t_boa: float

    def __init__(
        self,
        planetary_system: PlanetarySystem,
        vapour_engine: VapourEngine,
        outdir: str,
        opac_species: set,
        scatterers: set,
        opacity_path: str,
        path_to_eqconst: Union[None, str] = None,
        p_toa: float = 1e-8,
        p_grid_fastchem: np.ndarray = np.logspace(-8, 3, 100),
        t_grid_fastchem: np.ndarray = np.linspace(500, 6000, 100),
    ) -> None:
        """
        Init the phaethon runner.
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
        self.t_boa = self.planetary_system.planet.temperature.value
        self._p_grid_fastchem, self._t_grid_fastchem = self.fastchem_coupler.get_grid(
            pressures=p_grid_fastchem, temperatures=t_grid_fastchem
        )

    def info_dump(self) -> None:
        """Puts all info into self.outdir"""

        metadata = self.planetary_system.get_info()

        vapour_engine_dict = {
            "vapour_engine": {
                "type": str(type(self.vapour_engine)),
                "params": self.vapour_engine.get_info(),
            }
        }
        metadata.update(vapour_engine_dict)

        atmo_dict = {
            "atmosphere": {
                "species": list(self.opac_species),
                "scatterers": list(self.scatterers),
                "p_bar": self.atmo.p_total,
                "log_p": np.log10(self.atmo.p_total),
                "t_boa:": self.t_boa,
            }
        }
        metadata.update(atmo_dict)

        with open(self.outdir + "metadata.json", "w") as metadata_file:
            json.dump(metadata, metadata_file)

    def run(
        self,
        t_abstol: float = 35.0,
        standard_param_file: str = "phaethon/data/standard_lavaplanet_params.dat",
    ) -> None:

        # Dump info initially
        self.info_dump()

        self.planetary_system.calc_pl_temp()
        self.t_boa = self.planetary_system.planet.temperature.value

        t_melt_trace = [self.t_boa]
        t_melt_trace.append(self.t_boa)

        self._write_opacspecies_file()
        self._helios_setup(standard_param_file=standard_param_file)

        delta_t_melt = np.inf
        try_second_loop = False

        while delta_t_melt > t_abstol:
            # Vapour composition & pressure
            self._equilibriate_surface(surface_temperature=self.t_boa)
            assert not self.atmo.log_p.empty
            assert self.p_toa < self.atmo.p_total
            self._keeper.p_boa = float(self.atmo.p_total) / 1e-6

            # FastChem
            self.fastchem_coupler.run_fastchem(
                vapour=self.atmo,
                pressures=self._p_grid_fastchem,
                temperatures=self._t_grid_fastchem,
                outdir=self.outdir,
                outfile_name="chem.dat", # DO NOT CHANGE - helios searches for "chem.dat"
                cond_mode="none",
            )
            self._reader.read_species_mixing_ratios(self._keeper)

            # solve radiative transfer
            self._solve_rad_trans()

            delta_t_melt = abs(self.t_boa - self._keeper.T_lay[self._keeper.nlayer])
            self.t_boa = self._keeper.T_lay[self._keeper.nlayer]

            t_melt_trace.append(self.t_boa)

            if (
                len(t_melt_trace) >= 3
                and abs(t_melt_trace[-3] - t_melt_trace[-1]) <= t_abstol
            ):
                print("Melt temperature series seems to be oscillating")
                try_second_loop = True
                break

        # ----------- if oscillating, try branch-and-bound --------#
        if try_second_loop:
            search_range = [np.amin(t_melt_trace), np.amax(t_melt_trace)]

            while abs(delta_t_melt) > t_abstol:
                t_boa = (search_range[0] + search_range[1]) / 2.0

                # Vapour composition & pressure
                self._equilibriate_surface(surface_temperature=self.t_boa)
                assert not self.atmo.log_p.empty
                assert self.p_toa < self.atmo.p_total
                self._keeper.p_boa = float(self.atmo.p_total) / 1e-6

                # FastChem
                self.fastchem_coupler.run_fastchem(
                    vapour=self.atmo,
                    pressures=self._p_grid_fastchem,
                    temperatures=self._t_grid_fastchem,
                    outdir=self.outdir,
                    outfile_name="chem.dat", # DO NOT CHANGE - helios searches for "chem.dat"
                    cond_mode="none",
                )
                self._reader.read_species_mixing_ratios(self._keeper)

                # solve radiative transfer
                self._solve_rad_trans()

                delta_t_melt = abs(self.t_boa - self._keeper.T_lay[self._keeper.nlayer])
                self.t_boa = self._keeper.T_lay[self._keeper.nlayer]

                if delta_t_melt < 0:  # melt cooler than T_BOA
                    search_range[0] = t_boa
                elif delta_t_melt > 0:  # melt hotter than T_BOA
                    search_range[1] = t_boa

        self._write_helios_output()
        self.info_dump()

        # ---------- final FastChem run -----------#
        df = pd.read_csv(self.outdir + "HELIOS_iterative/tp.dat", header=1, delim_whitespace=True)
        self.fastchem_coupler.run_fastchem(
            vapour=self.atmo,
            pressures=df["press.[10^-6bar]"].to_numpy(float) * 1e-6,
            temperatures=df["temp.[K]"].to_numpy(float),
            outdir=self.outdir,
            outfile_name="chem_profile.dat", # DO NOT CHANGE - PhaethonResult searches for "chem_profile.dat"
            cond_mode="none",
        )

    # ============================================================================================
    # PRIVATE METHODS
    # ============================================================================================

    def _write_opacspecies_file(self):
        """
        Generates atmospheric species files. Necessary for HELIOS to know
        which species to include.

        Parameters
        ----------
        atmo_molar_chem : set
            contains the molar fraction of elements in the planets atmosphere,
            i.e. {'Si':0.34, 'O':0.56, 'Na':0.07, 'Fe':0.03}. Does not have to be normalized.
        ignore_species : set
            species to ignore, i.e. {'TiO', 'Cr'}
        """

        outfile = self.outdir + "/species_iterative.dat"

        with open(outfile, "w") as species_dat:
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

    def _equilibriate_surface(
        self,
        surface_temperature: float,
        extra_params: Union[None, dict] = None,
    ) -> None:
        if extra_params is not None:
            self.vapour_engine.set_extra_params(params=extra_params)
        self.atmo = self.vapour_engine.equilibriate_vapour(
            surface_temperature=surface_temperature
        )
        self.t_boa = surface_temperature

    def _helios_setup(
        self,
        standard_param_file: str,
        run_type: str = "iterative",
        opacity_mixing: str = "on-the-fly",
    ) -> None:
        """
        Parameters
        ----------
        calcname : str
            name of the calculation
        p_boa : float
            pressure at the bottom-of-the-atmosphere (BOA), in bar
        lavaplanet : dict
            dictionary with key parameters of a lavaplanet:
        star : class
            everything that characterizes the host star
        outdir : str
            directory where HELIOS Input (i.e., FastChem files) and HELIOS Output will reside
        """

        reader = read.Read()
        keeper = quant.Store()
        computer = comp.Compute()
        writer = write.Write()
        plotter = rt_plot.Plot()
        fogger = clouds.Cloud()

        # read input files and do preliminary calculations, like setting up the grid, etc.
        # TODO: remove this?
        reader.read_param_file_and_command_line(standard_param_file, keeper, fogger)

        reader.output_path = self.outdir

        keeper.run_type = run_type
        reader.run_type = run_type

        # set run type automatization
        if keeper.run_type == "iterative":
            keeper.singlewalk = np.int32(0)
            keeper.iso = np.int32(0)
            keeper.energy_correction = np.int32(1)
            keeper.name = "HELIOS_iterative"

        # TODO: implement proper treatment of p-T-profile. Also, assert that iterative has been run.
        elif keeper.run_type == "post-processing":
            keeper.singlewalk = np.int32(1)
            keeper.iso = np.int32(1)
            keeper.energy_correction = np.int32(0)
            keeper.name = "HELIOS_postproc"

        # species file
        if keeper.run_type == "post-processing":
            reader.species_file = (
                outdir + "/species_iterative.dat"
            )  # currently it seems that species for iterative and post-processing have to be the same
            reader.temp_path = self.outdir + "HELIOS_iterative/tp.dat"
            reader.temp_format = "helios"
        else:
            reader.species_file = self.outdir + "/species_iterative.dat"

        # specify opacity path
        reader.opacity_path = self.opacity_path

        # pass pressure, convert to weird HELIOS units (10⁻⁶ bar)
        keeper.p_toa = self.p_toa / 1e-6

        # path to FastChem results
        reader.fastchem_path = self.outdir

        # opacity mixing
        keeper.opacity_mixing = opacity_mixing
        if keeper.opacity_mixing == "premixed":
            reader.load_premixed_opacity_table(keeper)

        elif keeper.opacity_mixing == "on-the-fly":
            reader.read_species_file(keeper)
            reader.read_species_opacities(keeper)
            reader.read_species_scat_cross_sections(keeper)
        else:
            raise NotImplementedError

        reader.read_kappa_table_or_use_constant_kappa(keeper)
        reader.read_or_fill_surf_albedo_array(keeper)
        keeper.dimensions()

        # -------- read stellar spectrum -----#
        reader.stellar_model = self.planetary_system.star.file_or_blackbody

        if self.planetary_system.star.file_or_blackbody == "file":
            reader.stellar_path = self.planetary_system.star.path_to_h5
            reader.stellar_data_set = self.planetary_system.star.path_in_h5

        #   finish
        reader.read_star(keeper)

        # ----------- planetary & orbital parameters --------#
        # read planetary params (for code stability only), modify them and turn them to cgs
        # hsfunc.planet_param(keeper, reader)
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

        # ---------- radiation --------#
        # keeper.f_factor = float(self.planetary_system.planet.dilution_factor)

        self._reader = reader
        self._keeper = keeper
        self._computer = computer
        self._writer = writer
        self._plotter = plotter
        self._fogger = fogger

    def _solve_rad_trans(self):
        # this seems necessary to make it work with subsequent iterations
        self._keeper.delta_colmass = []
        self._keeper.delta_col_upper = []
        self._keeper.delta_col_lower = []

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

        # prints the success message - yay!
        hsfunc.success_message(self._keeper)
