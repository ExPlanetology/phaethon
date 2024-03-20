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

    def __init__(
        self,
        planetary_system: PlanetarySystem,
        vapour_engine: VapourEngine,
        outdir: str,
        opac_species: set,
        scatterers: set,
        opacity_path: str,
        path_to_eqconst: Union[None, str] = None,
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
        self.atmo = None

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
            }
        }
        metadata.update(atmo_dict)

        with open(self.outdir + "metadata.json", "w") as metadata_file:
            json.dump(metadata, metadata_file)

    def _write_atmospecies_file(self):
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

    def equilibriate_surface(
        self,
        surface_pressure: float,
        surface_temperature: float,
        extra_params: Union[None, dict] = None,
    ) -> None:
        if extra_params is not None:
            self.vapour_engine.set_extra_params(params=extra_params)
        self.atmo = self.vapour_engine.equilibriate_vapour(
            surface_pressure=surface_pressure, surface_temperature=surface_temperature
        )

    def _helios_setup(
        self,
        standard_param_file: str,
        run_type: str = "iterative",
        p_toa: float = 1e-8,
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

        assert self.atmo is not None

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
        keeper.p_boa = self.atmo.p_total / 1e-6
        keeper.p_toa = 1e-8 / 1e-6

        # # path to FastChem results
        # reader.fastchem_path = outdir

        # if keeper.opacity_mixing == "premixed":
        #     reader.load_premixed_opacity_table(keeper)

        # elif keeper.opacity_mixing == "on-the-fly":
        #     reader.read_species_file(keeper)
        #     reader.read_species_opacities(keeper)
        #     reader.read_species_scat_cross_sections(keeper)
        #     reader.read_species_mixing_ratios(keeper)
        # else:
        #     raise NotImplementedError

        # reader.read_kappa_table_or_use_constant_kappa(keeper)
        # reader.read_or_fill_surf_albedo_array(keeper)
        # keeper.dimensions()

        # # -------- read stellar spectru -----#
        # reader.stellar_model = star.file_or_BB

        # if star.file_or_BB == "file":
        #     reader.stellar_path = star.path_to_h5
        #     reader.stellar_data_set = star.path_in_h5

        # #   finish
        # reader.read_star(keeper)

        # # ----------- planetary & orbital parameters --------#
        # # read planetary params (for code stability only), modify them and turn them to cgs
        # # hsfunc.planet_param(keeper, reader)
        # keeper.g = keeper.fl_prec(lavaplanet.g * 100)
        # keeper.a = keeper.fl_prec(lavaplanet.semi_major_axis * 100)
        # keeper.R_planet = keeper.fl_prec(lavaplanet.R * 100)
        # keeper.R_star = keeper.fl_prec(star.R * 100)
        # keeper.T_star = keeper.fl_prec(star.T_eff)
        # keeper.T_intern = float(t_internal)

        # ---------- radiation --------#
        # keeper.f_factor = float(f_factor)

        self.reader = reader
        self.keeper = keeper
        self.computer = computer
        self.writer = writer
        self.plotter = plotter
        self.fogger = fogger


# def loop_helios(reader, keeper, computer, writer, plotter, fogger, init_T_profile=None):
#     hsfunc.set_up_numerical_parameters(keeper)
#     hsfunc.construct_grid(keeper)
#     hsfunc.initial_temp(keeper, reader, init_T_profile)

#     if keeper.approx_f == 1 and keeper.planet_type == "rocky":
#         hsfunc.approx_f_from_formula(keeper, reader)

#     hsfunc.calc_F_intern(keeper)
#     add_heat.load_heating_terms_or_not(keeper)

#     fogger.cloud_pre_processing(keeper)

#     # create, convert and copy arrays to be used in the GPU computations
#     keeper.create_zero_arrays()
#     keeper.convert_input_list_to_array()
#     keeper.copy_host_to_device()
#     keeper.allocate_on_device()

#     # ---------- conduct core computations on the GPU --------------#
#     computer.construct_planck_table(keeper)
#     computer.correct_incident_energy(keeper)

#     computer.radiation_loop(keeper, writer, reader, plotter)

#     computer.convection_loop(keeper, writer, reader, plotter)

#     computer.integrate_optdepth_transmission(keeper)
#     computer.calculate_contribution_function(keeper)
#     if keeper.convection == 1:
#         computer.interpolate_entropy(keeper)
#         computer.interpolate_phase_state(keeper)
#     computer.calculate_mean_opacities(keeper)
#     computer.integrate_beamflux(keeper)

#     # ---------------- BACK TO HOST --------------#
#     # copy everything from the GPU back to host and write output quantities to files
#     keeper.copy_device_to_host()
#     hsfunc.calculate_conv_flux(keeper)
#     hsfunc.calc_F_ratio(keeper)


# def write_helios(reader, keeper, computer, writer, plotter, fogger):
#     writer.create_output_dir_and_copy_param_file(reader, keeper)
#     writer.write_colmass_mu_cp_entropy(keeper, reader)
#     writer.write_integrated_flux(keeper, reader)
#     writer.write_downward_spectral_flux(keeper, reader)
#     writer.write_upward_spectral_flux(keeper, reader)
#     writer.write_TOA_flux_eclipse_depth(keeper, reader)
#     writer.write_direct_spectral_beam_flux(keeper, reader)
#     writer.write_planck_interface(keeper, reader)
#     writer.write_planck_center(keeper, reader)
#     writer.write_tp(keeper, reader)
#     writer.write_tp_cut(keeper, reader)
#     writer.write_opacities(keeper, reader)
#     writer.write_cloud_mixing_ratio(keeper, reader)
#     writer.write_cloud_opacities(keeper, reader)
#     writer.write_Rayleigh_cross_sections(keeper, reader)
#     writer.write_cloud_scat_cross_sections(keeper, reader)
#     writer.write_g_0(keeper, reader)
#     writer.write_transmission(keeper, reader)
#     writer.write_opt_depth(keeper, reader)
#     writer.write_cloud_opt_depth(keeper, reader)
#     writer.write_trans_weight_function(keeper, reader)
#     writer.write_contribution_function(keeper, reader)
#     writer.write_mean_extinction(keeper, reader)
#     writer.write_flux_ratio_only(keeper, reader)
#     writer.write_phase_state(keeper, reader)
#     writer.write_surface_albedo(keeper, reader)
#     writer.write_criterion_warning_file(keeper, reader)

#     if keeper.coupling == 1:
#         writer.write_tp_for_coupling(keeper, reader)
#         hsfunc.calculate_coupling_convergence(keeper, reader)

#     if keeper.approx_f == 1:
#         hsfunc.calc_tau_lw_sw(keeper, reader)

#     # prints the success message - yay!
#     hsfunc.success_message(keeper)


# def run_phaethon(
#     melt_wt_comp,
#     Delta_IW,
#     lavaplanet,
#     star,
#     outdir,
#     opac_species,
#     fO2_volatiles = {},
#     scatterers={},
#     do_background=False,
#     vaporization_code="muspell",
#     opacity_path="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/",
#     t_internal=None,
#     f_factor=1.0,
# ):
#     # --------- Is 'outdir' ok? ------#
#     if outdir == "":
#         outdir = "./"
#     elif outdir[-1] != "/":
#         outdir += "/"

#     if not os.path.isdir(outdir):
#         os.makedirs(outdir, exist_ok=True)

#     # TODO: make 'use_star_spec' usable
#     T_subst, T_eq = calc_Temp(lavaplanet, star, use_star_spec=False)
#     T_melt = T_subst

#     # if user specifies internal temperature, use as melt temperature instead
#     if t_internal is not None:
#         T_melt = t_internal
#         if t_internal < 1500:
#             raise Warning(
#                 "Melt tmeperature is dangerously low. These could destabilize the code."
#             )
#     else:
#         t_internal = 30.0

#     # TODO: make vaporock usable & adjust MAGMA pressures
#     print(f"Vaporizing melt @{T_melt} K")
#     mol_elem_frac, Ptotal, logfO2, cmelt = run_vaporize(
#         T_melt,
#         melt_wt_comp,
#         Delta_IW,
#         vaporization_code,
#         fO2_volatiles,
#     )
#     t_melt_trace = [T_melt]

#     run_fastchem_on_grid(mol_elem_frac, outdir)
#     lavaplanet.Ptotal = Ptotal
#     lavaplanet.substellar_temperature = T_subst
#     lavaplanet.equilibrium_temperature = T_eq

#     outfile = outdir + "/species_iterative.dat"
#     write_atmospecies_file(outfile, opac_species, scatterers)

#     # ----------- initial HELIOS run ------------#
#     reader, keeper, computer, writer, plotter, fogger = set_up_helios(
#         lavaplanet,
#         star,
#         outdir,
#         run_type="iterative",
#         opacity_path=opacity_path,
#         t_internal=t_internal,
#         f_factor=f_factor,
#     )
#     loop_helios(reader, keeper, computer, writer, plotter, fogger)

#     # ............. HELIOS loop .................#
#     T_abstol = 35.0
#     Delta_T_melt = abs(T_melt - keeper.T_lay[keeper.nlayer])
#     T_melt = keeper.T_lay[keeper.nlayer]

#     t_melt_trace.append(T_melt)

#     try_second_loop = False

#     while Delta_T_melt > T_abstol:
#         mol_elem_frac, Ptotal, logfO2, cmelt = run_vaporize(
#             T_melt,
#             melt_wt_comp,
#             Delta_IW,
#             vaporization_code,
#             fO2_volatiles,
#         )

#         run_fastchem_on_grid(mol_elem_frac, outdir)

#         keeper.p_boa = Ptotal / 1e-6
#         loop_helios(reader, keeper, computer, writer, plotter, fogger)

#         Delta_T_melt = abs(T_melt - keeper.T_lay[keeper.nlayer])
#         T_melt = keeper.T_lay[keeper.nlayer]

#         t_melt_trace.append(T_melt)

#         if (
#             len(t_melt_trace) >= 3
#             and abs(t_melt_trace[-3] - t_melt_trace[-1]) <= T_abstol
#         ):
#             print("Melt temperature series seems to be oscillating")
#             try_second_loop = True
#             break

#     # ----------- if oscillating, try branch-and-bound --------#
#     if try_second_loop:
#         search_range = [np.amin(t_melt_trace), np.amax(t_melt_trace)]

#         while abs(Delta_T_melt) > T_abstol:
#             # ..... run helios .....#
#             T_melt = (search_range[0] + search_range[1]) / 2.0

#             print(f"Vaporizing melt @{T_melt} K")
#             mol_elem_frac, Ptotal, logfO2, cmelt = run_vaporize(
#                 T_melt,
#                 melt_wt_comp,
#                 Delta_IW,
#                 vaporization_code,
#                 fO2_volatiles,
#             )

#             run_fastchem_on_grid(mol_elem_frac, outdir)

#             keeper.p_boa = Ptotal / 1e-6
#             loop_helios(reader, keeper, computer, writer, plotter, fogger)

#             Delta_T_melt = T_melt - keeper.T_lay[keeper.nlayer]

#             if Delta_T_melt < 0:  # melt cooler than T_BOA
#                 search_range[0] = T_melt
#             elif Delta_T_melt > 0:  # melt hotter than T_BOA
#                 search_range[1] = T_melt

#     # update melt parameters
#     lavaplanet.Ptotal = Ptotal
#     lavaplanet.T_melt = T_melt

#     store_metadata(
#         lavaplanet,
#         star,
#         cmelt,
#         logfO2,
#         Delta_IW,
#         vaporization_code,
#         mol_elem_frac,
#         outdir,
#         opac_species,
#         scatterers,
#         keeper.f_factor,
#     )

#     write_helios(reader, keeper, computer, writer, plotter, fogger)

#     # ---------- final FastChem run -----------#
#     filename = outdir + "HELIOS_iterative/tp.dat"

#     # some input values for temperature (in K) and pressure (in bar)
#     df = pd.read_csv(filename, header=1, delim_whitespace=True)
#     temperature = df["temp.[K]"].to_numpy(float)
#     pressure = df["press.[10^-6bar]"].to_numpy(float) * 1e-6

#     run_fastchem_on_PTprofile(pressure, temperature, outdir)
