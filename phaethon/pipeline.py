import numpy as np
import pandas as pd
import os
import json
import scipy.constants as sc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import sys

# from vaporock import System as vapoSystem
# from pyMAGMA import MAGMA
sys.path.append("/home/fabian/LavaWorlds/muspell/")
from muspell import MeltVaporSystem, VaporSystem, IdealGasMixture

import pyfastchem
from save_output import (
    saveChemistryOutput,
    saveMonitorOutput,
    saveChemistryOutputPandas,
    saveMonitorOutputPandas,
)
from to_helios_and_back import write_fastchem_input
from planet_spectrum import Emission_Reflection_Model

# ----- startools -----#
import star_tool.functions as startool_fn

# ----- HELIOS imports -----#
from HELIOS import read
from HELIOS import quantities as quant
from HELIOS import host_functions as hsfunc
from HELIOS import write
from HELIOS import computation as comp
from HELIOS import realtime_plotting as rt_plot
from HELIOS import clouds
from HELIOS import additional_heating as add_heat

# ============== Constants ================#
Mearth = 5.974e24
Rearth = 6.371e6
Rsun = 6.969e8
Msun = 1.989e30
Rjup = 6.9911e7


# =============== ESTIMATE SUBSTELLAR TEMPERATURE ================#
def calc_Temp(lavaplanet, star, use_star_spec=True):
    """
    Calculates the substellar & equilibrium temperature of the planet

    Parameters
    ----------
    lavaplanet : lavaplanet class
        contains all information about the planet
    star : star class
        contains all info about the star
    use_star_spec : bool
        (optional) use the actual stellar spectrum (if provided in "star"), otherwise blackbody
    """

    params = {
        "R_planet": lavaplanet.R,
        "semi_major_axis": lavaplanet.semi_major_axis,
        "Tmap": lavaplanet.Tmap_type,
        "R_star": star.R,
        "T_eff": star.T_eff,
        "distance": star.distance,
    }
    # if star.star_spec is not None and use_star_spec==True:
    # params['star_spec'] = star.star_spec

    E = Emission_Reflection_Model({"BB": 1}, params)
    T_subs = E.find_Tsub()
    T_eq = E.find_Teq()

    return T_subs, T_eq


def solve_for_d(T, lavaplanet, star, use_star_spec=True):
    params = {
        "R_planet": lavaplanet.R,
        "semi_major_axis": lavaplanet.semi_major_axis,
        "Tmap": lavaplanet.Tmap_type,
        "R_star": star.R,
        "T_eff": star.T_eff,
        "distance": star.distance,
    }
    # if star.star_spec is not None and use_star_spec==True:
    # params['star_spec'] = star.star_spec

    E = Emission_Reflection_Model({"BB": 1}, params)

    def f(d):
        E.semi_major_axis = d
        return E.find_Tsub() - T

    return brentq(f, 0.0001 * sc.astronomical_unit, 1000 * sc.astronomical_unit)


# =================== RUN VAPOROCK ==================#
def run_vaporize(
    T: float,
    melt_comp_wts: dict,
    Delta_IW: float,
    vaporization_code: str = "vaporock",
    fO2_volatiles = {},
) -> tuple[dict, float]:
    """
    Parameters
    ----------
    T : float
        Temperature of the melt
    melt_comp_wts : dict
        melt composition, in wt% (no need to normalize - done automatically)
    vaporization_code : str
        Either "vaporock" or "MAGMA"
    p_carbon_species : float
        pressure of CO & CO2 atmosphere, in bar
    """

    if vaporization_code == "vaporock" or vaporization_code == "VapoRock":
        system = vapoSystem()
        try:
            system.set_melt_comp(melt_comp_wts)
        except:
            raise ValueError(
                "VapoRock cannot find a solution for the given composition: ",
                melt_comp_wts,
            )

        P = 1e-10
        logfO2 = system.redox_buffer(T, P, buffer="IW") + Delta_IW

        logP = system.eval_gas_abundances(T, logfO2)
        mol_elem_frac, Ptotal = system.calc_gas_props(T, logP)

        #   get only interesting part
        mol_elem_frac = mol_elem_frac[T].to_dict()

        #   total pressure of atmosphere
        Ptotal = np.sum(10**logP, axis=0)[T]

    elif vaporization_code == "MAGMA":
        system = MAGMA()
        P, _ = system.run(T, melt_comp_wts)
        Ptotal = float(P.sum())
        mol_elem_frac = system.gas_info["mol%"].to_dict()
        logfO2 = 0

        del mol_elem_frac["Zn"]
        # mol_elem_frac['Cr'] = 1e-8
        # mol_elem_frac['Ti'] = 1e-8

    elif vaporization_code == "muspell":
        M = MeltVaporSystem(version="2023")
        V = VaporSystem()
        logP_mineral = M.run_magma(T, melt_comp_wts, Delta_IW)

        vapor_species_logp = V.form_vapor(
            temperature=T,
            logP_mineral=logP_mineral,
            logf_volatiles_ref=pd.Series(fO2_volatiles),
        )

        vapour = IdealGasMixture(10**vapor_species_logp.sort_index())

        print(vapour.log_p)

        mol_elem_frac = vapour.elem_molfrac.to_dict()
        print(mol_elem_frac)

        Ptotal = vapour.p_total
        logfO2 = M.logfO2

    else:
        raise NotImplementedError(
            'The only vaporization codes available are "VapoRock" and "MAGMA".'
        )

    return mol_elem_frac, Ptotal, logfO2, M.cmelt


# =================== ADDITIONAL ATMOSPHERIC GASES =====================#
# def add_volatiles(mol_elem_frac, logfO2):


# ======================== RUN FASTCHEM =========================#
def run_fastchem_on_grid(mol_elem_frac, outdir="./", monitor=True, verbose=True):
    """
    Gas speciation calculation with FastChem

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


def run_fastchem_on_PTprofile(
    pressure, temperature, outdir="./", monitor=True, verbose=True
):
    """
    Gas speciation calculation with FastChem

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


# =============== STORE METADATA =====================#
def store_metadata(
    lavaplanet,
    star,
    cmelt,
    logfO2,
    dIW,
    vaporcode,
    mol_elem_frac,
    outdir,
    atmo_species,
    scatterers,
    f_factor,
):
    metadata = {
        "name": lavaplanet.name,
        "vaporcode": vaporcode,
        "planet_params": {
            "M (Mearth)": lavaplanet.M / Mearth,
            "R (Rearth)": lavaplanet.R / Rearth,
            "semi_major_axis (AU)": lavaplanet.semi_major_axis / sc.astronomical_unit,
            "Tmap_type": lavaplanet.Tmap_type,
            "substellar_temperature": lavaplanet.substellar_temperature,
            "equilibrium_temperature": lavaplanet.equilibrium_temperature,
        },
        "radiation": {
            "f_factor": f_factor,
        },
        "star_params": {
            "M (Msun)": star.M / Msun,
            "R (Rsun)": star.R / Rsun,
            "T_eff": star.T_eff,
            "distance (pc)": star.distance / sc.parsec,
        },
        "P_BOA (bar)": lavaplanet.Ptotal,
        "T_melt (K)": lavaplanet.T_melt,
        "atmo_comp (mol%)": mol_elem_frac,
        "logfO2": logfO2,
        "dIW": dIW,
        "melt_comp (wt%)": (cmelt["wt%"]).to_dict(),
        "melt_comp (mol%)": (cmelt["mol%"]).to_dict(),
        "atmo_species (HELIOS)": list(atmo_species),
        "atmo_scatterers (HELIOS)": list(scatterers),
    }

    metadata_file = open(outdir + "metadata.json", "w")
    json.dump(metadata, metadata_file)

    metadata_file.close()


# ================ GENERATE ATMOSPECIES FILE (for HELIOS) =======#


def write_atmospecies_file(
    outfile: str, atmo_species: set, scatterers: set = {"e-", "O2"}
):
    """
    Generates atmospheric species files, which HELIOS will look up

    Parameters
    ----------
    atmo_molar_chem : set
        contains the molar fraction of elements in the planets atmosphere, i.e. {'Si':0.34, 'O':0.56, 'Na':0.07, 'Fe':0.03}. Does not have to be normalized.
    ignore_species : set
        species to ignore, i.e. {'TiO', 'Cr'}
    """

    # --------- write present species into file --------#
    with open(outfile, "w") as species_dat:
        species_dat.write("species     absorbing  scattering  mixing_ratio\n\n")

        for species in atmo_species.union(scatterers):
            outstr = species + " " * (13 - len(species))

            # species does absorb
            if species in atmo_species:
                outstr += "yes" + " " * 9
            else:
                outstr += "no" + " " * 10

            if species in scatterers:
                outstr += "yes" + " " * 9
            else:
                outstr += "no" + " " * 10

            outstr += "FastChem"
            outstr += "\n\n"

            species_dat.write(outstr)


# ==================== INPUT OBJECTS =======================#
class Star(object):
    def __init__(self, params: dict):
        self.name = params.get("name", "Star").replace(" ", "")
        self.M = params.get("M", np.nan)
        self.R = params.get("R")
        self.T_eff = params.get("T_eff")
        self.distance = params.get("distance")
        self.metallicity = params.get("metallicity", np.nan)

        # derived params
        self.logg = np.log(sc.G * self.M / (self.R**2))

        # default init
        self.file_or_BB = "blackbody"
        self.source_file = None
        self.star_spec = None

    @property
    def file_or_BB(self):
        return self._file_or_BB

    @file_or_BB.setter
    def file_or_BB(self, _file_or_BB):
        if _file_or_BB not in ["file", "blackbody"]:
            raise ValueError(
                '"file_or_BB" must be either "file" (for detailed stellar spectra) or "blackbody"'
            )
        self._file_or_BB = _file_or_BB

    def calc_from_file(
        self,
        data_format: str,
        source_file: str,
        outdir: str,
        plot_and_tweak: bool = False,
        skiprows: int = 0,
        R=200,
        w_conversion_factor=1,
        flux_conversion_factor=1,
    ):
        # --------- Is 'outdir' ok? ------#
        if outdir == "":
            outdir = "./"
        elif outdir[-1] != "/":
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # ---------- star spec source file ------------#
        if source_file is not None:
            self.source_file = source_file

        # ----------- run the HELIOS star-tool ---------#
        startool_params = {
            "data_format": data_format,
            "source_file": self.source_file,
            "name": self.name,
            "w_conversion_factor": w_conversion_factor,
            "flux_conversion_factor": flux_conversion_factor,
            "temp": self.T_eff,
            "distance_from_Earth": self.distance / sc.parsec,
            "R_star": self.R / Rsun,
        }
        startool_output_file = outdir + self.name + ".h5"
        orig_lambda, orig_flux, new_lambda, converted_flux = startool_fn.main_loop(
            startool_params,
            convert_to="r50_kdistr",
            # TODO implement automatich lookup of lambdagrid for star
            opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/SiO_opac_ip_kdistr.h5",
            output_file=startool_output_file,
            plot_and_tweak="no" if not plot_and_tweak else "yes",
            save_in_hdf5="yes",
            skiprows=skiprows,
        )

        self.file_or_BB = "file"
        self.path_to_h5 = startool_output_file
        self.path_in_h5 = "r50_kdistr/ascii/" + self.name

        self.orig_lambda = np.array(orig_lambda) * 1e4  # from Angstroem to micron
        self.new_lambda = np.array(new_lambda) * 1e4
        self.orig_flux = np.array(orig_flux) * 1e-13
        self.new_flux = np.array(converted_flux) * 1e-13
        self.star_spec = interp1d(
            self.orig_lambda, self.orig_flux, bounds_error=False, fill_value=0.0
        )

    def calc_from_phoenix(
        self,
        outdir: str,
        plot_and_tweak: bool = False,
        skiprows: int = 0,
        R=200,
        w_conversion_factor=1,
        flux_conversion_factor=1,
    ):
        # --------- Is 'outdir' ok? ------#
        if outdir == "":
            outdir = "./"
        elif outdir[-1] != "/":
            outdir += "/"

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # ----------- run the HELIOS star-tool ---------#
        startool_params = {
            "data_format": "phoenix",
            "name": self.name,
            "temp": self.T_eff,
            "log_g": self.logg,
            "m": self.metallicity,
        }

        startool_output_file = outdir + self.name + ".h5"
        orig_lambda, orig_flux, new_lambda, converted_flux = startool_fn.main_loop(
            startool_params,
            convert_to="r50_kdistr",
            # TODO implement automatich lookup of lambdagrid for star
            opac_file_for_lambdagrid="/home/fabian/LavaWorlds/phaethon/ktable/output/R"
            + str(R)
            + "/SiO_opac_ip_kdistr.h5",
            output_file=startool_output_file,
            plot_and_tweak="no" if not plot_and_tweak else "yes",
            save_in_hdf5="yes",
            skiprows=skiprows,
        )

        self.file_or_BB = "file"
        self.path_to_h5 = startool_output_file
        self.path_in_h5 = "r50_kdistr/ascii/" + self.name

        self.orig_lambda = np.array(orig_lambda) * 1e4  # from Angstroem to micron
        self.new_lambda = np.array(new_lambda) * 1e4
        self.orig_flux = np.array(orig_flux) * 1e-13
        self.new_flux = np.array(converted_flux) * 1e-13
        self.star_spec = interp1d(
            self.orig_lambda, self.orig_flux, bounds_error=False, fill_value=0.0
        )


class Lavaplanet(object):
    def __init__(self, params: dict):
        self.name = params.get("name", "NoName")
        self.M = params.get("M")
        self.R = params.get("R")
        self.g = sc.G * self.M / (self.R**2)
        self.semi_major_axis = params.get("semi_major_axis")
        self.Tmap_type = params.get("Tmap_type")


# ================= RUN HELIOS ========================#
def set_up_helios(
    lavaplanet,
    star,
    outdir,
    do_background=False,
    run_type="iterative",
    opacity_path="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/",
    t_internal=30,
    f_factor=1.0,
):
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
    reader.read_param_file_and_command_line(keeper, fogger)

    reader.output_path = outdir

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
    if do_background == True:
        reader.species_file = "phaethon/species_minimalist.dat"
    elif keeper.run_type == "post-processing":
        reader.species_file = (
            outdir + "/species_iterative.dat"
        )  # currently it seems that species for iterative and post-processing have to be the same
        reader.temp_path = outdir + "HELIOS_iterative/tp.dat"
        reader.temp_format = "helios"
    else:
        reader.species_file = outdir + "/species_iterative.dat"

    # specify opacity path
    reader.opacity_path = opacity_path

    # pass pressure, convert to weird HELIOS units (10⁻⁶ bar)
    keeper.p_boa = lavaplanet.Ptotal / 1e-6
    keeper.p_toa = 1e-8 / 1e-6

    # path to FastChem results
    reader.fastchem_path = outdir

    if keeper.opacity_mixing == "premixed":
        reader.load_premixed_opacity_table(keeper)

    elif keeper.opacity_mixing == "on-the-fly":
        reader.read_species_file(keeper)
        reader.read_species_opacities(keeper)
        reader.read_species_scat_cross_sections(keeper)
        reader.read_species_mixing_ratios(keeper)
    else:
        raise NotImplementedError

    reader.read_kappa_table_or_use_constant_kappa(keeper)
    reader.read_or_fill_surf_albedo_array(keeper)
    keeper.dimensions()

    # -------- read stellar spectru -----#
    reader.stellar_model = star.file_or_BB

    if star.file_or_BB == "file":
        reader.stellar_path = star.path_to_h5
        reader.stellar_data_set = star.path_in_h5

    #   finish
    reader.read_star(keeper)

    # ----------- planetary & orbital parameters --------#
    # read planetary params (for code stability only), modify them and turn them to cgs
    # hsfunc.planet_param(keeper, reader)
    keeper.g = keeper.fl_prec(lavaplanet.g * 100)
    keeper.a = keeper.fl_prec(lavaplanet.semi_major_axis * 100)
    keeper.R_planet = keeper.fl_prec(lavaplanet.R * 100)
    keeper.R_star = keeper.fl_prec(star.R * 100)
    keeper.T_star = keeper.fl_prec(star.T_eff)
    keeper.T_intern = float(t_internal)

    # ---------- radiation --------#
    # keeper.f_factor = float(f_factor)

    return reader, keeper, computer, writer, plotter, fogger


def loop_helios(reader, keeper, computer, writer, plotter, fogger, init_T_profile=None):
    hsfunc.set_up_numerical_parameters(keeper)
    hsfunc.construct_grid(keeper)
    hsfunc.initial_temp(keeper, reader, init_T_profile)

    if keeper.approx_f == 1 and keeper.planet_type == "rocky":
        hsfunc.approx_f_from_formula(keeper, reader)

    hsfunc.calc_F_intern(keeper)
    add_heat.load_heating_terms_or_not(keeper)

    fogger.cloud_pre_processing(keeper)

    # create, convert and copy arrays to be used in the GPU computations
    keeper.create_zero_arrays()
    keeper.convert_input_list_to_array()
    keeper.copy_host_to_device()
    keeper.allocate_on_device()

    # ---------- conduct core computations on the GPU --------------#
    computer.construct_planck_table(keeper)
    computer.correct_incident_energy(keeper)

    computer.radiation_loop(keeper, writer, reader, plotter)

    computer.convection_loop(keeper, writer, reader, plotter)

    computer.integrate_optdepth_transmission(keeper)
    computer.calculate_contribution_function(keeper)
    if keeper.convection == 1:
        computer.interpolate_entropy(keeper)
        computer.interpolate_phase_state(keeper)
    computer.calculate_mean_opacities(keeper)
    computer.integrate_beamflux(keeper)

    # ---------------- BACK TO HOST --------------#
    # copy everything from the GPU back to host and write output quantities to files
    keeper.copy_device_to_host()
    hsfunc.calculate_conv_flux(keeper)
    hsfunc.calc_F_ratio(keeper)


def write_helios(reader, keeper, computer, writer, plotter, fogger):
    writer.create_output_dir_and_copy_param_file(reader, keeper)
    writer.write_colmass_mu_cp_entropy(keeper, reader)
    writer.write_integrated_flux(keeper, reader)
    writer.write_downward_spectral_flux(keeper, reader)
    writer.write_upward_spectral_flux(keeper, reader)
    writer.write_TOA_flux_eclipse_depth(keeper, reader)
    writer.write_direct_spectral_beam_flux(keeper, reader)
    writer.write_planck_interface(keeper, reader)
    writer.write_planck_center(keeper, reader)
    writer.write_tp(keeper, reader)
    writer.write_tp_cut(keeper, reader)
    writer.write_opacities(keeper, reader)
    writer.write_cloud_mixing_ratio(keeper, reader)
    writer.write_cloud_opacities(keeper, reader)
    writer.write_Rayleigh_cross_sections(keeper, reader)
    writer.write_cloud_scat_cross_sections(keeper, reader)
    writer.write_g_0(keeper, reader)
    writer.write_transmission(keeper, reader)
    writer.write_opt_depth(keeper, reader)
    writer.write_cloud_opt_depth(keeper, reader)
    writer.write_trans_weight_function(keeper, reader)
    writer.write_contribution_function(keeper, reader)
    writer.write_mean_extinction(keeper, reader)
    writer.write_flux_ratio_only(keeper, reader)
    writer.write_phase_state(keeper, reader)
    writer.write_surface_albedo(keeper, reader)
    writer.write_criterion_warning_file(keeper, reader)

    if keeper.coupling == 1:
        writer.write_tp_for_coupling(keeper, reader)
        hsfunc.calculate_coupling_convergence(keeper, reader)

    if keeper.approx_f == 1:
        hsfunc.calc_tau_lw_sw(keeper, reader)

    # prints the success message - yay!
    hsfunc.success_message(keeper)


def is_oscillating_series(series, precision):
    """
    Determines wherever a series of floats is pseudo-oscillating, i.e. oscillating up to precison.
    """
    arr = np.array(series)

    odd_diff = np.diff(arr[::2])  # Differences between odd-indexed elements
    even_diff = np.diff(arr[1::2])  # Differences between even-indexed elements

    if (
        np.all(np.abs(odd_diff) <= precision)
        and np.all(np.abs(even_diff) <= precision)
        and np.abs(np.sum(odd_diff) - np.sum(even_diff)) <= 2 * precision
    ):
        return True

    return False


def run_phaethon(
    melt_wt_comp,
    Delta_IW,
    lavaplanet,
    star,
    outdir,
    atmo_species,
    fO2_volatiles = {},
    scatterers={},
    do_background=False,
    vaporization_code="muspell",
    opacity_path="/home/fabian/LavaWorlds/phaethon/ktable/output/R200_0.1_200_pressurebroad/",
    t_internal=None,
    f_factor=1.0,
):
    # --------- Is 'outdir' ok? ------#
    if outdir == "":
        outdir = "./"
    elif outdir[-1] != "/":
        outdir += "/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # TODO: make 'use_star_spec' usable
    T_subst, T_eq = calc_Temp(lavaplanet, star, use_star_spec=False)
    T_melt = T_subst

    # if user specifies internal temperature, use as melt temperature instead
    if t_internal is not None:
        T_melt = t_internal
        if t_internal < 1500:
            raise Warning(
                "Melt tmeperature is dangerously low. These could destabilize the code."
            )
    else:
        t_internal = 30.0

    # TODO: make vaporock usable & adjust MAGMA pressures
    print(f"Vaporizing melt @{T_melt} K")
    mol_elem_frac, Ptotal, logfO2, cmelt = run_vaporize(
        T_melt,
        melt_wt_comp,
        Delta_IW,
        vaporization_code,
        fO2_volatiles,
    )
    t_melt_trace = [T_melt]

    run_fastchem_on_grid(mol_elem_frac, outdir)
    lavaplanet.Ptotal = Ptotal
    lavaplanet.substellar_temperature = T_subst
    lavaplanet.equilibrium_temperature = T_eq

    outfile = outdir + "/species_iterative.dat"
    write_atmospecies_file(outfile, atmo_species, scatterers)

    # ----------- initial HELIOS run ------------#
    reader, keeper, computer, writer, plotter, fogger = set_up_helios(
        lavaplanet,
        star,
        outdir,
        run_type="iterative",
        opacity_path=opacity_path,
        t_internal=t_internal,
        f_factor=f_factor,
    )
    loop_helios(reader, keeper, computer, writer, plotter, fogger)

    # ............. HELIOS loop .................#
    T_abstol = 35.0
    Delta_T_melt = abs(T_melt - keeper.T_lay[keeper.nlayer])
    T_melt = keeper.T_lay[keeper.nlayer]

    t_melt_trace.append(T_melt)

    try_second_loop = False

    while Delta_T_melt > T_abstol:
        mol_elem_frac, Ptotal, logfO2, cmelt = run_vaporize(
            T_melt,
            melt_wt_comp,
            Delta_IW,
            vaporization_code,
            fO2_volatiles,
        )

        run_fastchem_on_grid(mol_elem_frac, outdir)

        keeper.p_boa = Ptotal / 1e-6
        loop_helios(reader, keeper, computer, writer, plotter, fogger)

        Delta_T_melt = abs(T_melt - keeper.T_lay[keeper.nlayer])
        T_melt = keeper.T_lay[keeper.nlayer]

        t_melt_trace.append(T_melt)

        if (
            len(t_melt_trace) >= 3
            and abs(t_melt_trace[-3] - t_melt_trace[-1]) <= T_abstol
        ):
            print("Melt temperature series seems to be oscillating")
            try_second_loop = True
            break

    # ----------- if oscillating, try branch-and-bound --------#
    if try_second_loop:
        search_range = [np.amin(t_melt_trace), np.amax(t_melt_trace)]

        while abs(Delta_T_melt) > T_abstol:
            # ..... run helios .....#
            T_melt = (search_range[0] + search_range[1]) / 2.0

            print(f"Vaporizing melt @{T_melt} K")
            mol_elem_frac, Ptotal, logfO2, cmelt = run_vaporize(
                T_melt,
                melt_wt_comp,
                Delta_IW,
                vaporization_code,
                fO2_volatiles,
            )

            run_fastchem_on_grid(mol_elem_frac, outdir)

            keeper.p_boa = Ptotal / 1e-6
            loop_helios(reader, keeper, computer, writer, plotter, fogger)

            Delta_T_melt = T_melt - keeper.T_lay[keeper.nlayer]

            if Delta_T_melt < 0:  # melt cooler than T_BOA
                search_range[0] = T_melt
            elif Delta_T_melt > 0:  # melt hotter than T_BOA
                search_range[1] = T_melt

    # update melt parameters
    lavaplanet.Ptotal = Ptotal
    lavaplanet.T_melt = T_melt

    store_metadata(
        lavaplanet,
        star,
        cmelt,
        logfO2,
        Delta_IW,
        vaporization_code,
        mol_elem_frac,
        outdir,
        atmo_species,
        scatterers,
        keeper.f_factor,
    )

    write_helios(reader, keeper, computer, writer, plotter, fogger)

    # ---------- final FastChem run -----------#
    filename = outdir + "HELIOS_iterative/tp.dat"

    # some input values for temperature (in K) and pressure (in bar)
    df = pd.read_csv(filename, header=1, delim_whitespace=True)
    temperature = df["temp.[K]"].to_numpy(float)
    pressure = df["press.[10^-6bar]"].to_numpy(float) * 1e-6

    run_fastchem_on_PTprofile(pressure, temperature, outdir)
