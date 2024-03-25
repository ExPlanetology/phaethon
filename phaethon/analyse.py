import numpy as np
import pandas as pd
import scipy.constants as sc
import json
from uncertainties import ufloat
import h5py
import astropy.units as unit
import astropy.constants as const

from astropy.units.core import Unit as AstropyUnit
from astropy.units.quantity import Quantity as AstropyQuantity


class PhaethonResult:
    def __init__(self, path, star_basepath=""):
        self.path = path
        if not self.path.endswith("/"):
            self.path += "/"

        # ------- load spectra ------#
        filename = self.path + "HELIOS_iterative/TOA_flux_eclipse.dat"
        df = pd.read_csv(filename, skiprows=2, delim_whitespace=True)
        self.wavl = df["cent_lambda[um]"].to_numpy() * unit.micron
        self.spectral_exitance_planet = df["F_up_at_TOA"].to_numpy() * (unit.erg / unit.s / (unit.cm**3))
        self.fpfs = df["planet/star"].to_numpy()  # turn into ppm
        self.t_bright = self.get_t_bright() * unit.K

        # ------ load pressure-temperature profile -------#
        filename = self.path + "HELIOS_iterative/tp.dat"
        df = pd.read_csv(filename, skiprows=1, delim_whitespace=True)
        self.temperature = df["temp.[K]"].to_numpy() * unit.K
        self.pressure = df["press.[10^-6bar]"].to_numpy() / 1e6 * unit.bar
        self.altitude = df["altitude[cm]"].to_numpy() * unit.cm

        # ------ load chemistry -----#
        filename = self.path + "chem_profile.dat"
        self.chem = pd.read_csv(filename, delim_whitespace=True)
        self.chem = self.chem.rename(
            columns={
                "#p(bar)": "P(bar)",
            }
        )
        # temperature = df['T(k)'].to_numpy(float)
        # pressure = df['#P(bar)'].to_numpy(float)
        # MMW = df['m(u)'].to_numpy(float)

        self.species = list(
            self.chem.drop(
                ["T(K)", "P(bar)", "n_<tot>(cm-3)", "n_g(cm-3)", "m(u)"], axis=1
            ).keys()
        )

        # ------------ planet params ------------#
        with open(self.path + "metadata.json") as f:
            params = json.load(f)

        self.planet_params = params["planet"]
        self.star_params = params["star"]
        self.orbit_params = params["orbit"]
        self.vapour_engine_params = params["vapour_engine"]

        # ----------- stellar spectrum ------------#
        try:
            with h5py.File(star_basepath + self.star_params["path_to_h5"], "r") as f:
                self.spectral_exitance_star = np.array(
                    f["r50_kdistr"]["ascii"][self.star_params["name"]]
                ) * (unit.erg / unit.s / (unit.cm**3))
                self.star_wavl = np.array(f["r50_kdistr"]["lambda"]) * unit.cm

        except FileNotFoundError:
            self.spectral_exitance_planet = None
            self.star_wavl = None

        # -------------- transmissivity ------------ #
        self.transmissivity = (
            pd.read_table(
                self.path + "/HELIOS_iterative/transmission.dat",
                skiprows=1,
                delim_whitespace=True,
                index_col=0,
            )
            .drop(["cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]"], axis=1)
            .to_numpy()
        )

        self.integrated_transmissivity = np.flip(
            np.cumprod(np.flip(self.transmissivity), axis=1)
        )

        # -------------- optical depth --------------- #
        self.optical_depth = (
            pd.read_table(
                self.path + "/HELIOS_iterative/optdepth.dat",
                skiprows=1,
                delim_whitespace=True,
                index_col=0,
            )
            .drop(["cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]"], axis=1)
            .to_numpy()
        )

        # ------------ contribution function -----------#
        self._contrib = pd.read_table(
            path + "/HELIOS_iterative/contribution.dat",
            skiprows=1,
            delim_whitespace=True,
            index_col=0,
        ).drop(
            ["cent_lambda[um]", "low_int_lambda[um]", "delta_lambda[um]"], axis=1
        ).to_numpy()

        self.contribution = self._contrib / self._contrib.sum(axis=1)[:,None]
        self.contribution = np.cumsum(np.flip(self.contribution), axis=1)
        self.contribution = np.flip(self.contribution)


    def get_photospheric_pressurelevel(self, photosphere_level=0.8, smoothing_window_size=11):

        def moving_average(arr, window_size):
            return np.convolve(arr, np.ones(window_size), 'same') / window_size

        mask = np.argmin(abs(self.integrated_transmissivity - photosphere_level), axis=1)
        y = self.pressure[1:][mask]

        return moving_average(y, smoothing_window_size)

    def get_t_bright(self):
        """
        Parameters
        ----------
            wavl : numpy array
                wavelength, in micron
            flux : numpy array
                flux from planet, in erg s^-1 cm^-3
        """
        wavl = self.wavl.copy().to("micron").value
        flux = self.spectral_exitance_planet.copy().value

        # to SI
        wavl *= 1e-6
        flux *= 1 / np.pi * 0.1

        return (
            sc.h
            * sc.c
            / (sc.k * wavl)
            / (np.log(1 + (2 * sc.h * sc.c**2) / (flux * wavl**5)))
        )

    def get_contrast_of_fluxes(self, wavl_bin1=[5.0, 6.7], wavl_bin2=[7.5, 10.0]):
        return compute_contrast_of_fluxes(
            self.wavl.value,
            self.fpfs,
            np.zeros_like(self.fpfs),
            wavl_bin1,
            wavl_bin2,
        )[0]

    def get_bins(self, wavl_pairs: list):
        return get_bins(self.wavl.value, self.fpfs, np.zeros_like(self.fpfs), wavl_pairs)

    def calc_fpfs(self, pl_radius: AstropyUnit) -> AstropyUnit:
        fpfs_calculated = (
                    self.spectral_exitance_planet
                    / self.spectral_exitance_star
                    * (
                        pl_radius.to("m")
                        / (self.star_params["radius"] * unit.R_sun.to("m"))
                    )
                    ** 2
                )
        return fpfs_calculated


# ======== binning function =========#
def bin_it(wavl, spec, err, wavl_dn, wavl_up):
    """
    wavl : np.ndarray
        wavelength, in micron
    """
    assert wavl_up > wavl_dn
    mask = (wavl > wavl_dn) * (wavl < wavl_up)
    N_bins = np.where(mask == True)[0].size
    wavl = wavl[mask]
    spec = spec[mask]
    err = err[mask]

    S = np.sum(spec) / N_bins
    sig = np.sqrt(np.sum(err**2)) / N_bins
    wavl_err = (wavl_up - wavl_dn) / 2.0
    wavl_mean = wavl_err + wavl_dn

    return wavl_mean, wavl_err, S, sig, N_bins


def get_bins(wavl, spec, err, wavl_pairs: list):
    x = []
    y = []
    xerr = []
    yerr = []

    for wavl_pair in wavl_pairs:
        wavl_mean, wavl_err, S, sig, N_bins = bin_it(
            wavl, spec, err, wavl_pair[0], wavl_pair[1]
        )
        x.append(wavl_mean)
        xerr.append(wavl_err)
        y.append(S)
        yerr.append(sig)

    return x, y, xerr, yerr


def bin_adjacent_bins(wavl, spec, err, nbins):
    # Calculate the total number of chunks
    num_chunks = len(wavl) // nbins + (len(wavl) % nbins > 0)

    # Slice the array into chunks
    chunks = [wavl[i * nbins : (i + 1) * nbins] for i in range(num_chunks)]

    x = []
    y = []
    xerr = []
    yerr = []
    for chunk in chunks:
        if len(chunk) > 1:
            wavl_mean, wavl_err, S, sig, N_bins = bin_it(
                wavl, spec, err, np.amin(chunk), np.amax(chunk)
            )
            x.append(wavl_mean)
            xerr.append(wavl_err)
            y.append(S)
            yerr.append(sig)
        else:
            pass

    return x, y, xerr, yerr


def compute_contrast_of_fluxes(
    real_lamda,
    real_fpfs,
    real_err,
    wavl_bin1=[5, 6.7],
    wavl_bin2=[7.5, 10.0],
):
    """Contrast of fluxratios: 8 micron / 6 micron"""

    _, _, bin_6mu, sig_6mu, N_bins6 = bin_it(
        real_lamda, real_fpfs, real_err, wavl_bin1[0], wavl_bin1[1]
    )
    _, _, bin_8mu, sig_8mu, N_bins8 = bin_it(
        real_lamda, real_fpfs, real_err, wavl_bin2[0], wavl_bin2[1]
    )

    # data_con_of_con = bin_8mu - bin_6mu
    # data_sigma_con = np.sqrt(sig_8mu**2 + sig_6mu**2)
    u6 = ufloat(bin_6mu, sig_6mu)
    u8 = ufloat(bin_8mu, sig_8mu)
    c = u8 / u6

    return c.nominal_value, c.std_dev


if __name__ == "__main__":
    dIW = -4
    path = "/home/fabian/LavaWorlds/phaethon/output/test/55Cnce/".format(dIW)

    data = PhaethonResult(path)
