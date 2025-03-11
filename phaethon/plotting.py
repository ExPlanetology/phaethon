# 
# Copyright 2024-2025 Fabian L. Seidler
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
Module with plotting utilities for a Phaethon result.
"""

from typing import Literal, Optional
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes,
)
from labellines import labelLines
import cmcrameri

from phaethon.analyse import PhaethonResult
from phaethon.utilities import formula_to_latex


def plot_chem(
    result: PhaethonResult,
    mixrat_limits: list,
    axis=None,
    molec_styles: Optional[dict] = None,
    lw: float = 2,
    legend_kws: Optional[dict] = None,
    use_mpl_log: bool = True,
    pressure_unit: str = "bar",
    use_labellines: bool = True,
    labellines_kwargs: Optional[dict] = None,
) -> None:
    """
    Plot mixing ratios as a function of pressure.

    Parameters
    ----------
        use_mpl_log : bool, default True
            If true, use matplotlibs internal log conversion.
            Otherwise, do log10(pressure).
    """

    # securely init dicts
    if molec_styles is None:
        molec_styles = {}
    if legend_kws is None:
        legend_kws = {}
    if labellines_kwargs is None:
        labellines_kwargs = {}

    # pressure temperature arrays
    result.chem["T(K)"].to_numpy(float)
    pressure = result.pressure.to(pressure_unit).value
    if not use_mpl_log:
        pressure = np.log10(pressure)

    # create axis (or not) ...
    if axis is None:
        fig, (ax, ax_legend) = plt.subplots(1, 2, width_ratios=[1.0, 0.2])
        ax_legend.axis("off")
    else:
        ax = axis
        ax_legend = ax

    if len(molec_styles) > 0:
        legend_handles = []
        for species in molec_styles.keys():
            if "_and_" in species:
                species_list: list = species.split("_and_")
                mixing_ratio = sum(
                    result.chem[_species] for _species in species_list
                ).to_numpy()
            else:
                mixing_ratio = result.chem[species].to_numpy()

            if not use_mpl_log:
                _mixing_ratio = np.log10(mixing_ratio)
            else:
                _mixing_ratio = mixing_ratio

            if np.any(mixing_ratio >= min(mixrat_limits)):
                mask = (mixing_ratio >= min(mixrat_limits)) * (
                    mixing_ratio <= max(mixrat_limits)
                )
                if np.any(mask):
                    ax.plot(
                        np.flip(_mixing_ratio[mask]),
                        np.flip(pressure[mask]),
                        label=molec_styles[species].get("label", species),
                        color=molec_styles[species].get("color", None),
                    )

                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            label=molec_styles[species].get("label", species),
                            color=molec_styles[species].get("color", None),
                            ls=molec_styles[species].get("ls", "solid"),
                            lw=molec_styles[species].get("lw", lw),
                        )
                    )
    else:
        legend_handles = []
        colors = [cmcrameri.cm.batlowS(i) for i in range(len(result.species))]
        for species, color in zip(result.species, colors):
            mixing_ratio = result.chem[species].to_numpy()
            if np.any(mixing_ratio >= min(mixrat_limits)):
                mask = (mixing_ratio >= min(mixrat_limits)) * (
                    mixing_ratio <= max(mixrat_limits)
                )

                if not use_mpl_log:
                    _mixing_ratio = np.log10(mixing_ratio)
                else:
                    _mixing_ratio = mixing_ratio

                ax.plot(
                    np.flip(_mixing_ratio[mask]),
                    np.flip(pressure[mask]),
                    label=formula_to_latex(species),
                    color=color,
                )

                legend_handles.append(
                    Line2D([0], [0], label=formula_to_latex(species), color=color, ls="solid", lw=2)
                )

    if use_mpl_log:
        ax.loglog()
        ax.set_ylabel(f"Pressure ({pressure_unit})")
        ax.set_xlabel("Volumetric mixing ratio")
        ax.set_xlim(min(mixrat_limits), max(mixrat_limits))
    else:
        ax.set_ylabel(r"$\log_{10}$ Pressure " + f"({pressure_unit})")
        ax.set_xlabel(r"$\log_{10}$ Volumetric mixing ratio")
        ax.set_xlim(np.log10(min(mixrat_limits)), np.log10(max(mixrat_limits)))

    # limits
    ax.set_ylim(ax.get_ylim()[::-1])

    # legend
    legend = ax_legend.legend(handles=legend_handles, frameon=False, **legend_kws)

    # grid
    ax.grid(True, ls="dotted", color="lightgrey")

    # labellines
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_labellines:
            try:
                labelLines(ax.get_lines(), **labellines_kwargs)
            except:
                pass

    # Figure is not part of a larger plot
    if axis is None:

        # Function to highlight the hovered line in the legend and the plot
        def on_move(event):
            # Check if the mouse is over the legend
            if legend.get_window_extent().contains(event.x, event.y):
                # Iterate over legend items
                for i, (text, line) in enumerate(zip(legend.texts, ax.lines)):
                    # Get the legend's label text and line artist
                    legend_line = legend.legend_handles[i]
                    label = text.get_text()

                    # Check if the mouse is over the legend handle
                    if legend_line.contains(event)[0]:
                        # Highlight the line in the plot
                        line.set_linewidth(4)
                        line.set_alpha(1.0)
                        line.set_zorder(3)

                        # Highlight the line in the legend
                        legend_line.set_linewidth(4)  # Make it thicker
                        text.set_fontweight("bold")  # Make the label bold
                    else:
                        # Reset the lines to their normal state when not hovered
                        legend_line.set_linewidth(2)
                        line.set_alpha(0.3)
                        line.set_zorder(0)
                        text.set_fontweight("normal")

                        # Reset the plot line style
                        line.set_linewidth(2)

                # Redraw the figure to update the changes
                fig.canvas.draw_idle()
            else:
                for i, (text, line) in enumerate(zip(legend.texts, ax.lines)):
                    line.set_alpha(1.0)
                    line.set_linewidth(2.0)

                # Redraw the figure to update the changes
                fig.canvas.draw_idle()

        # Connect the mouse motion event to the on_move function
        fig.canvas.mpl_connect("motion_notify_event", on_move)

        # Show
        plt.tight_layout()
        plt.show()


def plot_tau(
    result: PhaethonResult,
    axis=None,
    cmap: str = "cubehelix_r",
    photosphere_colour="w",
    xlim: Optional[list] = None,
    use_mpl_log: bool = True,
    wavl_unit: str = "micron",
    pressure_unit: str = "bar",
    **photosphere_kws,
):
    """Plot the optical depth as function of atmospheric height"""

    if xlim is None:
        xlim = [
            np.amin(result.wavl.to("micron").value),
            np.amax(result.wavl.to("micron").value),
        ]

    # axis already defined? if not, make new
    if axis is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = axis

    # data
    pressure = result.pressure.to(pressure_unit).value
    if not use_mpl_log:
        pressure = np.log10(pressure)
    wavl = result.wavl.to(wavl_unit)

    wavl_grid, pressure_grid = np.meshgrid(wavl, pressure[1:])
    log10_tau = np.log10(result.optical_depth.T)
    log10_tau[log10_tau < -6] = -6
    log10_tau[log10_tau > 2] = 2.0

    cs = ax.contourf(
        wavl_grid,
        pressure_grid,
        log10_tau,
        levels=55,
        vmin=-6.1,
        vmax=1.1,
        cmap=cmap,
    )
    # for c in cs.collections:
    #     c.set_edgecolor("face")
    cbaxes = inset_axes(ax, width="30%", height="3%", loc="upper center")
    plt.colorbar(
        cs,
        cax=cbaxes,
        label=r"$\log_{10} \tau$",
        orientation="horizontal",
        ticks=[-6, -4, -2, 0, 2],
    )

    # plot photosphere
    for photo_p_level, ls in zip([0.159, 0.5, 0.841], ["dotted", "solid", "dotted"]):
        photo_pressure: np.ndarray = (
            result.get_photospheric_pressurelevel(
                absorbed_frac=photo_p_level, **photosphere_kws
            )
            .to(pressure_unit)
            .value
        )

        if not use_mpl_log:
            photo_pressure = np.log10(photo_pressure)

        ax.plot(
            wavl,
            photo_pressure,
            color=photosphere_colour,
            lw=1,
            ls=ls,
        )

    # -------- cosmetic -------- #
    if use_mpl_log:
        ax.loglog()
        ax.set_ylabel(f"Pressure ({pressure_unit})")
    else:
        ax.semilogx()
        ax.set_ylabel(r"$\log_{10}$ Pressure " + f"({pressure_unit})")

    # flip to right orientation (only one axis - they are coupled by sharey)
    ax.invert_yaxis()

    # set limit
    # ax.set_ylim([pressure.min(), pressure.max()])

    # ticks & labels
    xticks = [0.3, 0.5, 1, 2, 3, 4.5, 6, 9, 20, 34]
    ax.set_xticks(ticks=xticks, labels=xticks)
    ax.set(xlabel="wavelength [micron]", xlim=xlim)

    if axis is None:
        fig.show()


def plot_condensation(
    result: PhaethonResult, cond_mode: Literal["none", "equilibrium", "rainout"]
):
    """
    A postprocessing tool to run condensation along p-T-profile.

    Parameters
    ----------
        cond_mode: Literal["none", "equilibrium", "rainout"]
            none:
                No condensation.
            equilibrium:
                Equilibrium condensation, conserves elemental comp. along column.
            rainout:
                Removes elements from layer if they condense, runs from bottom
                to top of the atmosphere.
    """

    element_cond_degree, cond_number_density = result.run_cond(
        cond_mode=cond_mode, full_output=False
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[0.3, 1, 1], sharey=True)

    # p-T-profile
    ax[0].semilogy(
        result.temperature,
        result.pressure,
    )

    # elemental condensation fraction
    for elem in element_cond_degree.columns:
        ax[1].semilogy(
            element_cond_degree[elem],
            result.pressure.to("bar").value,
            label=elem,
        )
    labelLines(ax[1].get_lines())
    ax[1].legend(loc=0, ncols=2)

    # cond species number density
    for cond in cond_number_density.columns:
        if np.any(cond_number_density[cond] > 0.0):
            mask = cond_number_density[cond] > 0.0
            ax[2].semilogy(
                cond_number_density[cond][mask],
                result.pressure.to("bar").value[mask],
                label=cond,
            )
    ax[1].semilogx()
    # labelLines(ax[2].get_lines())
    ax[2].legend(loc=0)

    # only do it once
    ax[0].invert_yaxis()

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    fig.show()
