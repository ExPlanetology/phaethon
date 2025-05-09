#
# Copyright 2025 Fabian L. Seidler
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
A root-finding function for phaethon that tries to minimize the number of steps taken during the
search for the lowest ΔT.
"""
from dataclasses import dataclass, field
import logging
from typing import Callable, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import bayes_opt


@dataclass
class TemperatureSet:
    """Store associated temperature values"""

    tmelt: float
    delta_tmelt: float
    tboa: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.tboa = self.tmelt + self.delta_tmelt


def sign_has_flipped(x: List[TemperatureSet]) -> bool:
    """
    Determines if temperature sets where `delta_tmelt` has different signs are present in `x`.
    """
    sign_arr = np.sign([entry.delta_tmelt for entry in x])

    # assume that case 0 (i.e., perfect convergence) has not happened yet
    if abs(sign_arr.sum()) < len(sign_arr):
        return True
    return False


def find_closest_bounding_values(
    x: List[TemperatureSet],
) -> Tuple[TemperatureSet, TemperatureSet]:
    """
    Finds the TemperatureSet instance in `x` that is closest to convergence (i.e., smallest ΔT),
    and then finds the TemperatureSet of opposite sign in ΔT that is closest in `tmelt`.
    """

    # get TemperatureSet with smallest ΔT=0:
    closest_ts = min(x, key=lambda ts: abs(ts.delta_tmelt))

    # find all that have a different sign
    closets_opposite_sign_ts = TemperatureSet(tmelt=np.inf, delta_tmelt=0.0)
    for ts in x:
        if np.sign(ts.delta_tmelt) != np.sign(closest_ts.delta_tmelt):
            if abs(ts.tmelt - closest_ts.tmelt) < abs(
                closets_opposite_sign_ts.tmelt - closest_ts.tmelt
            ):
                closets_opposite_sign_ts = ts

    return (closest_ts, closets_opposite_sign_ts)


def find_root_of_linear(x: TemperatureSet, y: TemperatureSet) -> float:
    """
    Determines the `tmelt` where a linear function, defined by two TemperatureSets (pairs of
    `tmelt` and `delta_tmelt` values) has a root. If no root exists (i.e., function is constant),
    `None` is returned.
    """

    b: float = (x.delta_tmelt - y.delta_tmelt) / (x.tmelt - y.tmelt)
    if b == 0.0:
        # function is constant
        return None
    a: float = x.delta_tmelt - b * x.tmelt

    return -a / b


def is_repetition(value: float, tolerance: float, in_array: ArrayLike) -> bool:
    """
    Test if a value is already contained in an array, with tolerance.
    """
    arr = np.asarray(in_array)

    return np.any(np.abs(arr - value) <= tolerance)


class PhaethonConvergenceError(Exception):
    """Exception for when the root finder could not find a solution"""

    def __init__(self):
        # TODO: report best values, abs_tol, t_init & number of iterations, as well as trace!
        super().__init__(
            "Phaethon could not converge to a solution in time."
            + " Maybe increase number of iterations,"
            + " or relax convergence constraint (delta_tmelt_abstol)"
        )


class PhaethonRootFinder:
    """
    Root-finder for the equilibrium ocean temperature, which depends in the atmospheric opacity (
    which, in turn, is a function of the outgassed vapour). The assumptions made here are too
    problem specific to be applicable elsewhere.

    NOTE: This should find A SOLUTION, which does not have to be unique.
    """

    max_iter: int
    """ Maximum number of iterations """

    n_iter: int
    """ Number of iterations taken by the algorithm """

    logger: logging.Logger
    """ Logger """

    def __init__(
        self,
        tboa_func: Callable,
        delta_temp_abstol: float,
        t_init: float,
        *,
        max_iter: int,
        tmelt_limits: Tuple[float],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the root-finder.

        Parametes
        ---------
            tboa_func : Callable
                Function for ΔT := T_BOA - T_melt
            delta_temp_abstol : float
                Tolerance on ΔT.
            t_init : float
                Initial temperature.
            max_iter : int
                Maximum number of iterations allowed. If reached, an error will be yeeted.
        """

        self.tboa_func = tboa_func
        self.max_iter = max_iter
        self.trace: List[TemperatureSet] = []
        self.n_iter = 0
        self.delta_temp_abstol = delta_temp_abstol
        self.t_init = t_init
        self.tmelt_limits = tmelt_limits
        self._bayesian_optimizer = None
        self.logger = logger

    def solve(self) -> None:
        """
        Solve for T_melt that satisfies T_melt - T_BOA = 0. The solution, if one is found, might
        not be unique. This function does not return anything, as it is only applied in `Phaethon`
        where it directly modifies its internal state.
        """

        converged: bool = False
        self.n_iter: int = 0
        self.trace: List[TemperatureSet] = []

        while not converged:
            new_tmelt: float = self.suggest()

            new_tboa: float = self.tboa_func(new_tmelt)
            new_delta_tmelt: float = new_tboa - new_tmelt

            if self.logger is not None:
                self.logger.info(f"t_melt: {round(new_tmelt, 2)} K")
                self.logger.info(f"t_boa: {round(new_tboa, 2)} K")
                self.logger.info(f"-> ΔT: {round(new_delta_tmelt, 2)} K")

            self.trace.append(
                TemperatureSet(tmelt=new_tmelt, delta_tmelt=new_delta_tmelt)
            )

            converged: bool = abs(new_delta_tmelt) <= self.delta_temp_abstol

            self.n_iter += 1

            if self.n_iter == self.max_iter and not converged:
                raise PhaethonConvergenceError()

    def suggest(self) -> float:
        """
        Suggest a new melt temperature.
        """

        match self.n_iter:
            case 0:
                return self.t_init
            case 1:
                return self.trace[-1].tboa
            case _:
                # if sign on ΔT(T) flips, a root must exist. Hurray! Narrow down on it!
                if sign_has_flipped(self.trace[-2:]):
                    bound1, bound2 = find_closest_bounding_values(self.trace)
                    return find_root_of_linear(bound1, bound2)

                # use last two iterations to predict a new tmelt
                suggested_tmelt: float = find_root_of_linear(
                    self.trace[-2], self.trace[-1]
                )

                # if function has no root, resort to violence
                if suggested_tmelt is None:
                    suggested_tmelt = self.trace[-1].tboa

                # is suggested tmelt within search limits? if not, return temperature limit.
                min_temp: float = min(self.tmelt_limits)
                if suggested_tmelt < min_temp:
                    suggested_tmelt = max(min_temp, self.trace[-1].tboa)

                max_temp: float = max(self.tmelt_limits)
                if suggested_tmelt > max_temp:
                    suggested_tmelt = min(max_temp, self.trace[-1].tboa)

                # check if solution has already been encountered before, i.e. the solver is
                # oscillating between two values:
                if is_repetition(
                    value=suggested_tmelt,
                    tolerance=self.delta_temp_abstol,
                    in_array=[ts.tmelt for ts in self.trace],
                ):
                    suggested_tmelt = self._bayesian_suggest()

                return suggested_tmelt

    def _bayesian_suggest(self) -> float:
        """
        Predict new value with a BayesianOptimizer, a last ditch effort to find an optimum. The 
        Optimizer is set to be exploitive, i.e. tries to predict an optimum value even if the 
        parameter space has not been fully explored yet. This bears the risk of converging to
        a non-optimal solution. TODO: Maybe its better to be explorative instead?

        NOTE: Re-initializes the optimizer everytime the function is called. Might be inefficient,
        but a.) this occasion should be rare and b.) would make the code slightly more convoluted
        so we keep it nicely isolated in a single function.
        """
        self._bayesian_optimizer = bayes_opt.BayesianOptimization(
            f=None,
            pbounds={"t_melt": self.tmelt_limits},
            # bounds_transformer = SequentialDomainReductionTransformer(minimum_window=ΔT * 2)
            acquisition_function=bayes_opt.acquisition.ExpectedImprovement(xi=0.0),
        )

        # warm start for the optimizer by passing the previous points
        for entry in self.trace:
            self._bayesian_optimizer.register(
                params={"t_melt": entry.tmelt},
                target=-abs(
                    entry.delta_tmelt
                ),  # negative because the optimizer maximises f
            )

        # predict new point
        next_point: Dict[str, float] = self._bayesian_optimizer.suggest()
        return next_point["t_melt"]

    def _visualize(self, target_temp: Optional[float], n_samples: int = 1000) -> None:
        """
        Visualize the problem. Only for debugging purposes. Should not be used with computationally
        expensive functions!
        """

        t_arr = np.linspace(min(self.tmelt_limits), max(self.tmelt_limits), n_samples)

        fig, (ax_t, ax_dt) = plt.subplots(2, 1, height_ratios=[1, 0.4])

        # plot function itself
        func_eval_arr = [self.tboa_func(t_melt) for t_melt in t_arr]
        ax_t.plot(t_arr, func_eval_arr, zorder=0)
        ax_dt.plot(t_arr, func_eval_arr - t_arr, zorder=0)  # ΔT

        # plot function evaluations
        x_points = [ts.tmelt for ts in self.trace]
        y_points = [ts.tboa for ts in self.trace]
        z_points = [ts.delta_tmelt for ts in self.trace]
        n_iter_arr = np.arange(1, self.n_iter + 1)
        cs = ax_t.scatter(x_points, y_points, marker="o", c=n_iter_arr, edgecolors="k")
        ax_dt.scatter(x_points, z_points, marker="o", c=n_iter_arr, edgecolors="k")
        plt.colorbar(cs)

        if target_temp is not None:
            ax_t.plot(t_arr, t_arr, color="lightgrey", zorder=-2, label="target value")
            ax_t.scatter(target_temp, target_temp, marker="x", color="red", zorder=1)

            ax_dt.axhline(0.0, color="lightgrey", zorder=-2, label="target value")
            ax_dt.scatter(target_temp, 0.0, marker="x", color="red", zorder=1)

        # cosmetic
        ax_dt.set_xlabel(r"$T_{melt}$ [K]")
        ax_t.set_ylabel(r"$T_{BOA}$ [K]")
        ax_t.set(ylim=[np.amin(func_eval_arr) * 0.9, np.amax(func_eval_arr) * 1.1])
        ax_dt.set_ylabel(r"$\Delta T$ [K]")

        fig.tight_layout()
        plt.show()
