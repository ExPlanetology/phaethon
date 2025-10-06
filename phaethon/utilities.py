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
Utilities for phaethon
"""
import functools
from typing import Dict, Tuple
from molmass import Formula
import signal


def formula_to_hill(formula: str) -> str:
    """
    Converts a chemical formula (e.g. SiO) into Hill's notation (O1Si1) for FastChem.
    """
    comp: Dict[str, Tuple[float, float, float]] = (
        Formula(formula).composition().asdict()
    )

    # if its only one element
    if len(comp) == 1:
        return list(comp.keys())[0]

    # comp is already sorted in Hill notation
    hill_formula: str = ""
    for elem, data in comp.items():
        hill_formula += elem + str(data[0])

    return hill_formula


def formula_to_latex(formula: str) -> str:
    """
    Converts a chemical formula (e.g. SiO, O1Si1) into latex markdown.
    """
    # lazy bugfix
    if formula == "e-":
        return "e-"

    # extract formula; FastChem has suffixes to formulas which we want to avoid at this stage.
    suffix = None
    if "_" in formula:
        (formula, suffix) = formula.split("_")
    if formula.endswith("trans"):
        formula = formula[:-5]
        suffix = "trans"
    if formula.endswith("cis"):
        formula = formula[:-4]
        suffix = "cis"
    if "(s)" in formula:
        (formula, _) = formula.split("(s)")
        suffix = "s"
    if "(s,l)" in formula:
        (formula, _) = formula.split("(s,l)")
        suffix = "s,l"
    if "(l)" in formula:
        (formula, _) = formula.split("(l)")
        suffix = "l"


    # extract stoichiometry
    comp: Dict[str, Tuple[float, float, float]] = (
        Formula(formula).composition().asdict()
    )

    # most commonly, O is in the end of the formula
    oxy_string = ""
    if "O" in comp:
        oxy_string += "O"
        oxy_stoich: int = comp.pop("O")[0]
        if oxy_stoich > 1:
            oxy_string += "$_{" + str(oxy_stoich) + "}$"

    # comp is already sorted in Hill notation
    latex_formula: str = ""
    for elem, data in comp.items():
        stoich: int = data[0]
        if elem == "e-":
            if stoich > 0:
                latex_formula += "-"
            else:
                latex_formula += "+"
        else:
            latex_formula += elem
            if stoich > 1:
                latex_formula += "$_{" + str(stoich) + "}$"

    latex_formula += oxy_string

    if suffix is not None:
        latex_formula += f"({suffix})"

    return latex_formula


class Timeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise Timeout("Function call timed out")


def timeout(timeout=5):
    """
    Decorator to run a function with a timeout. Time in seconds.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set the timeout handler for the alarm signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)  # Set an alarm for the specified timeout
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel the alarm
            return result

        return wrapper

    return decorator
