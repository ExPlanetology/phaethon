import matplotlib.pyplot as plt
from labellines import labelLines
import numpy as np
from phaethon.root_finder import PhaethonRootFinder, PhaethonConvergenceError, find_closest_bounding_values, find_root_of_linear

def test_simple_linear():

    t_correct: float = 3000.0
    dt_at_zero: float = 400.
    tmelt_abstol: float = 35.

    def tboa_func(tmelt: float) -> float:
        return t_correct + dt_at_zero - dt_at_zero / t_correct * tmelt

    root_finder = PhaethonRootFinder(
        tboa_func=tboa_func,
        delta_temp_abstol=tmelt_abstol,
        t_init=2000.,
        max_iter=15,
        tmelt_limits=(500., 5000.)
    )
    root_finder.solve()
    root_finder._visualize(target_temp=t_correct)

    assert(abs(root_finder.trace[-1].tmelt - t_correct) <= tmelt_abstol)

def test_wedge():

    t_correct: float = 3000.0
    dt_at_zero: float = 400.
    tmelt_abstol: float = 35.

    def tboa_func(tmelt: float) -> float:
        return t_correct + abs(dt_at_zero - dt_at_zero / t_correct * tmelt)

    root_finder = PhaethonRootFinder(
        tboa_func=tboa_func,
        delta_temp_abstol=tmelt_abstol,
        t_init=2000.,
        max_iter=15,
        tmelt_limits=(500., 5000.)
    )
    root_finder.solve()
    root_finder._visualize(target_temp=t_correct)

    assert(abs(root_finder.trace[-1].tmelt - t_correct) <= tmelt_abstol)


def test_parabola():

    t_correct: float = 3000.0
    dt_at_zero: float = 400.
    tmelt_abstol: float = 35.

    def tboa_func(tmelt: float) -> float:
        return tmelt + 1e-3 * (tmelt - t_correct)**2

    root_finder = PhaethonRootFinder(
        tboa_func=tboa_func,
        delta_temp_abstol=tmelt_abstol,
        t_init=2000.,
        max_iter=15,
        tmelt_limits=(500., 5000.)
    )
    root_finder.solve()
    root_finder._visualize(target_temp=t_correct)


def test_ushape():

    tmelt_abstol: float = 35.

    t_correct: float = 3000.0
    width_of_well: float = 100.0 # K
    plateu_value: float = 200.0 # K
    def tboa_func(tmelt: float) -> float:
        if abs(tmelt - t_correct) <= width_of_well:
            return tmelt + plateu_value / (width_of_well**2) * (tmelt - t_correct)**2
        return tmelt + plateu_value

    root_finder = PhaethonRootFinder(
        tboa_func=tboa_func,
        delta_temp_abstol=tmelt_abstol,
        t_init=1930.,
        max_iter=15,
        tmelt_limits=(500., 5000.)
    )
    root_finder.solve()
    root_finder._visualize(target_temp=t_correct)

def test_failure():

    t_correct: float = 3000.0
    dt_at_zero: float = 400.
    tmelt_abstol: float = 35.

    def tboa_func(tmelt: float) -> float:
        return t_correct * 2


    root_finder = PhaethonRootFinder(
        tboa_func=tboa_func,
        delta_temp_abstol=tmelt_abstol,
        t_init=2000.,
        max_iter=15,
        tmelt_limits=(500., 5000.)
    )

    # must throw an exception
    root_finder.solve()
