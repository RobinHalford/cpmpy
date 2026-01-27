# file that initiates experiments for diverse MUS enumeration
import cpmpy as cp
import numpy as np
import importlib
importlib.reload(cp)
from cpmpy.tools.explain.marco import marco
from cpmpy.tools.explain.utils import diversity_matrix
from cpmpy.tools.explain.diverse_enumeration import marco_select_top_k, marco_diverse, marco_until_diverse, ocus_enum
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from cpmpy.tools.explain.visualize_diversity import visualize_heatmap
from cpmpy.tools.explain.experiments_diverse.utils_experiments import create_unsat_nr_model, enum_sat_competition_instances



# Sudoku experiments

def create_sudoku_model(given):
    """
    Create a CPMpy model for the given Sudoku puzzle.
    """
    e = 0
    puzzle = cp.intvar(1,9, shape=given.shape, name="puzzle")
    hard = []
    soft = []
    soft.append(puzzle[given!=e] == given[given!=e])
    hard.append([cp.AllDifferent(row) for row in puzzle])
    hard.append([cp.AllDifferent(col) for col in puzzle.T])
    # Constraints on blocks
    for i in range(0,9, 3):
        for j in range(0,9, 3):
            hard.append(cp.AllDifferent(puzzle[i:i+3, j:j+3])) # python's indexing
    return puzzle, hard, soft


e = 0
given = np.array([
    [e, 2, e,  2, e, 5,  e, e, e],
    [e, 2, e,  e, 2, e,  7, 3, e],
    [e, e, 2,  e, e, 9,  e, 6, e],

    [2, e, e,  e, e, 2,  4, e, 9],
    [e, 2, e,  e, 7, 2,  e, 2, e],
    [6, e, 9,  e, e, e,  e, e, 1],

    [e, 8, e,  4, e, 2,  1, 2, e],
    [e, 6, 3,  e, 2, e,  e, 8, e],
    [e, e, 2,  6, e, 8,  e, e, e]])



# Nurse rostering experiments


def experiments_NR():
    for i in range(0,9):
        model, _, _ = create_unsat_nr_model(i, 0.9)
        assert model.solve() is False
        for j, (kind, subset) in enumerate(marco(model.constraints, solver="exact", map_solver="exact", return_mcs=False)):
            print("found MUS for NR instance ", i)
            if j == 4:
                break



# sat competition experiments

def load_sat_competition_instances():
    enum_sat_competition_instances()
    return
    


if __name__ == "__main__":
    # experiments_NR()
    load_sat_competition_instances()