from cpmpy.tools.explain.utils import make_assump_model
from cpmpy.tools.explain.marco import marco
import cpmpy as cp
import numpy as np
import pickle as pkl

def test_pkl_marco_bug():

    filename= "SchurrLemma-030-9-mod.xml.lzma.pkl"
    path = "cpmpy/tools/explain/experiments_diverse/data/XCSP_MUS/" + filename

    model = cp.Model().from_file(path)
    # with open(path, 'rb') as f:
        # model = pkl.load(f)

    print("Printing constraints: \n\n")
    for c in model.constraints:
        print(c)
    
    print("Solving model: \n\n")
    result = model.solve("exact")
    print("Result: " + str(result))



def create_sudoku_model():
    e = 0 # value for empty cells
    given = np.array([
        [e, e, e,  2, e, 5,  e, e, e],
        [e, 9, e,  e, e, e,  7, 3, e],
        [e, e, 2,  e, e, 9,  e, 6, e],

        [2, e, e,  e, e, e,  4, e, 9],
        [e, e, e,  e, 7, e,  e, e, 9],
        [6, e, 9,  e, e, e,  e, e, 1],

        [e, 8, e,  4, e, e,  1, e, e],
        [e, 6, 3,  e, e, e,  e, 8, e],
        [e, e, e,  6, e, 8,  e, e, e]])


    # Variables
    puzzle = cp.intvar(1,9, shape=given.shape, name="puzzle")


    model = cp.Model(
        # Constraints on values (cells that are not empty)
        puzzle[given!=e] == given[given!=e], # numpy's indexing, vectorized equality
        # Constraints on rows and columns
        [cp.AllDifferent(row) for row in puzzle],
        [cp.AllDifferent(col) for col in puzzle.T], # numpy's Transpose
    )

    # Constraints on blocks
    for i in range(0,9, 3):
        for j in range(0,9, 3):
            model += cp.AllDifferent(puzzle[i:i+3, j:j+3]) # python's indexing

    return model


def test_pkl_write_read():
    model = create_sudoku_model()
    print("model constraints before writing pkl: \n\n")
    for c in model.constraints:
        print(c)
    print("Writing to pkl file.")
    with open("cpmpy/tools/explain/sudoku.pkl", "wb") as f:
        pkl.dump(model, f)
    print("Loading model constraints: \n\n")
    with open("cpmpy/tools/explain/sudoku.pkl", "rb") as f:
        loaded_model = pkl.load(f)
    for c in loaded_model.constraints:
        print(c)
    

if __name__ == "__main__":

    smodel = create_sudoku_model()

    generator = marco(smodel.constraints, solver="exact", map_solver="exact", return_mcs=False)

    for kind, subset in generator:
        print("Found a MUS")
    print("enumeration complete")

    # test_pkl_marco_bug()
    # test_pkl_write_read()