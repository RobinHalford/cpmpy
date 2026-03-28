
from cpmpy.tools.explain.marco import marco
import cpmpy as cp
import pickle as pkl

def test_pkl_marco_bug():

    filename= "SchurrLemma-030-9-mod.xml.lzma.pkl"
    path = "cpmpy/tools/explain/experiments_diverse/data/XCSP_MUS/" + filename

    # model = cp.Model().from_file(path)
    with open(path, 'rb') as f:
        model = pkl.load(f)

    print("Printing constraints: \n\n")
    for c in model.constraints:
        print(c + "\n\n")
    
    print("Solving model: \n\n")
    # result = model.solve("exact")
    # print("Result: " + str(result))

    generator = marco(model.constraints, solver="exact", map_solver="exact", return_mcs=False)
    for kind, subset in generator:
        print("Found a MUS \n")
        break



if __name__ == "__main__":
    test_pkl_marco_bug()