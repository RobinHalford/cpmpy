
from cpmpy.tools.explain.marco import marco
import cpmpy as cp
import pickle as pkl

def test_pkl_marco_bug():

    filename= "SchurrLemma-030-9-mod.xml.lzma.pkl"
    path = "cpmpy/tools/explain/experiments_diverse/data/XCSP_MUS/" + filename

    # model = cp.Model().from_file(path)
    model = pkl.load(path)

    print("Printing constraints: \n\n")
    for c in model.constraints:
        print(c + "\n\n")
    
    print("Solving model: \n\n")
    result = model.solve("exact")
    print("Result: " + str(result))


if __name__ == "__main__":
    test_pkl_marco_bug()