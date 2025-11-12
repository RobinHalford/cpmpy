import cpmpy as cp
import numpy as np
from .marco import marco
from .utils import diversity
from itertools import combinations


# SHORT TEMPORARY DESCRIPTIONS


# enumerate a fixed amount i MUSes, compute the diversity matrix, select the subset with the highest diversity and return it.
def marco_select_top_k():
    return


# enumerate MUSes with marco, at every iteration grow the diversity matrix and
# return the top k diverse MUSes when diversity of 1 is reached or after max iterations i
def marco_until_diverse(constraints, k, i):
    
    # initiate list and matrix
    muses = []
    diversity_matrix = np.empty((1,1), dtype=float)
    diversity_matrix[0, 0] = 0
    
    print("Starting to enumerate MUSes ...")
    top_indx = None
    avg = 0

    for j, (_, mus) in enumerate(marco(constraints, solver="exact", map_solver="exact", return_mcs=False)):
        muses.append(frozenset(mus))

        if j > 0:
            diversity_matrix = np.pad(diversity_matrix, ((0, 1), (0, 1)) ,mode="constant")

        print(f"Found MUS number {j}")
        
        for l in range(len(muses)-1):
            print(f"Diversity between MUS {j} and MUS {l}: {diversity(muses[l], mus)}")
            diversity_matrix[l,j] = diversity(muses[l], mus)
        
        print(f"The diversity matrix is now: \n {diversity_matrix}.")

        if j == k:
            top_indx, avg = select_top_k(diversity_matrix, k,incremental_last=False)
            print(f"Current max average: {avg}")
            if avg >= 1:
                break

        if j > k:
            top_indx, avg = select_top_k(diversity_matrix, k,incremental_last=True, max_comb=top_indx, max_avg=avg)
            print(f"Current max average: {avg}")
            if avg >= 1:
                break

        if j == i:
            break

    top_muses = [muses[i] for i in top_indx]
    
    return top_indx, top_muses, avg, diversity_matrix


# enumerate k amount of MUSes with ocus, updating the objective every iteration (minimize the constraints that are already found)
def ocus_enum():
    return


# an modified version of marco where the grow and shrink procedures select constraints first that would make it more diverse
# (and the map solver minimizes already found constraints?) (or should this be a seperate function, and the combination a separate too?)
def marco_diverse_greedy():
    return


# a modified version of marco where the map solver, grow and shrink are optimized towards diversity with the help of ocus
def marco_diverse_ocus():
    return



# helper function

def select_top_k(matrix, k, incremental_last=False, max_comb=None, max_avg = 0):
    """
        Returns a tuple with the indeces of the top-k highest average in the matrix.

        :param: matrix: the upper triangular matrix with values
        :param: k: the size of the top subset to be computed
        :param: incremental_last: whether only the combinations with the last element
             in it are computed because this function is used incrementally.
    """

    n = len(matrix)

    if k <= 0 or n<=1:
        return max_comb, max_avg
    
    total_pairs = k * (k - 1) / 2.0

    if incremental_last:

        last = n - 1
        for base in combinations(range(n-1),k-1):
            comb = base + (last,)
            print(f"combination: {comb}")
            curr_sum = 0
            for indx in combinations(comb, 2):
                curr_sum += matrix[indx]
                print(f"taking avg elem: {matrix[indx]}")

            curr_avg = curr_sum / total_pairs
            print(f"The avg div of this combination was: {curr_avg}.")

            if curr_avg > max_avg:
                max_avg = curr_avg
                max_comb = comb
    
    else:
        for comb in combinations(range(n), k):
            curr_sum = 0
            print(f"combination: {comb}")
            for indx in combinations(comb, 2):
                print(f"taking avg elem: {matrix[indx]}")
                curr_sum += matrix[indx]
        
            curr_avg = curr_sum / total_pairs
            print(f"The avg div of this combination was: {curr_avg}.")

            if curr_avg > max_avg:
                max_avg = curr_avg
                max_comb = comb

    return max_comb, max_avg