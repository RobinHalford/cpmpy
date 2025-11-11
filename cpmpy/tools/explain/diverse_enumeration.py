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
    

    for j, (_, mus) in enumerate(marco(constraints, solver="exact", map_solver="exact", return_mcs=False)):
        muses.append(frozenset(mus))
        diversity_matrix = np.pad(diversity_matrix, ((0, 1), (0, 1)) ,mode="constant")
        
        for l in range():
            diversity_matrix[l,j] = diversity(muses[l], mus)
        
        if j >= k:
            top_indx, avg = select_top_k(diversity_matrix, k)
        
        if avg >= 1:
            break

        if j == i:
            break

    top_muses = [muses[i] for i in top_indx]
    
    return top_muses, avg, diversity_matrix


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

def select_top_k(matrix, k):
    """
        Returns a tuple with the indeces of the top-k most diverse MUSes.

        :param: matrix: the diversity matrix
        :param: k: the size of the top subset to be computed 
    """

    max_div = 0
    max_comb = None

    for comb in combinations(range(0, len(matrix[0])), k):
        avg_div = 0
        
        for indx in combinations(comb, 2):
            avg_div += matrix[indx]
        
        avg_div = avg_div / len(comb)

        if avg_div > max_div:
            max_div = avg_div
            max_comb = comb

    return max_comb, max_div