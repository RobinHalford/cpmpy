# SHORT TEMPORARY DESCRIPTIONS

# enumerate MUSes with marco, at every iteration grow the diversity matrix and
# return the top k diverse MUSes when diversity of 1 is reached or after max iterations i
def marco_until_diverse():
    return


# enumerate a fixed amount i MUSes, compute the diversity matrix, select the subset with the highest diversity and return it.
def marco_select_top_k():
    return


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