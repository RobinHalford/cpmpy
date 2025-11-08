#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## utils.py
##
"""
    Utilities for explanation techniques

    =================
    List of functions
    =================

    .. autosummary::
        :nosignatures:

        make_assump_model
"""

import numpy as np
import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from itertools import combinations


def make_assump_model(soft, hard=[], name=None):
    """
        Construct implied version of all soft constraints.
        Can be used to extract cores (see :func:`tools.mus() <cpmpy.tools.explain.mus.mus>`).
        Provide name for assumption variables with `name` param.
    """
    # ensure toplevel list
    soft2 = toplevel_list(soft, merge_and=False)

    # make assumption variables
    assump = cp.boolvar(shape=(len(soft2),), name=name)

    # hard + implied soft constraints
    hard = toplevel_list(hard)
    model = cp.Model(hard + [assump.implies(soft2)])  # each assumption variable implies a candidate

    return model, soft2, assump


def diversity(set1, set2, measure="Jaccard"):
    """
        Compute the diversity between two sets of constraints (can be MUS, MCS, MSS, ...).
        Can be used as a measure to obtain diverse explanations. ()
        Provide a diversity measure with `measure` param.

        The value of a diversity is always between 0 and 1, where 1 is the highest diversity possible and zero the lowest.

        :param: set1: the first set
        :param: set2: the second set
        :param: measure: name of a diversity measure ("Jaccard", "overlap", "set difference")
    """
    set1, set2 = frozenset(set1), frozenset(set2)
    s1, s2 = len(set1), len(set2)

    if (s1 == 0) and (s2 == 0):  # both sets are empty -> no diversity
        return 0
    if (s1 == 0) or (s2 == 0):  # one set is empty -> max diversity
        return 1

    if measure in ("Jaccard", "Jaccard index"):
        common = len(set1 & set2)
        union = s1 + s2 - common
        # J(A, B) = |A ∩ B| / |A ∪ B|
        jacc = common / union
        score = 1 - jacc
        return score
    
    elif measure in ("overlap", "overlap coefficient", "Szymkiewicz–Simpson"):
        common = len(set1 & set2)
        # overlap(A, B) = |A ∩ B| / min(|A|,|B|)
        overlap = common / min(s1, s2)
        score = 1 - overlap
        return score
    
    elif measure in ("set difference", "symmetric set difference"):
        # normalized symmetric set difference is the same as (1 - jaccardIndex) so this is kinda pointless
        common = len(set1 & set2)
        union = s1 + s2 - common
        diff = union - common
        return diff / union
    
    else:
        raise ValueError(f"Unknown diversity measure: {measure}")


def diversity_matrix(list_of_sets, measure="Jaccard"):
    """
        Computes the diversity matrix (upper triangular) with the pairwise diversities between all the sets of `list_of_sets`.
        The value at index [i, j] is the diversity between set i and set j.

        :param: list_of_sets: A list of sets for which the diversity matrix will be computed
        :param: measure: name of a diversity measure ("Jaccard", "overlap", "set difference")
    """
    n = len(list_of_sets)
    divs = np.zeros((n,n))

    for (i, j) in combinations(range(n), 2):
        a, b = list_of_sets[i], list_of_sets[j]
        divs[i, j] = diversity(a, b, measure)

    return divs
