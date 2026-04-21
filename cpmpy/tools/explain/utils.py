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

import copy
import numpy as np
import cpmpy as cp
from cpmpy.expressions.utils import is_any_list
from cpmpy.expressions.variables import NegBoolView
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


def replace_cons_with_assump(cpm_cons, assump_map):
    """
        Replace soft constraints with assumption variables in a Boolean CPMpy expression.
    """

    if is_any_list(cpm_cons):
        return [replace_cons_with_assump(c, assump_map) for c in cpm_cons]
    
    if cpm_cons in assump_map:
        return assump_map[cpm_cons]
    
    elif hasattr(cpm_cons, "args"):
        cpm_cons = copy.copy(cpm_cons)
        cpm_cons.update_args(replace_cons_with_assump(cpm_cons.args, assump_map))
        return cpm_cons

    elif isinstance(cpm_cons, NegBoolView):
        return ~replace_cons_with_assump(cpm_cons._bv, assump_map)
    return cpm_cons

class OCUSException(Exception):
    pass


def diversity_pair(set1, set2, measure="overlap"):
    """
        Compute the diversity between two sets of constraints (can be MUS, MCS, MSS, ...).
        Diversity is always between 0 and 1.

        USE INTEGERS (for example id mapping) INSTEAD OF PLAIN SET OF CONSTRAINTS TO AVOID REPRESENTATION COLLISION

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

    common = len(set1 & set2)

    if measure in ("Jaccard", "Jaccard index"):
        union = s1 + s2 - common
        # J(A, B) = |A ∩ B| / |A ∪ B|
        jacc = common / union
        score = 1 - jacc
        return score
    
    elif measure in ("overlap", "overlap coefficient", "Szymkiewicz–Simpson"):
        # overlap(A, B) = |A ∩ B| / min(|A|,|B|)
        overlap = common / min(s1, s2)
        score = 1 - overlap
        return score
    
    else:
        raise ValueError(f"Unknown diversity measure: {measure}")


def diversity_matrix(list_of_sets, measure="overlap"):
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
        divs[i, j] = diversity_pair(a, b, measure)

    return divs


def diversity_setOfMUSes(list_of_sets, measure="overlap"):
    """
        Returns the minimal pairwise diversity between all the sets of `list_of_sets`.
        The value is the minimal of all the values in the upper triangular of the diversity matrix.

        :param: list_of_sets: A list of sets for which the diversity will be computed
        :param: measure: name of a diversity measure ("Jaccard", "overlap", "set difference")
    """
    divs = diversity_matrix(list_of_sets, measure)
    min_div = divs[np.triu_indices_from(divs, k=1)].min()
    return min_div