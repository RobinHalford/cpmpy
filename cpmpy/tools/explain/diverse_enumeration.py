import cpmpy as cp
import numpy as np
from .utils import make_assump_model, diversity_pair
from itertools import combinations
from cpmpy.tools.explain.marco import timed_marco
import time
from cpmpy.solvers.solver_interface import ExitStatus


def remaining_time(deadline):
    """
        Seconds left in the global budget, or None if unlimited.

        :param deadline: deadline (used with time.monotonic())
    """
    if deadline is None:
        return None
    return deadline - time.monotonic()


def timed_solve(solver, deadline, **kwargs):
    """
        Solve with remaining global time.
        Returns solver result or None if the global time budget is exhausted.

        :param solver: the solver to use the timed solve.
        :param deadline: the deadline for the time limit.
    """
    rem = remaining_time(deadline)
    if rem is not None and rem <= 0:
        return None
    solver.solve(time_limit=rem, **kwargs)
    status = solver.status()
    if status.exitstatus == ExitStatus.OPTIMAL or status.exitstatus == ExitStatus.FEASIBLE:
        return True
    elif status.exitstatus == ExitStatus.UNSATISFIABLE:
        return False
    elif status.exitstatus == ExitStatus.UNKNOWN:
        return None 
    else:
        # This should not happen as long as there are no other ExitStatus implemented.
        raise NotImplementedError(f"ExitStatus {status.exitstatus} is not implemented.")


def overlap_CP_EXPR(x, y):
    """
        Returns the CPMpy Expression that represents the Overlap similarity between x and y.
        The Overlap is represented as an integer between 0 and 100 because CPMpy doesn't support float division.

        :param x: a list or np.array, the first list
        :param y: a list or np.array, the second list
    """
    x = np.array(x)
    y = np.array(y)
    return (100 * cp.sum((x & y))) // cp.min([cp.sum(x), cp.sum(y)])


def marco_until_diverse(constraints, k, time_limit, solver="exact", map_solver="exact"):
    """
        Enumerate MUSes with marco until either an optimally diverse set of k MUSes
        is found (pairwise min diversity >= 1), or the time limit is exhausted.
        Returns the status and the diversity curve: a list of (timestamp, best_min_diversity)
        tuples, one per MUS found. best_min_diversity is the best min pairwise diversity
        achievable with any k-subset of all MUSes found so far (0 if fewer than k MUSes found).

        :param constraints: soft constraints
        :param k: desired number of diverse MUSes
        :param time_limit: total time budget in seconds (a 2s buffer is reserved for post-processing)
        :param solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param map_solver: the hitting-set (MAP) solver to use, ideally incremental such as "gurobi", "pysat" or "exact"

    """

    start_time = time.monotonic()
    deadline = start_time + time_limit - 2  # reserve 2 seconds for post-processing

    constraint_to_idx = {id(c): i for i, c in enumerate(constraints)}

    muses = []  # stored as index lists
    curve = []  # list of (timestamp, best_min_div) snapshots
    div_matrix = np.zeros((0, 0), dtype=float)
    top_indx = None
    min_div = 0
    # enumerate MUSes with MARCO given the deadline
    for label, mus, _ in timed_marco(constraints, solver=solver, map_solver=map_solver,
                                              time_limit=remaining_time(deadline), return_mcs=False):
        if label == "TIMEOUT":
            break  # uses remaining 2 seconds returning the results
        if label != "MUS":
            continue
        timestamp = time.monotonic() - start_time

        mus_idx = [constraint_to_idx[id(c)] for c in mus]
        j = len(muses)  # 0-based index of this MUS before appending
        muses.append(mus_idx)

        # Expand diversity matrix and fill in pairwise diversities with new MUS
        if j == 0:
            div_matrix = np.zeros((1, 1), dtype=float)
        else:
            div_matrix = np.pad(div_matrix, ((0, 1), (0, 1)), mode="constant")
            for l in range(j):
                div_matrix[l, j] = diversity_pair(muses[l], mus_idx, measure="overlap")

        # Once we have at least k MUSes, track the most diverse k-subset
        if j == k - 1:
            # First time we have exactly k MUSes: only one combination, full scan
            top_indx, min_div = select_top_k(div_matrix, k, incremental_last=False)
            curve.append((timestamp, min_div))
            if min_div >= 1:
                break
        elif j >= k:
            # Incrementally check only combinations that include the newest MUS
            top_indx, min_div = select_top_k(div_matrix, k, incremental_last=True,
                                              max_comb=top_indx, max_min_div=min_div)
            curve.append((timestamp, min_div))
            if min_div >= 1:
                break
        else:
            # Fewer than k MUSes found so far: diversity undefined, record 0
            curve.append((timestamp, 0))

    if top_indx is None:
        return "TIMEOUT", curve
    else:
        return "COMPLETE", curve


def marco_diverse_Min(soft, hard=[], solver="exact", map_solver="exact", return_mus=True, return_mcs=False, do_solution_hint=True, time_limit=None):
    """
        A modified version of MARCO where the Grow and Shrink procedures' selection order is favoring diversity, 
        and the map solver is minimizes seen constraints.

        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: map_solver: the hitting-set (MAP) solver to use, ideally incremental such as "gurobi", "pysat" or "exact", must support objective function
        :param: return_mus: whether the algorithm should return MUSes
        :param: return_mcs: whether the algorithm should return MCSes
        :param: do_solution_hint: when true, will favor large seeds generated by the map-solver, and hence more likely
                                     to return MUSes. Especially useful when `return_mus=True`.
        :param: time_limit: total global time budget in seconds.
    """
    start_time = time.monotonic()
    deadline = None if time_limit is None else start_time + time_limit
    assert hasattr(cp.SolverLookup.get(solver), "get_core"), "MARCO requires a solver that supports assumption variables"

    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft))
    s = cp.SolverLookup.get(solver, model)

    # map solver for computing hitting sets
    map_solver = cp.SolverLookup.get(map_solver)
    do_solution_hint = do_solution_hint and hasattr(map_solver, 'solution_hint')  # solver may not support solution hinting...

    map_solver += cp.any(assump)
    if do_solution_hint:
        map_solver.solution_hint(assump, [1]*len(assump)) # we want large subsets, more likely to be a MUS

    # keep a map of which constraints are seen in previously generated MUSes
    seenmap = dict(zip(assump, [0]*len(assump)))
    
    while True:
        # get a seed from the map
        map_result = timed_solve(map_solver, deadline)
        if map_result is None:
            yield "TIMEOUT", None, time.monotonic() - start_time
            return
        if map_result is False:
            return

        seed = [a for a in assump if a.value()]

        # check if seed is sat or unsat
        sat_result = timed_solve(s, deadline, assumptions=seed)
        if sat_result is None:
            yield "TIMEOUT", None, time.monotonic() - start_time
            return

        if sat_result is True:
            # SAT, grow, to full MSS
            # Grow with already seen *similar* !! constraints first 
            # (a blocked similar MCS will stimulate more diverse MUS)

            # Assumptions encode indicator constraints a -> c, find all true assumptions
            #    and those that could just as well be made true given the current solution
            mss = [a for a,c in zip(assump, soft) if a.value() or c.value()]
            for to_add in sorted(set(assump) - set(mss), key=seenmap.get):
                grow_result = timed_solve(s, deadline, assumptions=mss + [to_add])
                if grow_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if grow_result is True:
                    mss.append(to_add)
            mcs = [a for a in assump if a not in frozenset(mss)] # take complement
            map_solver += cp.any(mcs) # block in map solver

            if return_mcs:
                yield "MCS", [dmap[a] for a in mcs], time.monotonic() - start_time

        else: 
            # UNSAT, shrink to MUS, re-use MUSX
            # Shrink with already seen constraints first
            core = set(s.get_core())
            for c in sorted(core, key=seenmap.get, reverse=True):
                if c not in core: # already removed
                    continue
                core.remove(c)
                shrink_result = timed_solve(s, deadline, assumptions=list(core))
                if shrink_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if shrink_result is True:
                    core.add(c)
                else: # UNSAT, shrink to new solver core (clause set refinement)
                    core = set(s.get_core())

            map_solver += ~cp.all(core) # block in map solver
            
            # update seenmap
            for a in core:
                seenmap[a] += 1

            if return_mus:
                yield "MUS", [dmap[a] for a in core], time.monotonic() - start_time
        
        # ensure solution hint is still active
        if do_solution_hint:
            map_solver.solution_hint(assump, [1]*len(assump)) # we want large subsets, more likely to be a MUS
        # Minimize over already seen constraints for next seed calculation
        map_solver.minimize(cp.sum([seenmap[a] for a in assump]*assump)) 


def marco_diverse_noMin(soft, hard=[], solver="exact", map_solver="exact", return_mus=True, return_mcs=False, time_limit=None):
    """
        A modified version of MARCO where the Grow and Shrink procedures' selection order is favoring diversity, 
        and the solution hint is set to promote unseen constraints.

        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: map_solver: the hitting-set (MAP) solver to use, ideally incremental such as "gurobi", "pysat" or "exact"
        :param: return_mus: whether the algorithm should return MUSes
        :param: return_mcs: whether the algorithm should return MCSes
        :param: do_solution_hint: when true, will favor large seeds generated by the map-solver, and hence more likely
                                     to return MUSes. Especially useful when `return_mus=True`.
        :param: time_limit: total global time budget in seconds.
    """
    start_time = time.monotonic()
    deadline = None if time_limit is None else start_time + time_limit
    assert hasattr(cp.SolverLookup.get(solver), "get_core"), "MARCO requires a solver that supports assumption variables"
    assert hasattr(cp.SolverLookup.get(map_solver), "solution_hint"), "This version of MARCO requires a map solver that supports solution hinting"

    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft))
    s = cp.SolverLookup.get(solver, model)

    # map solver for computing hitting sets
    map_solver = cp.SolverLookup.get(map_solver)

    map_solver += cp.any(assump)
    
    map_solver.solution_hint(assump, [1]*len(assump)) # we want large subsets, more likely to be a MUS
  
    # keep a map of which constraints are seen in previously generated MUSes
    seenmap = dict(zip(assump, [0]*len(assump)))
    
    while True:
        # get a seed from the map
        map_result = timed_solve(map_solver, deadline)
        if map_result is None:
            yield "TIMEOUT", None, time.monotonic() - start_time
            return
        if map_result is False:
            return
        
        seed = [a for a in assump if a.value()]

        # check if seed is sat or unsat
        sat_result = timed_solve(s, deadline, assumptions=seed)
        if sat_result is None:
            yield "TIMEOUT", None, time.monotonic() - start_time
            return

        if sat_result is True:
            # SAT, grow, to full MSS
            # Grow with already seen *similar* !! constraints first 
            # (a blocked similar MCS will stimulate more diverse MUS)

            
            # Assumptions encode indicator constraints a -> c, find all true assumptions
            #    and those that could just as well be made true given the current solution
            mss = [a for a,c in zip(assump, soft) if a.value() or c.value()]
            for to_add in sorted(set(assump) - set(mss), key=seenmap.get):
                grow_result = timed_solve(s, deadline, assumptions=mss + [to_add])
                if grow_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if grow_result is True:
                    mss.append(to_add)
            mcs = [a for a in assump if a not in frozenset(mss)] # take complement
            map_solver += cp.any(mcs) # block in map solver

            if return_mcs:
                yield "MCS", [dmap[a] for a in mcs], time.monotonic() - start_time


        else: 
            # UNSAT, shrink to MUS, re-use MUSX
            # Shrink with already seen constraints first 
            # (remove similarity in MUSes) 

            core = set(s.get_core())
            for c in sorted(core, key=seenmap.get, reverse=True):
                if c not in core: # already removed
                    continue
                core.remove(c)
                shrink_result = timed_solve(s, deadline, assumptions=list(core))
                if shrink_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if shrink_result is True:
                    core.add(c)
                else: # UNSAT, shrink to new solver core (clause set refinement)
                    core = set(s.get_core())

            map_solver += ~cp.all(core) # block in map solver
            
            # update seenmap
            for a in core:
                seenmap[a] += 1

            if return_mus:
                yield "MUS", [dmap[a] for a in core], time.monotonic() - start_time
        
        map_solver.solution_hint(assump, [1]*len(assump)) # we want large subsets, more likely to be a MUS


def marco_diverse_optimal(soft, hard=[], solver="exact", map_solver="exact", return_mus=True, return_mcs=False, do_solution_hint=True, time_limit=None):
    """
        A modified version of MARCO where the Grow and Shrink procedures' selection order is favoring diversity, 
        and the map solver is set to maximize the diversity compared to all previously generated MUSes.

        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: map_solver: the hitting-set (MAP) solver to use, ideally incremental such as "gurobi", "pysat" or "exact"
        :param: return_mus: whether the algorithm should return MUSes
        :param: return_mcs: whether the algorithm should return MCSes
        :param: do_solution_hint: when true, will favor large seeds generated by the map-solver, and hence more likely
                                     to return MUSes. Especially useful when `return_mus=True`.
        :param: time_limit: total global time budget in seconds.
    """
    start_time = time.monotonic()
    deadline = None if time_limit is None else start_time + time_limit
    assert hasattr(cp.SolverLookup.get(solver), "get_core"), "MARCO requires a solver that supports assumption variables"

    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft))
    s = cp.SolverLookup.get(solver, model)

    prev_MUSes_assump = []

    # map solver for computing hitting sets
    map_solver = cp.SolverLookup.get(map_solver)

    map_solver += cp.any(assump)

    if do_solution_hint:
        map_solver.solution_hint(assump, [1]*len(assump)) # we want large subsets, more likely to be a MUS
  
    # keep a map of which constraints are seen in previously generated MUSes
    seenmap = dict(zip(assump, [0]*len(assump)))
    
    while True:
        # get a seed from the map
        map_result = timed_solve(map_solver, deadline)
        if map_result is None:
            yield "TIMEOUT", None, time.monotonic() - start_time
            return
        if map_result is False:
            return
        
        seed = [a for a in assump if a.value()]

        # check if seed is sat or unsat
        sat_result = timed_solve(s, deadline, assumptions=seed)
        if sat_result is None:
            yield "TIMEOUT", None, time.monotonic() - start_time
            return

        if sat_result is True:
            # SAT, grow, to full MSS
            # Grow with already seen *similar* !! constraints first 
            # (a blocked similar MCS will stimulate more diverse MUS)

            
            # Assumptions encode indicator constraints a -> c, find all true assumptions
            #    and those that could just as well be made true given the current solution
            mss = [a for a,c in zip(assump, soft) if a.value() or c.value()]
            for to_add in sorted(set(assump) - set(mss), key=seenmap.get):
                grow_result = timed_solve(s, deadline, assumptions=mss + [to_add])
                if grow_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if grow_result is True:
                    mss.append(to_add)
            mcs = [a for a in assump if a not in frozenset(mss)] # take complement
            map_solver += cp.any(mcs) # block in map solver

            if return_mcs:
                yield "MCS", [dmap[a] for a in mcs], time.monotonic() - start_time


        else: 
            # UNSAT, shrink to MUS, re-use MUSX
            # Shrink with already seen constraints first 
            # (remove similarity in MUSes) 

            core = set(s.get_core())
            for c in sorted(core, key=seenmap.get, reverse=True):
                if c not in core: # already removed
                    continue
                core.remove(c)
                shrink_result = timed_solve(s, deadline, assumptions=list(core))
                if shrink_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if shrink_result is True:
                    core.add(c)
                else: # UNSAT, shrink to new solver core (clause set refinement)
                    core = set(s.get_core())

            map_solver += ~cp.all(core) # block in map solver
            
            # update seenmap
            for a in core:
                seenmap[a] += 1

            if return_mus:
                yield "MUS", [dmap[a] for a in core], time.monotonic() - start_time

            one_hot_MUS = np.array([a in core for a in assump], dtype=bool)
            prev_MUSes_assump.append(one_hot_MUS)
        # ensure solution hint is still active
        if do_solution_hint:
            map_solver.solution_hint(assump, [1]*len(assump)) # we want large subsets, more likely to be a MUS
        
        # minimize the max overlap (= maximize min diversity) over all pairs with previous MUSes
        overlaps = [overlap_CP_EXPR(assump, prev) for prev in prev_MUSes_assump]
        overlap = cp.max(overlaps) # max overlap over all pairs (minimizing this maximizes min diversity)
        map_solver.minimize(overlap)


def ocus_enum_1(soft, hard=[], solver="ortools", hs_solver="gurobi", time_limit=None):
    """
        A modified version of the OCUS algorithm that enumerates k diverse MUSes one by one, starting with the smallest MUS
        and then computing each next MUS with the weights of seen constraints increased by the size of the last found MUS.

        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: hs_solver: the hitting-set solver to use, ideally incremental such as "gurobi"
        :param: time_limit: the global time budget in seconds.
    """
    start_time = time.monotonic()
    deadline = None if time_limit is None else start_time + time_limit
    
    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"optimal_mus requires a solver that supports assumption variables"
    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft))
    seenmap = dict(zip(assump, [1]*len(assump)))
    
    s = cp.SolverLookup.get(solver, model)

    if hasattr(s, 'solution_hint'):
        s.solution_hint(assump, [1]*len(assump))

    # initialize hitting set solver
    hs_solver = cp.SolverLookup.get(hs_solver)

    # generate MUSes loop
    while True:
        seen = [seenmap[a] for a in assump]
        hs_solver.minimize(cp.sum(assump * np.array(seen)))

        # hitting set loop
        unsat_hitting_set = None
        while True:
            hs_result = timed_solve(hs_solver, deadline)
            if hs_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if hs_result is False:
                break
            
            hitting_set = [a for a in assump if a.value()]

            sat_result = timed_solve(s, deadline, assumptions=hitting_set)
            if sat_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if sat_result is False:
                unsat_hitting_set = hitting_set
                break

            # else, the hitting set is SAT, now try to extend it without extra solve calls.
            # Check which other assumptions/constraints are satisfied (using c.value())
            # complement of grown subset is a correction subset
            # Assumptions encode indicator constraints a -> c, find all false assumptions
            #   that really have to be false given the current solution.
            new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
            hs_solver += cp.sum(new_corr_subset) >= 1

            # greedily search for other corr subsets disjoint to this one
            sat_subset = list(new_corr_subset)
            while True:
                sat_result = timed_solve(s, deadline, assumptions=sat_subset)
                if sat_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if sat_result is False:
                    break
                new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
                sat_subset += new_corr_subset # extend sat subset with new corr subset, guaranteed to be disjoint
                hs_solver += cp.sum(new_corr_subset) >= 1 # add new corr subset to hitting set solver
                
        if unsat_hitting_set is None:
            return
        hitting_set = set(unsat_hitting_set)
        inc = len(hitting_set)
        # increase the weights of all seen constraints in this MUS by len(MUS)
        for a in hitting_set:
            seenmap[a] += inc
        # block found MUS in hitting set solver
        hs_solver += ~cp.all(hitting_set)
        yield "MUS", [dmap[a] for a in hitting_set], time.monotonic() - start_time
    

def ocus_enum_shrink(soft, hard=[], solver="ortools", hs_solver="gurobi", time_limit=None):
    """
        A modified version of the OCUS algorithm that enumerates k diverse MUSes one by one, starting with the smallest MUS,
        and then resetting all weights to zero. All the seen constraints' weights are then set to 1.
        After finding a unsat subset, the Shrink procedure is done because this doesn't guarantee MUSes (weights can be zero).

        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: hs_solver: the hitting-set solver to use, ideally incremental such as "gurobi"
        :param: time_limit: the global time budget in seconds.
    """
    start_time = time.monotonic()
    deadline = None if time_limit is None else start_time + time_limit
    
    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"optimal_mus requires a solver that supports assumption variables, use optimal_mus_naive with {solver} instead"
    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft))
    # first MUS to compute is smallest MUS
    seenmap = dict(zip(assump, [1]*len(assump)))
    
    s = cp.SolverLookup.get(solver, model)
    
    if hasattr(s, 'solution_hint'):
        s.solution_hint(assump, [1]*len(assump))

    # initialize hitting set solver
    hs_solver = cp.SolverLookup.get(hs_solver)
    first = True
    # generate MUSes loop
    while True:
        seen = [seenmap[a] for a in assump]
        hs_solver.minimize(cp.sum(assump * np.array(seen)))

        # hitting set loop
        unsat_hitting_set = None
        while True:
            hs_result = timed_solve(hs_solver, deadline)
            if hs_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if hs_result is False:
                break
            
            hitting_set = [a for a in assump if a.value()]

            sat_result = timed_solve(s, deadline, assumptions=hitting_set)
            if sat_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if sat_result is False:
                unsat_hitting_set = hitting_set
                break 

            # else, the hitting set is SAT, now try to extend it without extra solve calls.
            # Check which other assumptions/constraints are satisfied (using c.value())
            # complement of grown subset is a correction subset
            # Assumptions encode indicator constraints a -> c, find all false assumptions
            #   that really have to be false given the current solution.
            new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
            hs_solver += cp.sum(new_corr_subset) >= 1

            # greedily search for other corr subsets disjoint to this one
            sat_subset = list(new_corr_subset)
            while True:
                sat_result = timed_solve(s, deadline, assumptions=sat_subset)
                if sat_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if sat_result is False:
                    break

                new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
                sat_subset += new_corr_subset # extend sat subset with new corr subset, guaranteed to be disjoint
                hs_solver += cp.sum(new_corr_subset) >= 1 # add new corr subset to hitting set solver
        
        if unsat_hitting_set is None:
            return
        # shrink to a MUS
        hitting_set = set(unsat_hitting_set)
        for c in sorted(hitting_set, key=seenmap.get, reverse=True):
            if c not in hitting_set: # already removed
                continue
            hitting_set.remove(c)
            sat_result = timed_solve(s, deadline, assumptions=list(hitting_set))
            if sat_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if sat_result:
                hitting_set.add(c)
            else: # UNSAT, shrink to new solver core (clause set refinement)
                hitting_set = set(s.get_core())

        # when first MUS (smallest) is found, reset all weights to zero
        if first:
            first = False
            seenmap = dict(zip(assump, [0]*len(assump)))

        for a in hitting_set:
            seenmap[a] += 1
        # block found MUS in hitting set solver
        hs_solver += ~cp.all(hitting_set)
        yield "MUS", [dmap[a] for a in hitting_set], time.monotonic() - start_time





def ocus_enum_opt_nextMUS(soft, hard=[], solver="ortools", hs_solver="gurobi", time_limit=None):
    """
        A modified version of the OCUS algorithm that enumerates k diverse MUSes one by one, starting with the smallest MUS,
        then computing the next MUS based on the objective function that minimizes the overlap between the next MUS and all 
        previously found MUSes. The unsat subset is shrunk because the objective is not monotonically increasing.

        :param: solver: name of a solver, must support assumptions (e.g, "ortools", "exact", "z3" or "pysat")
        :param: hs_solver: the hitting-set solver to use, ideally incremental such as "gurobi"
        :param: time_limit: the global time budget in seconds.
    """
    start_time = time.monotonic()
    deadline = None if time_limit is None else start_time + time_limit

    assert hasattr(cp.SolverLookup.get(solver), "get_core"), f"ocus requires a solver that supports assumption variables."

    model, soft, assump = make_assump_model(soft, hard)
    dmap = dict(zip(assump, soft)) # map assumption variables to constraints

    prev_MUSes_assump = []

    s = cp.SolverLookup.get(solver, model)
    if hasattr(s, 'solution_hint'):
        s.solution_hint(assump, [1]*len(assump))

    # initialize hitting set solver
    hs_solver = cp.SolverLookup.get(hs_solver)
    first = True

    # generate MUSes loop
    while True:
        # make sure objective is reset
        hs_solver.objective_value_ = None
        if first:
            hs_solver.minimize(cp.sum(assump * np.ones(len(assump), dtype=int)))
        else:
            # minimize the max overlap (= maximize min diversity) over all pairs with previous MUSes
            overlaps = [overlap_CP_EXPR(assump, prev) for prev in prev_MUSes_assump]
            overlap = cp.max(overlaps) # max overlap over all pairs (minimizing this maximizes min diversity)
            hs_solver.minimize(overlap)

         # hitting set loop
        unsat_hitting_set = None
        while True:
            hs_result = timed_solve(hs_solver, deadline)
            if hs_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if hs_result is False:
                break
            
            hitting_set = [a for a in assump if a.value()]

            sat_result = timed_solve(s, deadline, assumptions=hitting_set)
            if sat_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if sat_result is False:
                unsat_hitting_set = hitting_set
                break

            # else, the hitting set is SAT, now try to extend it without extra solve calls.
            # Check which other assumptions/constraints are satisfied (using c.value())
            # complement of grown subset is a correction subset
            # Assumptions encode indicator constraints a -> c, find all false assumptions
            #   that really have to be false given the current solution.
            new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
            hs_solver += cp.sum(new_corr_subset) >= 1

            # greedily search for other corr subsets disjoint to this one
            sat_subset = list(new_corr_subset)
            while True:
                sat_result = timed_solve(s, deadline, assumptions=sat_subset)
                if sat_result is None:
                    yield "TIMEOUT", None, time.monotonic() - start_time
                    return
                if sat_result is False:
                    break
                new_corr_subset = [a for a,c in zip(assump, soft) if a.value() is False and c.value() is False]
                sat_subset += new_corr_subset # extend sat subset with new corr subset, guaranteed to be disjoint
                hs_solver += cp.sum(new_corr_subset) >= 1 # add new corr subset to hitting set solver
        
        if unsat_hitting_set is None:
            return
        if first:
            first = False

        # Shrinking because obejctive is not monotone (not guaranteed a MUS)
        hitting_set = set(unsat_hitting_set)
        assump_idx = {a: i for i, a in enumerate(assump)}
        for c in sorted(hitting_set, key=assump_idx.get, reverse=True):
            if c not in hitting_set: # already removed
                continue
            hitting_set.remove(c)
            sat_result = timed_solve(s, deadline, assumptions=list(hitting_set))
            if sat_result is None:
                yield "TIMEOUT", None, time.monotonic() - start_time
                return
            if sat_result:
                hitting_set.add(c)
            else: # UNSAT, shrink to new solver core (clause set refinement)
                hitting_set = set(s.get_core())

        # make 1 hot encoded vector of MUS so we can use it for objective
        one_hot_MUS = np.array([a in hitting_set for a in assump], dtype=bool)
        prev_MUSes_assump.append(one_hot_MUS)
        # block found MUS in hitting set solver
        hs_solver += ~cp.all(hitting_set)
        
        yield "MUS", [dmap[a] for a in hitting_set], time.monotonic() - start_time


def select_top_k(matrix, k, incremental_last=False, max_comb=None, max_min_div=0):
    """
        Returns a tuple with the indices of the top-k subset with the highest minimal pairwise diversity in the matrix.

        :param: matrix: the upper triangular matrix with values
        :param: k: the size of the top subset to be computed
        :param: incremental_last: whether only the combinations with the last element
             in it are computed because this function is used incrementally.
    """

    n = len(matrix)

    if k <= 0 or n <= 1:
        return max_comb, float(max_min_div)

    if incremental_last:
        last = n - 1
        for base in combinations(range(n - 1), k - 1):
            comb = base + (last,)
            curr_min = min(matrix[indx] for indx in combinations(comb, 2))

            if curr_min > max_min_div:
                max_min_div = curr_min
                max_comb = comb

    else:
        for comb in combinations(range(n), k):
            curr_min = min(matrix[indx] for indx in combinations(comb, 2))

            if curr_min > max_min_div:
                max_min_div = curr_min
                max_comb = comb

    return max_comb, float(max_min_div)