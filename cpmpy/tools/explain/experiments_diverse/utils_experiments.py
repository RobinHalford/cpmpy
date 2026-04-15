"""
PyTorch-style Dataset for Nurserostering instances from schedulingbenchmarks.org

Simply create a dataset instance and start iterating over its contents:
The `metadata` contains usefull information about the current problem instance.
"""
import os
import pathlib
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock
import gc
import multiprocessing as mp
import pandas as pd
import time
from cpmpy.tools.explain.mus import smus
from cpmpy.tools.explain.marco import timed_marco
from cpmpy.tools.explain.diverse_enumeration import marco_diverse_Min, marco_diverse_noMin, marco_diverse_optimal, marco_until_diverse, ocus_enum_1, ocus_enum_shrink, ocus_enum_opt_nextMUS
from examples.nurserostering import NurseRosteringDataset, nurserostering_model, parse_scheduling_period


import cpmpy as cp

DATA_DIR = pathlib.Path(__file__).parent / "data"


def get_optimal(model, time_limit=60, solver="ortools"):
    model.solve(time_limit=time_limit, solver=solver)
    print(model.objective_value())
    return model.objective_value()


def create_unsat_model(model, optimal, difficulty_factor):
    objective = model.objective_
    model.objective_ = None
    model += objective <= int(difficulty_factor * optimal)
    return model


def write_results_to_csv(result, fieldnames, output_file):
    # write results to csv
    lock_file = f"{output_file}.lock"
    lock = FileLock(lock_file)
    try:
        with lock:
            write_header = not os.path.exists(output_file)
            with open(output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass


def execute_solver_combinations(time_limit: int, output_file_NR: str, output_file_xcsp: str):
    dataset = NurseRosteringDataset(root=".", download=True, transform=parse_scheduling_period)
    data, metadata = dataset[0]
    model, _ = nurserostering_model(**data)
    print("created sat model")
    optimal = get_optimal(model, time_limit=time_limit, solver="exact")
    print(f"found optimal value: {optimal}")
    model = create_unsat_model(model, optimal, 0.1)
    print("created unsat model")
    instance = metadata["name"]
    execute_solver_combination_instance(instance, model, time_limit, output_file_NR)
    print("finished NR instance")
    # now a xcsp instance
    filenames = pd.read_csv("cpmpy/tools/explain/experiments_diverse/data/constraints_stats.csv", usecols=["filename"])["filename"].tolist()
    filename = filenames[22]
    path = "cpmpy/tools/explain/experiments_diverse/data/XCSP_MUS/" + filename
    print(f"Processing file: {filename}")
    model = cp.Model().from_file(path)
    print("loaded model")
    instance = filename
    execute_solver_combination_instance(instance, model, time_limit, output_file_xcsp)
    print("finished xcsp instance")
    return


def execute_solver_combination_instance(instance, model, time_limit: int, output_file: str):
    fieldnames = ["instance", "algorithm", "solver", "map_solver", "hs_solver", "status","num_mus", "runtimes", "error_message"]
    algorithms = ["marco", "ocus"]
    solvers = ["ortools", "exact", "pysat"]
    map_solvers = ["exact", "pysat", "gurobi"]
    hs_solvers = ["ortools", "exact", "gurobi"]
    for algorithm in algorithms:
        cmodel = model.copy()
        if algorithm == "marco":
            for solver in solvers:
                for map_solver in map_solvers:
                    result = dict.fromkeys(fieldnames)   # initialize result dict with empty values   
                    result["instance"] = instance
                    result["algorithm"] = algorithm
                    result["solver"] = solver
                    result["map_solver"] = map_solver
                    result["hs_solver"] = None
                    generator = enumerate(timed_marco(cmodel.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=20))
                    try:
                        runtimes = []
                        start_time = time.time()
                        for j, (_, subset) in generator:
                            runtimes.append(time.time() - start_time)
                            if time.time() - start_time > time_limit:
                                result["status"] = "TIMEOUT"
                                break
                        else:
                            result["status"] = "COMPLETE"
                        result["num_mus"] = j + 1
                        result["runtimes"] = runtimes
                    except Exception as e:
                        result["status"] = "error"
                        result["error_message"] = str(e)
                    write_results_to_csv(result, fieldnames, output_file)
        elif algorithm == "ocus":
            for solver in solvers:
                for hs_solver in hs_solvers:
                    result = dict.fromkeys(fieldnames)   # initialize result dict with empty values   
                    result["instance"] = instance
                    result["algorithm"] = algorithm
                    result["solver"] = solver
                    result["map_solver"] = None
                    result["hs_solver"] = hs_solver
                    # just OCUS, no enumeration
                    start_time = time.time()
                    try:
                        mus = smus(cmodel.constraints, solver=solver, hs_solver=hs_solver)
                        runtime = time.time() - start_time
                        result["status"] = "COMPLETE"
                        result["num_mus"] = 1
                        result["runtimes"] = [runtime]
                    except Exception as e:
                        result["status"] = "error"
                        result["error_message"] = str(e)
                    write_results_to_csv(result, fieldnames, output_file)


def execute_unsat_nr_models(num_mus, solver, map_solver, hs_solver, difficulty_factor, time_limit, output_file):
    dataset = NurseRosteringDataset(root=".", download=True, transform=parse_scheduling_period)
    fieldnames = ["instance", "algorithm", "solver", "map_solver", "hs_solver", "status", "runtimes", "error_message", "MUSes"]
    for i in range(0,1):
        data, metadata = dataset[i]
        # run all algorithms
        for algorithm in ["marco", # "marco_select_top_k", "marco_until_diverse", "ocus_enum",
                          "marco_diverse_no_min",
                          "marco_diverse_min"]:
            result = dict.fromkeys(fieldnames)   # initialize result dict with empty values   
            result["instance"] = metadata["name"]
            model, _ = nurserostering_model(**data)
            print("created sat model")
            optimal = get_optimal(model, time_limit=time_limit, solver=solver)
            print(f"found optimal value: {optimal}")
            model = create_unsat_model(model, optimal, difficulty_factor)
            print("created unsat model")
            assert model.solve(time_limit=60, solver="ortools") is False
            print("asserted model is unsat")
            try:
                result["algorithm"] = algorithm
                result["solver"] = solver
                if algorithm != "ocus_enum":
                    result["map_solver"] = map_solver
                    result["hs_solver"] = None
                else:
                    result["hs_solver"] = hs_solver
                    result["map_solver"] = None
                start_total = time.time()
                runtimes = []
                muses = []
                if algorithm == "marco":
                    generator = enumerate(timed_marco(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
                elif algorithm == "marco_select_top_k":
                    ... # TODO (not enumaration function)
                elif algorithm == "marco_diverse_min":
                    generator = enumerate(marco_diverse_Min(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
                elif algorithm == "marco_diverse_no_min":
                    generator = enumerate(marco_diverse_noMin(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
                elif algorithm == "ocus_enum":
                    # generator = enumerate(ocus_enum(model.constraints, solver=solver, hs_solver=hs_solver))
                    ... # TODO 
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                while True:
                    step_start = time.time()
                    try:
                        j, (_, subset) = next(generator)
                    except StopIteration:
                        break
                    muses.append(subset)
                    runtimes.append(time.time() - start_total)
                    # timeout check 
                    if time.time() - step_start > time_limit:
                        result["status"] = "TIMEOUT"
                        break
                    if j == num_mus - 1:
                        result["status"] = "COMPLETE"
                        break
                result["MUSes"] = muses
                result["runtimes"] = runtimes
            except Exception as e:
                result["algorithm"] = algorithm
                result["solver"] = solver
                if algorithm != "ocus_enum":
                    result["map_solver"] = map_solver
                    result["hs_solver"] = None
                else:
                    result["hs_solver"] = hs_solver
                    result["map_solver"] = None
                result["status"] = "error"
                result["error_message"] = str(e)
            write_results_to_csv(result, fieldnames, output_file)


_XCSP_FIELDNAMES = [
    "instance", "algorithm", "solver", "map_solver", "hs_solver",
    "status", "runtimes", "error_message", "MUSes"
]
_XCSP_FIELDNAMES_TOPK = ["instance", "algorithm", "solver", "map_solver",
    "status", "diversity_curve", "error_message"]


def run_single_xcsp_instance(queue, path, filename, algorithm, solver, map_solver, hs_solver, time_limit, num_mus):
    """
        Run one XCSP instance for the given algorithm.
    """
    fieldnames = ["instance", "algorithm", "solver", "map_solver", "hs_solver", "status", "runtimes", "error_message", "MUSes"]

    result = dict.fromkeys(fieldnames)  # initialize result dict with empty values
    result["instance"] = filename
    result["algorithm"] = algorithm
    result["solver"] = solver
    result["map_solver"] = map_solver if algorithm not in ["ocus_enum1", "ocus_enum_shrink", "ocus_enum_opt"] else None
    result["hs_solver"] = hs_solver if algorithm in ["ocus_enum1", "ocus_enum_shrink", "ocus_enum_opt"] else None
    result["error_message"] = None
    result["status"] = "STARTED"  # default unless proven otherwise

    model = None
    generator = None
    
    runtimes = []
    muses = []
        
    try:
        model = cp.Model().from_file(path)
        constraint_to_idx = {id(c): i for i, c in enumerate(model.constraints)}

        if algorithm == "marco":
            generator = timed_marco(model.constraints,solver=solver,map_solver=map_solver, time_limit=time_limit)
        # elif algorithm == "marco_select_top_k":
            #
        elif algorithm == "marco_diverse_noMin":
            generator = marco_diverse_noMin(model.constraints,solver=solver,map_solver=map_solver, time_limit=time_limit)
        elif algorithm == "marco_diverse_min":
            generator = marco_diverse_Min(model.constraints,solver=solver,map_solver=map_solver, time_limit=time_limit)
        elif algorithm == "marco_diverse_opt":
            generator = marco_diverse_optimal(model.constraints, solver=solver, map_solver=map_solver, time_limit=time_limit)
        elif algorithm == "ocus_enum1":
            generator = ocus_enum_1(model.constraints,solver=solver,hs_solver=hs_solver, time_limit=time_limit)
        elif algorithm == "ocus_enum_shrink":
            generator = ocus_enum_shrink(model.constraints,solver=solver,hs_solver=hs_solver, time_limit=time_limit)
        elif algorithm == "ocus_enum_opt":
            generator = ocus_enum_opt_nextMUS(model.constraints,solver=solver,hs_solver=hs_solver, time_limit=time_limit)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        for status, subset, elapsed_time in generator:
            if status == "MUS":
                muses.append([constraint_to_idx[id(c)] for c in subset])
                runtimes.append(elapsed_time)
                if len(muses) >= num_mus:
                    result["status"] = "COMPLETE"
                    break
            elif status == "MCS":
                # this should never happen
                raise ValueError(f"Received a MCS instead of MUS")
            elif status == "TIMEOUT":
                result["status"] = "TIMEOUT"
                break
            else:
                raise ValueError(f"Unknown generator status: {status}")
        if result["status"] == "STARTED":
            # In case generator ends naturally before num_mus
            result["status"] = "EXHAUSTED"  
    except Exception as e:
        result["status"] = "ERROR"
        result["error_message"] = f"{type(e).__name__}: {e}"
    finally:
        result["MUSes"] = muses
        result["runtimes"] = runtimes
        # Explicit cleanup inside subprocess
        del generator
        del model
        del muses
        del runtimes
        gc.collect()
        
        queue.put(result)


def run_single_xcsp_instance_top_k(queue, path, filename, solver, map_solver, time_limit, num_mus):
    """
        Run one XCSP instance for marco until diverse.
    """
    result = dict.fromkeys(_XCSP_FIELDNAMES_TOPK)  # initialize result dict with empty values
    result["instance"] = filename
    result["algorithm"] = "marco_until_diverse"
    result["solver"] = solver
    result["map_solver"] = map_solver
    result["error_message"] = None
    result["status"] = "STARTED"  # default unless proven otherwise

    model = None
    curve = []

    try:
        model = cp.Model().from_file(path)

        status, curve = marco_until_diverse(model.constraints, num_mus, time_limit=time_limit, solver=solver, map_solver=map_solver)
        if status == "COMPLETE":
            result["status"] = "COMPLETE"
        elif status == "TIMEOUT":
            result["status"] = "TIMEOUT"
        else:
            raise ValueError(f"Unknown status: {status}")
    except Exception as e:
        result["status"] = "ERROR"
        result["error_message"] = f"{type(e).__name__}: {e}"
    finally:
        result["diversity_curve"] = curve
        # Explicit cleanup inside subprocess
        del model
        del curve
        gc.collect()

        queue.put(result)


def _run_xcsp_task(path, filename, algorithm, solver, map_solver, hs_solver, time_limit, num_mus):
    """
        Create  a subprocess for one (filename, algorithm) pair and return the result dict.
    """
    queue = mp.Queue()
    proc = mp.Process(
        target=run_single_xcsp_instance,
        args=(queue, path, filename, algorithm, solver, map_solver, hs_solver, time_limit, num_mus)
    )
    proc.start()
    proc.join(timeout=time_limit + 60)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()

    if not queue.empty():
        result = queue.get()
    else:
        result = dict.fromkeys(_XCSP_FIELDNAMES)
        result["instance"] = filename
        result["algorithm"] = algorithm
        result["solver"] = solver
        result["map_solver"] = map_solver if algorithm not in ["ocus_enum1", "ocus_enum_shrink", "ocus_enum_opt"] else None
        result["hs_solver"] = hs_solver if algorithm in ["ocus_enum1", "ocus_enum_shrink", "ocus_enum_opt"] else None
        result["status"] = "ERROR"
        result["error_message"] = f"Subprocess exited with code {proc.exitcode}"
        result["runtimes"] = []
        result["MUSes"] = []

    queue.close()
    queue.join_thread()
    del queue
    del proc
    gc.collect()
    return result


def _run_xcsp_top_k(path, filename, solver, map_solver, time_limit, num_mus):
    """
        Create a subprocess for one instance and return the result dict.
    """
    queue = mp.Queue()
    proc = mp.Process(
        target=run_single_xcsp_instance_top_k,
        args=(queue, path, filename, solver, map_solver, time_limit, num_mus)
    )
    proc.start()
    proc.join(timeout=time_limit + 60)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()

    if not queue.empty():
        result = queue.get()
    else:
        result = dict.fromkeys(_XCSP_FIELDNAMES_TOPK)
        result["instance"] = filename
        result["algorithm"] = "marco_until_diverse"
        result["solver"] = solver
        result["map_solver"] = map_solver
        result["status"] = "ERROR"
        result["error_message"] = f"Subprocess exited with code {proc.exitcode}"
        result["diversity_curve"] = []

    queue.close()
    queue.join_thread()
    del queue
    del proc
    gc.collect()
    return result


def execute_xcsp_instances(num_mus, solver, map_solver, hs_solver, time_limit, output_file, max_workers=4):
    """
        Starts multi-threaded experiments on XCSP instances with all algorithms except marco until diverse.
    """
    filenames = pd.read_csv(
        DATA_DIR / "constraints_stats.csv",
        usecols=["filename"]
    )["filename"].tolist()

    algorithms = [
        "marco",
        "marco_diverse_noMin",
        "marco_diverse_min",
        "marco_diverse_opt",
        "ocus_enum1",
        "ocus_enum_shrink",
        "ocus_enum_opt"
    ]

    tasks = [
        (filename, algorithm)
        for filename in filenames
        for algorithm in algorithms
    ]

    def run_task(filename, algorithm):
        path = str(DATA_DIR / "XCSP_MUS" / filename)
        print(f"File {filename}, running algorithm: {algorithm}", flush=True)
        result = _run_xcsp_task(path, filename, algorithm, solver, map_solver, hs_solver, time_limit, num_mus)
        write_results_to_csv(result, _XCSP_FIELDNAMES, output_file)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, fn, alg): (fn, alg) for fn, alg in tasks}
        for future in as_completed(futures):
            fn, alg = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Task ({fn}, {alg}) raised unexpected exception: {e}", flush=True)


def execute_xcsp_top_k(num_mus, solver, map_solver, time_limit, output_file, max_workers=4):
    """
        Starts multi-threaded experiments on XCSP instances for MARCO until diverse (top k).
    """
    filenames = pd.read_csv(
        DATA_DIR / "constraints_stats.csv",
        usecols=["filename"]
    )["filename"].tolist()
    tasks = [filename for filename in filenames]
    def run_task(filename):
        path = str(DATA_DIR / "XCSP_MUS" / filename)
        print(f"File {filename}, running marco until diverse", flush=True)
        result = _run_xcsp_top_k(path, filename, solver, map_solver, time_limit, num_mus)
        write_results_to_csv(result, _XCSP_FIELDNAMES_TOPK, output_file)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task, fn): fn for fn in tasks}
        for future in as_completed(futures):
            fn = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Task ({fn}) raised unexpected exception: {e}", flush=True)
