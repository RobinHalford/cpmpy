"""
PyTorch-style Dataset for Nurserostering instances from schedulingbenchmarks.org

Simply create a dataset instance and start iterating over its contents:
The `metadata` contains usefull information about the current problem instance.
"""
import os
import pathlib
from io import StringIO
from os.path import join
from typing import Tuple, Any
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import zipfile
import csv
import pickle
from filelock import FileLock
import pandas as pd
import time
from cpmpy.tools.explain import marco
from cpmpy.tools.explain.mus import smus 
from cpmpy.tools.explain.diverse_enum_assumps import marco_assumps, marco_diverse_Min_assump, marco_diverse_noMin_assump, ocus_enum_1_assump, ocus_enum_shrink_assump
from cpmpy.tools.explain.utils import average_diversity
from examples.nurserostering import NurseRosteringDataset, nurserostering_model, parse_scheduling_period


import cpmpy as cp


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
                    generator = enumerate(marco_assumps(cmodel.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=20))
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
                    generator = enumerate(marco_assumps(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
                elif algorithm == "marco_select_top_k":
                    ... # TODO (not enumaration function)
                elif algorithm == "marco_diverse_min":
                    generator = enumerate(marco_diverse_Min_assump(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
                elif algorithm == "marco_diverse_no_min":
                    generator = enumerate(marco_diverse_noMin_assump(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
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


def execute_xcsp_instances(num_mus, solver, map_solver, hs_solver, time_limit, output_file):
    filenames = pd.read_csv("cpmpy/tools/explain/experiments_diverse/data/constraints_stats.csv", usecols=["filename"])["filename"].tolist()
    # for now only run on 1 instance for testing

    fieldnames = ["instance", "algorithm", "solver", "map_solver", "hs_solver", "status", "runtimes", "error_message", "MUSes"]
    algorithms = ["marco", "marco_diverse_noMin", "marco_diverse_min", "ocus_enum1", "ocus_enum_shrink"]

    for filename in filenames[:1]:
        # load instance from pickle file
        path = "cpmpy/tools/explain/experiments_diverse/data/XCSP_MUS/" + filename
        print(f"Processing file: {filename}")
        
        for algorithm in algorithms:
            result = dict.fromkeys(fieldnames)  # initialize result dict with empty values
            result["instance"] = filename
            result["algorithm"] = algorithm
            result["solver"] = solver
            result["map_solver"] = map_solver if algorithm not in ["ocus_enum1", "ocus_enum_shrink"] else None
            result["hs_solver"] = hs_solver if algorithm in ["ocus_enum1", "ocus_enum_shrink"] else None
            result["error_message"] = None
            result["status"] = "STARTED"  # default unless proven otherwise

            model = cp.Model().from_file(path)
           
            runtimes = []
            muses = []
            
            try:
                if algorithm == "marco":
                    generator = marco_assumps(model.constraints,solver=solver,map_solver=map_solver, time_limit=time_limit)
                elif algorithm == "marco_diverse_noMin":
                    generator = marco_diverse_noMin_assump(model.constraints,solver=solver,map_solver=map_solver, time_limit=time_limit)
                elif algorithm == "marco_diverse_min":
                    generator = marco_diverse_Min_assump(model.constraints,solver=solver,map_solver=map_solver, time_limit=time_limit)
                elif algorithm == "ocus_enum1":
                    generator = ocus_enum_1_assump(model.constraints,solver=solver,hs_solver=hs_solver, time_limit=time_limit)
                elif algorithm == "ocus_enum_shrink":
                    generator = ocus_enum_shrink_assump(model.constraints,solver=solver,hs_solver=hs_solver, time_limit=time_limit)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")

                for status, subset, elapsed_time in generator:
                    if status == "MUS":
                        muses.append(subset)
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
                    
            except Exception as e:
                result["status"] = "ERROR"
                result["error_message"] = str(e)
            
            result["MUSes"] = muses
            result["runtimes"] = runtimes
            write_results_to_csv(result, fieldnames, output_file)

   
    

# SAT COMPETITION HELPER FUNCTIONS
def enum_sat_competition_instances():
    with open("cpmpy/tools/explain/experiments_diverse/data/unsat_instances_2025.uri", "r") as f:
        count = 0
        for line in f:
            url = line.strip()
            if not url:
                continue
            else:
                # load instance from url and yield model
                try:
                    # download file to temporary location
                    tmp_path = pathlib.Path("cpmpy/tools/explain/experiments_diverse/data/satcompinstances/tmp_instance_" + str(count) + ".txt")
                    urlretrieve(url, str(tmp_path))
                    count += 1
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available on {url}. Error: {str(e)}")
    return


"""
 def get_filename_from_uri(url, response):
    # 1) Try Content-Disposition header
    cd = response.headers.get("Content-Disposition")
    if cd:
        match = FILENAME_RE.search(cd)
        if match:
            return pathlib.Path(match.group(1)).name

    # 2) Fallback: last part of URL path
    path = urlparse(url).path
    name = pathlib.Path(path).name
    return name if name else "downloaded_file"
"""