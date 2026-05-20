"""
Required Arguments
------------------

--num-mus : int
    The number of diverse MUSes to enumerate for each instance.

--solver : str
    The name of the solver to benchmark (e.g., "ortools", "exact", "choco").

--map-solver : str
    The name of the map-solver to benchmark (e.g., "ortools", "exact", "choco").

--hs-solver : str
    The name of the hitting set solver to benchmark (e.g., "ortools", "gurobi").


Optional Arguments
------------------

--time-limit : int, default=60
    Time limit in seconds per instance.

--output-dir : str, default='results'
    Directory where result CSV files will be saved.
"""
from datetime import datetime
import cpmpy as cp
import importlib
importlib.reload(cp)
import argparse
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from cpmpy.tools.explain.experiments_diverse.utils_experiments import execute_xcsp_instances, execute_solver_combinations, execute_xcsp_top_k


def test_solver_combinations(time_limit: int, output_dir: str):
    """
    Test different combinations of solvers and map-solvers on a small set of instances to identify promising configurations for the full benchmarks.

    Args:
        time_limit (int): Time limit in seconds.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Get current timestamp in a filename-safe format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Define output file path with timestamp
    output_file_NR = str(output_dir / f"solver_combinations_NR_{timestamp}.csv")
    output_file_xcsp = str(output_dir / f"solver_combinations_xcsp_{timestamp}.csv")
    execute_solver_combinations(time_limit=time_limit, output_file_NR=output_file_NR, output_file_xcsp=output_file_xcsp)
    return
    
    

def benchmark_xcsp_MUS(num_mus: int, solver: str, 
                 map_solver: str, hs_solver: str,
                 time_limit: int = 60,
                 output_dir: str = 'results') -> str:
    """
    Benchmark diverse MUS enumeration on selection of XCSP instances from 
    https://www.xcsp.org/specifications/ (satisfiable instances were transformed to unsatisfiable instances).

    Returns:
        str: Path to the output CSV file.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Get current timestamp in a filename-safe format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Define output file path with timestamp
    output_file = str(output_dir / f"xcsp_{timestamp}.csv")
    # execute experiments
    execute_xcsp_instances(
                           solver=solver,
                           map_solver=map_solver,
                           hs_solver=hs_solver,
                           time_limit=time_limit,
                           output_file=output_file)
    return output_file



def benchmark_xcsp_marco_select_top_k(num_mus: int, solver: str,
                 map_solver: str,
                 time_limit: int = 60,
                 output_dir: str = 'results') -> str:
    """
    Benchmark marco select-top-k on XCSP instances.
    Runs timed_marco for (time_limit - 2) seconds per instance to generate up to n MUSes,
    then selects the top-k subset by min pairwise diversity using select_top_k.

    Returns:
        str: Path to the output CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = str(output_dir / f"xcsp_marco_select_top_k_{timestamp}.csv")
    execute_xcsp_top_k(num_mus=num_mus,
                       solver=solver,
                       map_solver=map_solver,
                       time_limit=time_limit,
                       output_file=output_file)
    return output_file




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark diverse MUS enumeration given the solvers to use.")
    parser.add_argument('--num-mus', type=int, required=True, help="The number of diverse MUSes to enumerate for each instance.")
    parser.add_argument('--solver', type=str, required=True, help="The SAT solver to use.")
    parser.add_argument('--map-solver', type=str, required=True, help="The map-solver to use.")
    parser.add_argument('--hs-solver', type=str, required=True, help="The hitting set solver to use.")
    parser.add_argument('--time-limit', type=int, default=60, help="Time limit in seconds per instance.")
    parser.add_argument('--output-dir', type=str, default='results', help="Directory where result CSV files will be saved.")
    args = parser.parse_args()

    # use **vars(args) to pass all arguments to the benchmark functions
    # benchmark_NR(**vars(args))
    # benchmark_xcsp_MUS(**vars(args))
    # benchmark_xcsp_until_diverse(**vars(args))
    benchmark_xcsp_marco_select_top_k(num_mus=5, solver="exact", map_solver="exact", time_limit=60, output_dir="results")