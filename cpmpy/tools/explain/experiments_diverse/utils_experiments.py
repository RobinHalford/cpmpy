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
from cpmpy.tools.explain.diverse_enumeration import marco_assumps, marco_diverse, marco_diverse_no_min, ocus_enum
from cpmpy.tools.explain.utils import average_diversity


try:
    from faker import Faker
except ImportError as e:
    print("Install `faker` package using `pip install faker`")
    raise e
try:
    from natsort import natsorted
except ImportError as e:
    print("Install `natsort` package using `pip install natsort`")
    raise e

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

import cpmpy as cp

class NurseRosteringDataset(object):  # torch.utils.data.Dataset compatible

    """
    Nurserostering Dataset in a PyTorch compatible format.

    Arguments:
        root (str): Root directory containing the nurserostering instances (if 'download', instances will be downloaded to this location)
        transform (callable, optional): Optional transform to be applied on the instance data
        target_transform (callable, optional): Optional transform to be applied on the file path
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """

    def __init__(self, root: str = ".", transform=None, target_transform=None, download: bool = False):
        """
        Initialize the Nurserostering Dataset.
        """

        self.root = pathlib.Path(root)
        self.instance_dir = pathlib.Path(join(self.root, "nurserostering"))
        self.transform = transform
        self.target_transform = target_transform

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

        if not self.instance_dir.exists():
            if not download:
                raise ValueError(f"Dataset not found in local file system. Please set download=True to download the dataset.")
            else:
                url = f"https://schedulingbenchmarks.org/nrp/data/instances1_24.zip" # download full repo...
                zip_path = pathlib.Path(join(root,"jsplib-master.zip"))

                print(f"Downloading Nurserostering instances from schedulingbenchmarks.org")

                try:
                    urlretrieve(url, str(zip_path))
                except (HTTPError, URLError) as e:
                    raise ValueError(f"No dataset available on {url}. Error: {str(e)}")

                # make directory and extract files
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    self.instance_dir.mkdir(parents=True, exist_ok=True)

                    # Extract files
                    for file_info in zip_ref.infolist():
                            filename = pathlib.Path(file_info.filename).name
                            with zip_ref.open(file_info) as source, open(self.instance_dir / filename, 'wb') as target:
                                target.write(source.read())

                 # Clean up the zip file
                zip_path.unlink()


    def __len__(self) -> int:
        """Return the total number of instances."""
        return len(list(self.instance_dir.glob("*.txt")))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single Nurserostering instance filename and metadata.

        Args:
            index (int): Index of the instance to retrieve

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - The filename of the instance
                - Metadata dictionary with file name, track, year etc.
        """
        if isinstance(index, int) and not (0 <= index < len(self)):
            raise IndexError("Index out of range")

        # Get all instance files and sort for deterministic behavior
        files = natsorted(list(self.instance_dir.glob("*.txt"))) # use .txt files instead of xml files
        file_path = files[index]

        filename = str(file_path)
        if self.transform:
            # user might want to process the filename to something else
            filename = self.transform(filename)

        metadata = dict(name=file_path.stem)

        if self.target_transform:
            metadata = self.target_transform(metadata)

        return filename, metadata


import re
def _tag_to_data(string, tag, skip_lines=0, datatype=pd.DataFrame, *args, **kwargs):

    regex = rf'{tag}[\s\S]*?($|(?=\n\s*\n))'
    match = re.search(regex, string)

    data = "\n".join(match.group().split("\n")[skip_lines+1:])
    if datatype == pd.DataFrame:
        kwargs = {"header":0, "index_col":0} | kwargs
        df = pd.read_csv(StringIO(data), *args, **kwargs)
        return  df.rename(columns=lambda x: x.replace("#","").strip())
    return datatype(data, *args, **kwargs)

def parse_scheduling_period(fname):
    fake = Faker()
    fake.seed_instance(0)

    with open(fname, "r") as f:
        string = f.read()


    horizon = _tag_to_data(string, "SECTION_HORIZON", skip_lines=2, datatype=int)
    shifts = _tag_to_data(string, "SECTION_SHIFTS", names=["ShiftID", "Length", "cannot follow"],
                          dtype={'ShiftID':str, 'Length':int, 'cannot follow':str})
    shifts.fillna("", inplace=True)
    shifts["cannot follow"] = shifts["cannot follow"].apply(lambda val : [v.strip() for v in val.split("|") if len(v.strip())])

    staff = _tag_to_data(string, "SECTION_STAFF", index_col=False)
    maxes = staff["MaxShifts"].str.split("|", expand=True)
    for col in maxes:
        shift_id = maxes[col].iloc[0].split("=")[0]
        column = maxes[col].apply(lambda x : x.split("=")[1])
        staff[f"max_shifts_{shift_id}"] = column.astype(int)

    staff["name"] = [fake.unique.first_name() for _ in staff.index]

    days_off = _tag_to_data(string, "SECTION_DAYS_OFF", datatype=str)
    # process string to be EmployeeID, Day off for each line
    rows = []
    for line in days_off.split("\n")[1:]:
        employee_id , *days = line.split(",")
        rows += [dict(EmployeeID=employee_id, DayIndex= int(d)) for d in days]
    days_off = pd.DataFrame(rows)


    shift_on = _tag_to_data(string, "SECTION_SHIFT_ON_REQUESTS", index_col=False)
    shift_off = _tag_to_data(string, "SECTION_SHIFT_OFF_REQUESTS", index_col=False)
    cover = _tag_to_data(string, "SECTION_COVER", index_col=False)

    return dict(horizon=horizon, shifts=shifts, staff=staff, days_off=days_off, shift_on=shift_on, shift_off=shift_off, cover=cover)


def nurserostering_model(horizon, shifts:pd.DataFrame, staff, days_off, shift_on, shift_off, cover):

    n_nurses = len(staff)

    FREE = 0
    SHIFTS = ["F"] + list(shifts.index)

    nurse_view = cp.intvar(0,len(shifts), shape=(n_nurses, horizon), name="nv")

    model = cp.Model()

    # Shifts which cannot follow the shift on the previous day.
    for id, shift in shifts.iterrows():
        for other_shift in shift['cannot follow']:
            model += (nurse_view[:,:-1] == SHIFTS.index(id)).implies(nurse_view[:,1:] != SHIFTS.index(other_shift))

    # Maximum number of shifts of each type that can be assigned to each employee.
    for i, nurse in staff.iterrows():
        for shift_id, shift in shifts.iterrows():
            max_shifts = nurse[f"max_shifts_{shift_id}"]
            model += cp.Count(nurse_view[i], SHIFTS.index(shift_id)) <= max_shifts

    # Minimum and maximum amount of total time in minutes that can be assigned to each employee.
    shift_length = cp.cpm_array([0] + shifts['Length'].tolist()) # FREE = length 0
    for i, nurse in staff.iterrows():
        time_worked = cp.sum(shift_length[nurse_view[i,d]] for d in range(horizon))
        model += time_worked <= nurse['MaxTotalMinutes']
        model += time_worked >= nurse['MinTotalMinutes']

    # Maximum number of consecutive shifts that can be worked before having a day off.
    for i, nurse in staff.iterrows():
        max_days = nurse['MaxConsecutiveShifts']
        for d in range(horizon - max_days):
            window = nurse_view[i,d:d+max_days+1]
            model += cp.Count(window, FREE) >= 1 # at least one holiday in this window

    # Minimum number of concecutive shifts that must be worked before having a day off.
    for i, nurse in staff.iterrows():
        min_days = nurse['MinConsecutiveShifts']
        for d in range(1,horizon):
            is_start_of_working_period = (nurse_view[i, d-1] == FREE) & (nurse_view[i, d] != FREE)
            model += is_start_of_working_period.implies(cp.all(nurse_view[i,d:d+min_days] != FREE))

    # Minimum number of concecutive days off.
    for i, nurse in staff.iterrows():
        min_days = nurse['MinConsecutiveDaysOff']
        for d in range(1,horizon):
            is_start_of_free_period = (nurse_view[i, d - 1] != FREE) & (nurse_view[i, d] == FREE)
            model += is_start_of_free_period.implies(cp.all(nurse_view[i, d:d + min_days] == FREE))

    # Max number of working weekends for each nurse
    weekends = [(i - 1, i) for i in range(1,horizon) if (i + 1) % 7 == 0]
    for i, nurse in staff.iterrows():
        n_weekends = cp.sum((nurse_view[i,sat] != FREE) | (nurse_view[i,sun] != FREE) for sat,sun in weekends)
        model += n_weekends <= nurse['MaxWeekends']

    # Days off
    for _, holiday in days_off.iterrows(): # could also do this vectorized... TODO?
        i = (staff['ID'] == holiday['EmployeeID']).argmax() # index of employee
        model += nurse_view[i,holiday['DayIndex']] == FREE

    # Shift requests, encode in linear objective
    objective = 0
    for _, request in shift_on.iterrows():
        i = (staff['ID'] == request['EmployeeID']).argmax() # index of employee
        cpm_request = nurse_view[i, request['Day']] == SHIFTS.index(request['ShiftID'])
        objective += request['Weight'] * ~cpm_request

    # Shift off requests, encode in linear objective
    for _, request in shift_off.iterrows():
        i = (staff['ID'] == request['EmployeeID']).argmax() # index of employee
        cpm_request = nurse_view[i, request['Day']] != SHIFTS.index(request['ShiftID'])
        objective += request['Weight'] * ~cpm_request

    # Cover constraints, encode in objective with slack variables
    for _, cover_request in cover.iterrows():
        nb_nurses = cp.Count(nurse_view[:, cover_request['Day']], SHIFTS.index(cover_request['ShiftID']))
        slack_over, slack_under = cp.intvar(0, len(staff), shape=2)
        model += nb_nurses - slack_over + slack_under == cover_request["Requirement"]

        objective += cover_request["Weight for over"] * slack_over + cover_request["Weight for under"] * slack_under

    model.minimize(objective)
    return model, nurse_view


def get_optimal(model, time_limit=60, solver="ortools"):
    model.solve(time_limit=time_limit, solver=solver)
    print(model.objective_value())
    return model.objective_value()


def create_unsat_model(model, optimal, difficulty_factor):
    objective = model.objective_
    model.objective_ = None
    model += objective <= int(difficulty_factor * optimal)
    return model


def execute_unsat_nr_models(num_mus, solver, map_solver, hs_solver, difficulty_factor, time_limit, output_file):
    dataset = NurseRosteringDataset(root=".", download=True, transform=parse_scheduling_period)
    fieldnames = ["instance", "algorithm", "solver", "map_solver", "hs_solver", "status", "runtimes", "error_message", "MUSes"]
    for i in range(0,1):
        data, metadata = dataset[i]
        # run all algorithms
        for algorithm in ["marco", # "marco_select_top_k", "marco_until_diverse", "ocus_enum",
                          "marco_diverse_no_min",
                          "marco_diverse"]:
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
                elif algorithm == "marco_diverse":
                    generator = enumerate(marco_diverse(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
                elif algorithm == "marco_diverse_no_min":
                    generator = enumerate(marco_diverse_no_min(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
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


def execute_xcsp_instances(num_mus, solver, map_solver, hs_solver, time_limit, output_file):
    filenames = pd.read_csv("cpmpy/tools/explain/experiments_diverse/data/constraints_stats.csv", usecols=["filename"])["filename"].tolist()
    # for now only run on 1 instance for testing
    for filename in filenames[:1]:
        # load instance from pickle file
        path = "cpmpy/tools/explain/experiments_diverse/data/XCSP_MUS/" + filename
        print(f"Processing file: {filename}")
        model = cp.Model().from_file(path)
        print("loaded model")
        assert model.solve(time_limit=60, solver="ortools") is False
        print("asserted model is unsat")
        print(model.status())
        generator = enumerate(marco_assumps(model.constraints, solver=solver, map_solver=map_solver, return_mcs=False, time_limit=time_limit))
        muses = []
        for c in model.constraints:
            print(c)
        print("done")
        """
        while True:
            try:
                j, (_, subset) = next(generator)
                print("found a MUS")
            except StopIteration:
                break
            muses.append(subset)
            if j == num_mus - 1:
                break
        diversity = average_diversity(muses)
        print(f"Found {len(muses)} MUSes with average pairwise diversity of {diversity:.4f}")
        """

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