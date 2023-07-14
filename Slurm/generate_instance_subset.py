#! /usr/bin/env python
import os
import time
import yaml
import pdb
import shutil
import argparse
from utilities import run_python_slurm_job, get_filename, remove_temp_files, is_dir
from parameters import *


def filter_instances(instance_paths, solution_dir, results_dir, outfiles_dir, instance_subset_dir,
                     load_solutions=False, print_solutions=False):
    """
    The main function call for filtering instances from consideration

    Args:
        instance_paths (list): The list of paths to each instance from our original set
        solution_dir (dir): The directory where we will either dump or load solutions from
        results_dir (dir): The directory where all intermediate results we will be stored
        outfiles_dir (dir): The directory where all intermediate .out files from SLURM will be stored
        instance_subset_dir (dir): The directory where we will move our reduced instance set after filtering
        load_solutions (bool): Whether solutions should be pre-loaded for each instance
        print_solutions (bool): Whether solutions should be printed for each solution after being solved

    Returns:

    """

    # Remove former output and results files
    remove_temp_files(results_dir)
    remove_temp_files(outfiles_dir)

    # Get the instance files and the appropriate solution file names
    instance_files = [instance_path.split('/')[-1] for instance_path in instance_paths]
    instances = [instance_file.split('.mps')[0] for instance_file in instance_files]
    solution_paths = [os.path.join(solution_dir, instance + '.sol') for instance in instances]

    # Submit all the jobs
    slurm_job_ids = []
    for i in range(len(instances)):
        r = 0
        p = 0
        load_solution_path = solution_paths[i] if load_solutions else None
        print_solution_path = solution_paths[i] if print_solutions else None
        if load_solutions:
            scip_time_lim = TRAINING_SET_TIME_LIMIT_WITH_SOLUTION
        else:
            scip_time_lim = TRAINING_SET_TIME_LIMIT_WITHOUT_SOLUTION
        ji = run_python_slurm_job(python_file='Slurm/solve_instance.py',
                                  job_name='{}--{}--{}'.format(instances[i], r, p),
                                  outfile=os.path.join(outfiles_dir, '%j__{}__{}__{}.out'.format(
                                      instances[i], r, p)),
                                  time_limit=2*scip_time_lim,
                                  arg_list=[results_dir, instance_paths[i], instances[i], r,
                                            p, scip_time_lim, True, print_solution_path,
                                            load_solution_path, True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 1.0, 1, 1, 1.0, 0.2, 0.4, False, True, False, False]
                                  )
        slurm_job_ids.append(ji)

    # Wait for the jobs to finish
    wait_for_jobs_to_finish(slurm_job_ids, outfiles_dir)

    # If we just wanted to generate the solution then return all instances
    if print_solutions:
        return instance_paths

    # Now get the results of all instances and reject those that failed some criteria
    good_instance_ids = []
    for i in range(len(instances)):
        yml_file = get_filename(results_dir, instances[i], rand_seed=0, permutation_seed=0, ext='yml')
        if not os.path.isfile(yml_file):
            continue
        with open(yml_file) as s:
            instance_data = yaml.safe_load(s)
        # Only add the instance to candidate instances if it is suitable
        if instance_data['status'] == 'optimal':
            if TRAINING_SET_MIN_NODES <= instance_data['num_nodes'] <= TRAINING_SET_MAX_NODES:
                if TRAINING_SET_MIN_TIME_LIMIT <= instance_data['solve_time']:
                    if instance_data['presolve_time'] <= TRAINING_SET_PRESOLVE_TIME_LIMIT:
                        good_instance_ids.append(i)

    # Remove old training instances and copy over new ones
    remove_temp_files(instance_subset_dir)
    for i in good_instance_ids:
        shutil.copy(instance_paths[i], os.path.join(instance_subset_dir, instance_files[i]))

    print('Selected {} from {} many instances! Instance Selected: {}'.format(
        len(good_instance_ids), len(instances), [instances[i] for i in good_instance_ids]), flush=True)

    return [instance_paths[i] for i in good_instance_ids]


def wait_for_jobs_to_finish(slurm_job_ids, outfiles_dir):
    """
    Function for waiting on SLURM jobs to finish
    Args:
        slurm_job_ids (list): List of SLURM job IDs that we're waiting on before continuing
        outfiles_dir (dir): Directory where the signal file should be created

    Returns:
        Nothing. Just finished when all jobs are done.
    """

    # Now submit the checker job that has dependencies slurm_job_ids
    safety_file_no_ext = os.path.join(outfiles_dir, 'safety_check')
    _ = run_python_slurm_job(python_file='Slurm/safety_check.py',
                             job_name='cleaner',
                             outfile=safety_file_no_ext + '.out',
                             time_limit=10,
                             arg_list=[safety_file_no_ext + '.txt'],
                             dependencies=slurm_job_ids)
    # Put the program to sleep until all of slurm jobs are complete
    time.sleep(5)
    while not os.path.isfile(safety_file_no_ext + '.txt'):
        time.sleep(5)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_dir", type=is_dir)
    parser.add_argument("solution_dir", type=str)
    parser.add_argument("instance_subset_dir", type=is_dir)
    parser.add_argument("outfiles_dir", type=is_dir)
    parser.add_argument("results_dir", type=is_dir)
    args = parser.parse_args()

    full_instance_files = sorted(os.listdir(args.instance_dir))
    full_instance_paths = [os.path.join(args.instance_dir, instance_file) for instance_file in full_instance_files]

    # Remove solution files (will generate them ourselves)
    remove_temp_files(args.solution_dir)

    full_instance_paths = filter_instances(full_instance_paths, args.solution_dir, args.results_dir, args.outfiles_dir,
                                           args.instance_subset_dir, False, False)
    full_instance_paths = filter_instances(full_instance_paths, args.solution_dir, args.results_dir, args.outfiles_dir,
                                           args.instance_subset_dir, False, True)
    full_instance_paths = filter_instances(full_instance_paths, args.solution_dir, args.results_dir, args.outfiles_dir,
                                           args.instance_subset_dir, True, False)

