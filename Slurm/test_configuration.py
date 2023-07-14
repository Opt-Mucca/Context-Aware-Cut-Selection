import argparse
import os
import time
from utilities import is_file, is_dir, remove_slurm_files, str_to_bool, run_python_slurm_job
from parameters import *


def test_configuration(config_file, exclusive, instance_dir, solution_dir, results_dir, outfiles_dir):
    """
    The main function for testing a given configuration. It loads the configuration from file, then issues SLURM
    jobs (exclusively if requested) to determine the performance of the new cut selector

    Args:
        config_file (file): .txt file containing all parameter choices
        exclusive (bool): Whether the run should be exclusively run or not
        instance_dir (dir): Directory containing all instance files
        solution_dir (dir): Directory containing all solution files
        results_dir (dir): Directory where all results files will be stored
        outfiles_dir (dir): Directory where all .out files from SLURM will be stored

    Returns:
        Nothing. Scan the results afterwards from the individual files.
    """

    # The list of hyperparameters we've optimised and now want to test over
    hyperparameters = ["isp", "obp", "eff", "exp", "psc", "loc", "sparsity_bonus", "end_sparsity_bonus",
                       "root_budget", "tree_budget", "max_parallel", "parallel_penalty", "max_density",
                       "filter_dense_cuts", "filter_parallel_cuts", "penalise_locks", "penalise_obp"]
    # Get the actual values of the hyperparameters
    config_list = read_config_file(config_file, hyperparameters)

    # Create the results and outfile directory for this SMAC run
    for cut_dir in [outfiles_dir, results_dir]:
        if os.path.isdir(cut_dir):
            remove_slurm_files(cut_dir)
        else:
            os.mkdir(cut_dir)

    # Get all the instance paths
    instance_paths = [os.path.join(instance_dir, file) for file in os.listdir(instance_dir)]
    instances = [path.split('/')[-1].split('.mps')[0] for path in instance_paths]
    if solution_dir is None:
        solution_paths = [None for _ in range(len(instances))]
    else:
        solution_paths = [os.path.join(solution_dir, instance + ".sol.gz") for instance in instances]

    # Iterate over all random seed and permutation seed combinations
    slurm_job_ids = []
    scip_time_lim = SCIP_TEST_TIME_LIMIT
    for p in PERMUTATION_SEEDS:
        for r in RANDOM_SEEDS:
            for i, instance in enumerate(instances):
                ji = run_python_slurm_job(python_file='Slurm/solve_instance.py',
                                          job_name='{}--{}--{}'.format(instance, r, p),
                                          outfile=os.path.join(outfiles_dir, '%j__{}__{}__{}.out'.format(
                                              instance, r, p)),
                                          time_limit=2*scip_time_lim,
                                          arg_list=[results_dir, instance_paths[i], instance, r,
                                                    p, scip_time_lim, True, None,
                                                    solution_paths[i], False] + config_list,
                                          exclusive=exclusive
                                          )
                slurm_job_ids.append(ji)

    # Wait for the jobs to finish
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


def read_config_file(config_file, hyperparameters):
    """
    Read the configuration .txt file for parameter values
    Args:
        config_file (file): The configuration file containing all hyperparameter values
        hyperparameters (list): List of hyperparameters we expect to be in the .txt file

    Returns:
        The hyperparameter values
    """

    with open(config_file, 'r') as s:
        lines = s.readlines()

    config_dict = {}

    for line in lines:
        for hyperparameter in hyperparameters:
            if line.startswith(hyperparameter + ": "):
                config_dict[hyperparameter] = line.split(": ")[-1].split("\n")[0]
                if config_dict[hyperparameter] == "True":
                    config_dict[hyperparameter] = True
                elif config_dict[hyperparameter] == "False":
                    config_dict[hyperparameter] = False
                else:
                    config_dict[hyperparameter] = float(config_dict[hyperparameter])

    return [config_dict[hyperparameter] for hyperparameter in hyperparameters]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=is_file)
    parser.add_argument('exclusive', type=str_to_bool)
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('solution_dir', type=str)
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('outfiles_dir', type=is_dir)
    args = parser.parse_args()

    if args.solution_dir == "None":
        args.solution_dir = None

    test_configuration(args.config_file, args.exclusive, args.instance_dir, args.solution_dir,
                       args.results_dir, args.outfiles_dir)
