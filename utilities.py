#! /usr/bin/env python
import os
import sys
import numpy as np
import subprocess
import shutil
import logging
import argparse
from pyscipopt import Model, SCIP_PARAMSETTING
import parameters


def build_scip_model(instance_path, node_lim, rand_seed, pre_solve, propagation, separators, heuristics,
                     permutation_seed=0, time_limit=None, sol_path=None, default_cutsel=False):
    """
    General function to construct a PySCIPOpt model.
    Args:
        instance_path: The path to the instance
        node_lim: The node limit
        rand_seed: The random seed for all SCIP plugins (and LP solver)
        pre_solve: Whether pre-solve should be enabled or disabled
        propagation: Whether propagators should be enabled or disabled
        separators: Whether separators should be enabled or disabled
        heuristics: Whether heuristics should be enabled or disabled
        permutation_seed: The random seed used to permute the rows and columns before solving
        time_limit: The time_limit of the model
        sol_path: An optional path to a valid solution file containing a primal solution to the instance
        default_cutsel: Is the default cut selector used
    Returns:
        pyscipopt model
    """
    assert os.path.exists(instance_path)
    assert type(node_lim) == int and type(rand_seed) == int
    assert all([type(param) == bool for param in [pre_solve, propagation, separators, heuristics]])

    # Create the base PySCIPOpt model and start setting parameters
    scip = Model()
    scip.setParam('limits/nodes', node_lim)
    scip.setParam('limits/memory', parameters.SCIP_TRAIN_MEMORY)
    if time_limit is not None:
        scip.setParam('limits/time', time_limit)

    # WARNING: Because of limit on computational resources, exclusive jobs are not possible.
    # WARNING: From now CPU seconds will be used. (Not entirely deterministic still, but an improvement)
    # WARNING: We also restrict our LP solver to a single thread
    scip.setParam('timing/clocktype', 1)
    scip.setParam('lp/threads', 1)

    if permutation_seed > 0:
        scip.setParam('randomization/permutevars', True)
        scip.setParam('randomization/permutationseed', rand_seed)
    scip.setParam('randomization/randomseedshift', rand_seed)

    # Set presolve, propagation, separator, and heuristic options
    if not pre_solve:
        scip.setPresolve(SCIP_PARAMSETTING.OFF)
    if not separators:
        scip.setSeparating(SCIP_PARAMSETTING.OFF)
    if not separators:
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    if not propagation:
        scip.disablePropagation()

    # Read in the problem
    scip.readProblem(instance_path)

    # Add the solution if it exists to the SCIP model
    if sol_path is not None:
        assert os.path.isfile(sol_path) and '.sol' in sol_path
        # Create the solution to add to SCIP
        sol = scip.readSolFile(sol_path)
        # Add the solution. This automatically frees the loaded solution
        scip.addSol(sol)
        # If a solution is provided, then we are focusing on branching, so disable heuristics
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)

    # Make sure the appropriate cut selector has the highest priority
    if not default_cutsel:
        scip.setParam('cutselection/ensemble/priority', 100000)
    else:
        scip.setParam('cutselection/hybrid/priority', 100000)

    return scip


def remove_slurm_files(outfile_dir):
    """
    Removes all files from outfile_dir.
    Args:
        outfile_dir: The output directory containing all of our slurm .out files
    Returns:
        Nothing. It simply deletes the files
    """

    assert not outfile_dir == '/' and not outfile_dir == ''

    # Delete everything
    shutil.rmtree(outfile_dir)

    # Make the directory itself again
    os.mkdir(outfile_dir)

    return


def remove_temp_files(temp_dir):
    """
    Removes all files from the given directory
    Args:
        temp_dir: The directory containing all information that is batch specific
    Returns:
        Nothing, the function deletes all files in the given directory
    """

    # Get all files in the directory
    files = os.listdir(temp_dir)

    # Now cycle through the files and delete them
    for file in files:
        os.remove(os.path.join(temp_dir, file))

    return


def run_python_slurm_job(python_file, job_name, outfile, time_limit, arg_list, dependencies=None, exclusive=False):
    """
    Function for calling a python file through SLURM. This offloads the job from the current call, and lets multiple
    jobs run simultaneously. Please make sure to set up an appropriate dependency using safety_check.py so
    the main process does not continue to run while the other jobs are still running. All information
    between jobs is communicated through input / output files.
    Note: Spawned processes cannot directly communicate with each other
    Args:
        python_file: The python file that wil be run
        job_name: The name to give the python run in slurm
        outfile: The file in which all output from the python run will be stored
        time_limit: The time limit on the slurm job in seconds
        arg_list: The list containing all args that will be added to the python call
        dependencies: A list of slurm job ID dependencies that must first complete before this job starts
        exclusive: Whether the job is exclusive or not
    Returns:
        Nothing. It simply starts a python job through the command line that will be run in slurm
    """

    if dependencies is None:
        dependencies = []
    assert os.path.isfile(python_file) and python_file.endswith('.py')
    assert not os.path.isfile(outfile) and outfile.endswith('.out'), '{}'.format(outfile)
    assert os.path.isdir(os.path.dirname(outfile)), '{}'.format(outfile)
    assert type(time_limit) == int and 0 <= time_limit <= 1e+8
    assert type(arg_list) == list
    assert dependencies is None or (type(dependencies) == list and
                                    all(type(dependency) == int for dependency in dependencies))

    # Get the current working environment.
    my_env = os.environ.copy()

    # Give the base command line call for running a single slurm job through shell.
    cmd_1 = ['sbatch',
             '--job-name={}'.format(job_name),
             '--time=0-00:00:{}'.format(time_limit),
             '--cpus-per-task=1']
    if exclusive:
        cmd_1 += ['--exclusive']
    else:
        cmd_1 += ['--ntasks=1']

    # This you want to add memory limits
    if exclusive:
        cmd_2 = ['--mem={}'.format(parameters.SLURM_TEST_MEMORY)]
    elif parameters.SLURM_TRAIN_MEMORY is not None:
        cmd_2 = ['--mem={}'.format(parameters.SLURM_TRAIN_MEMORY)]
    else:
        cmd_2 = []

    if dependencies is not None and len(dependencies) > 0:
        # Add the dependencies if they exist
        dependency_str = ''.join([str(dependency) + ':' for dependency in dependencies])[:-1]
        cmd_2 += ['--dependency=afterany:{}'.format(dependency_str)]

    if exclusive:
        cmd_3 = ['--constraint={}'.format(parameters.SLURM_CONSTRAINT),
                 '--output',
                 outfile,
                 '--error',
                 outfile,
                 '-A',
                 'PUT EXCLUSIVE GROUP HERE',
                 '--partition=PUT EXCLUSIVE PARTITION HERE',
                 '{}'.format(python_file)]
    else:
        cmd_3 = ['--constraint={}'.format(parameters.SLURM_CONSTRAINT),
                 '--output',
                 outfile,
                 '--error',
                 outfile,
                 '-A',
                 'PUT NON_EXCLUSIVE GROUP HERE',
                 '--partition=PUT NON_EXCLUSIVE PARTITION HERE',
                 '{}'.format(python_file)]

    cmd = cmd_1 + cmd_2 + cmd_3

    # Add all arguments of the python file afterwards
    for arg in arg_list:
        cmd.append('{}'.format(arg))

    # Run the command in shell.
    p = subprocess.Popen(cmd, env=my_env, stdout=subprocess.PIPE)
    p.wait()

    # Now access the stdout of the subprocess for the job ID
    job_line = ''
    for line in p.stdout:
        job_line = str(line.rstrip())
        break
    assert 'Submitted batch job' in job_line, print(job_line)
    job_id = int(job_line.split(' ')[-1].split("'")[0])

    del p

    return job_id


def get_filename(parent_dir, instance, rand_seed=None, permutation_seed=None, ext='yml'):
    """
    The main function for retrieving the file names for all non-temporary files. It is a shortcut to avoid constantly
    rewriting the names of the different files, such as the .yml, .sol, .stats, and .mps files
    Args:
        parent_dir: The parent directory where the file belongs
        instance: The instance name of the SCIP problem
        rand_seed: The random seed used in the SCIP run
        permutation_seed: The random seed used to permute the variables and constraints before solving
        ext: The extension of the file, e.g. yml or sol
    Returns:
        The filename e.g. 'parent_dir/toll-like__trans__seed__2__permute__0.mps'
    """

    # Initialise the base_file name. This always contains the instance name
    base_file = instance
    if rand_seed is not None:
        base_file += '__seed__{}'.format(rand_seed)
    if permutation_seed is not None:
        base_file += '__permute__{}'.format(permutation_seed)

    # Add the extension to the base file
    if ext is not None:
        base_file += '.{}'.format(ext)

    # Now join the file with its parent dir
    return os.path.join(parent_dir, base_file)


def str_to_bool(word):
    """
    This is used to check if a string is trying to represent a boolean True.
    We need this because argparse doesnt by default have such a function, and using bool('False') evaluates to True
    Args:
        word: The string we want to convert to a boolean
    Returns:
        Whether the string is representing True or not.
    """
    assert type(word) == str
    return word.lower() in ["yes", "true", "t", "1"]


def is_dir(path):
    """
    This is used to check if a string is trying to represent a directory when we parse it into argparse.
    Args:
        path (str): The path to a directory
    Returns:
        The path if it is a valid directory else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isdir(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid directory'.format(path))
    else:
        return path


def is_file(path):
    """
    This is used to check if a string is trying to represent a file when we parse it into argparse.
    Args:
        path (str): The path to a file
    Returns:
        The path if it is a valid file else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isfile(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid file'.format(path))
    else:
        return path


def get_slurm_output_file(outfile_dir, instance, rand_seed=1, permutation_seed=0):
    """
    Function for getting the slurm output log for the current run.
    Args:
        outfile_dir: The directory containing all slurm .log files
        instance: The instance name
        rand_seed: The random seed
        permutation_seed: The permutation seed
    Returns:
        The slurm .out file which is currently being used
    """

    assert os.path.isdir(outfile_dir)
    assert type(instance) == str
    assert type(rand_seed) == int
    assert type(permutation_seed) == int

    # Get all slurm out files
    out_files = os.listdir(outfile_dir)

    # Get a unique substring that will only be contained for a single run
    file_substring = '__{}__{}__{}'.format(instance, rand_seed, permutation_seed)

    unique_file = [out_file for out_file in out_files if file_substring in out_file]
    assert len(unique_file) == 1, 'Instance {}, r {}, p {} has no outfile in {}'.format(
        instance, rand_seed, permutation_seed, outfile_dir)

    return os.path.join(outfile_dir, unique_file[0])
