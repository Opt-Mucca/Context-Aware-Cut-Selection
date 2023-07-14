#! /usr/bin/env python
import os
import argparse
import yaml
from utilities import is_dir, is_file, build_scip_model, get_filename, str_to_bool
import parameters


def run_instance(results_dir, instance_path, instance, rand_seed, permutation_seed, time_limit, print_stats,
                 print_sol_path, solution_path=None, default_cutsel=False, isp_weight=0.1, obp_weight=0.1,
                 eff_weight=1.0, exp_weight=0.0, psc_weight=0.0, loc_weight=0.0,
                 sparsity_bonus=0.2, end_sparsity_bonus=0.4, root_budget=2,
                 tree_budget=2, max_parallel=0.9, parallel_penalty=0.2, max_density=0.4,
                 filter_dense_cuts=False, filter_parallel_cuts=True, penalise_locks=False,
                 penalise_obp=False):
    """
    The call to solve a single instance. A model will be created and an instance file (and potentially solution file)
    loaded in. Appropriate settings as defined by this function call are then set and the model solved.
    After the model is solved, found infeasible, or some time limit hit, information is extracted and put into
    a yml file. All calls to solve an instance should go through this function and 'run_python_slurm_job' in
    utilities.py.
    Args:
        results_dir: The directory in which all result files will be stored
        instance_path: The path to the MIP
        instance: The instance base name of the MIP file
        rand_seed: The random seed which will be used to shift all SCIP randomisation
        permutation_seed: The random seed which will be used to permute the problem before solving
        time_limit: The time limit, if it exists for our SCIP instance (in seconds).
        print_stats: Whether the .stats file from the run should be printed or not
        print_sol_path: If not None then the best solution at time of completion be printed to print_sol_path
        solution_path: The path to the solution file which will be loaded
        default_cutsel: Is the default cut selector used
        isp_weight: The weight of integer support
        obp_weight: The weight of objective parallelism
        eff_weight: The weight of normalised efficacy
        exp_weight: The weight of normalised expected objective improvement
        psc_weight: The weight of normalised pseudo-costs
        loc_weight: The weight of normalised num locks
        sparsity_bonus: The max bonus given if below some threshold of sparsity
        end_sparsity_bonus: When the sparsity bonus reaches 0
        root_budget: The non-zero budget per separation round at the root node
        tree_budget: The non-zero budget per separation round after the root node
        max_parallel: Maximum parallelism for two cuts to be considered sufficiently different
        parallel_penalty: The penalty awarded to remaining cuts that are considered too parallel to selected one
        max_density: The maximum density of an allowed cut if density filtering is applied
        filter_dense_cuts: Whether density filtering should be applied
        filter_parallel_cuts: Whether parallelism based filtering should be applied
        penalise_locks: Whether locks should be penalised instead of rewarded
        penalise_obp: Whether objective parallelism should be penalised
    Returns:
        Nothing. All results from this run should be output to a file in results_dir.
        The results should contain all information about the run, (e.g. solve_time, dual_bound etc)
    """

    # Make sure the input is of the right type
    assert type(time_limit) == int and time_limit > 0
    assert is_dir(results_dir)
    assert is_file(instance_path)
    assert instance == os.path.split(instance_path)[-1].split('.mps')[0]
    assert type(rand_seed) == int and rand_seed >= 0
    assert isinstance(print_stats, bool)
    if solution_path is not None:
        assert is_file(solution_path) and instance == os.path.split(solution_path)[-1].split('.sol')[0]

    # Set the time limit if None is provided.
    time_limit = None if time_limit < 0 else time_limit
    node_lim = -1
    heuristics = True if solution_path is None else False

    # Build the actual SCIP model from the information now
    scip = build_scip_model(instance_path, node_lim, rand_seed, True, True, True, heuristics,
                            permutation_seed, time_limit, solution_path, default_cutsel)

    # Set cut selector specific parameters
    if not default_cutsel:
        scip.setParam('cutselection/ensemble/efficacyweight', eff_weight)
        scip.setParam('cutselection/ensemble/endsparsitybonus', end_sparsity_bonus)
        scip.setParam('cutselection/ensemble/expimprovweight', exp_weight)
        if filter_parallel_cuts:
            scip.setParam('cutselection/ensemble/filterparalcuts', filter_parallel_cuts)
        else:
            scip.setParam('cutselection/ensemble/penaliseparalcuts', True)
            scip.setParam('cutselection/ensemble/paralpenalty', parallel_penalty)
        scip.setParam('cutselection/ensemble/intsupportweight', isp_weight)
        scip.setParam('cutselection/ensemble/locksweight', loc_weight)
        scip.setParam('cutselection/ensemble/maxnonzerorootround', root_budget)
        scip.setParam('cutselection/ensemble/maxnonzerotreeround', tree_budget)
        scip.setParam('cutselection/ensemble/maxparal', max_parallel)
        scip.setParam('cutselection/ensemble/maxsparsitybonus', sparsity_bonus)
        scip.setParam('cutselection/ensemble/objparalweight', obp_weight)
        scip.setParam('cutselection/ensemble/pscostweight', psc_weight)
        scip.setParam('cutselection/ensemble/filterdensecuts', filter_dense_cuts)
        scip.setParam('cutselection/ensemble/maxcutdensity', max_density)
        scip.setParam('cutselection/ensemble/penaliselocks', penalise_locks)
        scip.setParam('cutselection/ensemble/penaliseobjparal', penalise_obp)

    # Solve the SCIP model and extract all solve information
    solve_model_and_extract_solve_info(scip, rand_seed, permutation_seed, instance, results_dir,
                                       print_stats=print_stats)

    # Print solution if requested
    if print_sol_path is not None:
        scip.writeBestSol(filename=print_sol_path)

    # Free the SCIP instance
    scip.freeProb()

    return


def solve_model_and_extract_solve_info(scip, rand_seed, permutation_seed, instance, results_dir,
                                       print_stats=False):
    """
    Solves the given SCIP model and after solving creates a YML file with all potentially interesting
    solve information. This information will later be read and used to update the neural_network parameters
    Args:
        scip: The PySCIPOpt model that we want to solve
        rand_seed: The random seed used in the scip parameter settings
        permutation_seed: The random seed used to permute the problems rows and columns before solving
        instance: The instance base name of our problem
        results_dir: The directory in which all result files will be stored
        print_stats: A kwarg that informs if the .stats file from the run should be saved
    Returns:
        Nothing. A YML results file is created
    """

    # Solve the MIP instance. All parameters should be pre-set
    scip.optimize()

    # Initialise the dictionary that will store our solve information
    data = {}

    # Get the solve_time
    data['solve_time'] = scip.getSolvingTime()
    # Get the presolve time
    data['presolve_time'] = scip.getPresolvingTime()
    # Get the number of cuts applied
    data['num_cuts'] = scip.getNCutsApplied()
    # Get the number of nodes in our branch and bound tree
    data['num_nodes'] = scip.getNTotalNodes()
    # Get the best primal solution if available
    data['primal_bound'] = scip.getObjVal() if len(scip.getSols()) > 0 else 1e+20
    # Get the gap provided a primal solution exists
    data['gap'] = scip.getGap() if len(scip.getSols()) > 0 else 1e+20
    # Get the best dual bound
    data['dual_bound'] = scip.getDualbound()
    # Get the number of LP iterations
    data['num_lp_iterations'] = scip.getNLPIterations()
    # Get status of solve
    data['status'] = scip.getStatus()
    # Get the primal-dual difference
    data['primal_dual_difference'] = data['primal_bound'] - data['dual_bound'] if len(scip.getSols()) > 0 else 1e+20
    # Get the number of separation rounds
    data['num_sepa_rounds'] = scip.getNSepaRounds()

    # Get the primal dual integral.
    # It is only accessible through the solver statistics. TODO: Write a wrapper function for this
    stat_file = get_filename(results_dir, instance, rand_seed, permutation_seed=permutation_seed, ext='stats')
    assert not os.path.isfile(stat_file)
    scip.writeStatistics(filename=stat_file)
    if data['status'] in ['optimal', 'timelimit', 'nodelimit', 'memlimit']:
        with open(stat_file) as s:
            stats = s.readlines()
        # TODO: Make this safer to access.
        for line_i, line in enumerate(stats):
            if 'primal-dual' in line:
                data['primal_dual_integral'] = float(line.split(':')[1].split('     ')[1])
            if 'number of runs' in line:
                data['num_runs'] = float(line.split(':')[1].strip(' '))
            if 'LP                 ' in line:
                sb_iters = int(stats[line_i + 7][42:53])
                conflict_iters = int(stats[line_i + 7][42:53])
                data['num_total_lp_iterations'] = data['num_lp_iterations'] + sb_iters + conflict_iters

    # If we haven't asked to save the file, then remove it.
    if not print_stats:
        os.remove(stat_file)

    # Dump the yml file containing all of our solve info into the right place
    yml_file = get_filename(results_dir, instance, rand_seed=rand_seed, permutation_seed=permutation_seed, ext='yml')
    with open(yml_file, 'w') as s:
        yaml.dump(data, s)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('instance_path', type=is_file)
    parser.add_argument('instance', type=str)
    parser.add_argument('rand_seed', type=int)
    parser.add_argument('permutation_seed', type=int)
    parser.add_argument('time_limit', type=int)
    parser.add_argument('print_stats', type=str_to_bool)
    parser.add_argument('print_sol_path', type=str)
    parser.add_argument('solution_path', type=str)
    parser.add_argument('default_cutsel', type=str_to_bool)
    parser.add_argument('isp_weight', type=float)
    parser.add_argument('obp_weight', type=float)
    parser.add_argument('eff_weight', type=float)
    parser.add_argument('exp_weight', type=float)
    parser.add_argument('psc_weight', type=float)
    parser.add_argument('loc_weight', type=float)
    parser.add_argument('sparsity_bonus', type=float)
    parser.add_argument('end_sparsity_bonus', type=float)
    parser.add_argument('root_budget', type=float)
    parser.add_argument('tree_budget', type=float)
    parser.add_argument('max_parallel', type=float)
    parser.add_argument('parallel_penalty', type=float)
    parser.add_argument('max_density', type=float)
    parser.add_argument('filter_dense_cuts', type=str_to_bool)
    parser.add_argument('filter_parallel_cuts', type=str_to_bool)
    parser.add_argument('penalise_locks', type=str_to_bool)
    parser.add_argument('penalise_obp', type=str_to_bool)
    args = parser.parse_args()

    # Check if the solution file exists
    if args.solution_path == 'None':
        args.solution_path = None
    else:
        assert os.path.isfile(args.solution_path)
    if args.print_sol_path == 'None':
        args.print_sol_path = None

    # The main function call to run a SCIP instance with cut-sel params
    run_instance(args.results_dir, args.instance_path, args.instance, args.rand_seed, args.permutation_seed,
                 args.time_limit, args.print_stats, args.print_sol_path, args.solution_path, args.default_cutsel,
                 args.isp_weight, args.obp_weight, args.eff_weight, args.exp_weight,
                 args.psc_weight, args.loc_weight, args.sparsity_bonus, args.end_sparsity_bonus,
                 args.root_budget, args.tree_budget, args.max_parallel, args.parallel_penalty, args.max_density,
                 args.filter_dense_cuts, args.filter_parallel_cuts, args.penalise_locks,
                 args.penalise_obp)
