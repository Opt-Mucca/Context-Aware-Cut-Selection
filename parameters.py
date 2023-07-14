# Make sure to set this before you begin any runs. This is the SLURM node list that'll be used
# TODO: The user must define this by themselves!!!!!
SLURM_CONSTRAINT = 'Xeon&Gold6342'

# If you want to implement memory limits on the individual slurm jobs (units are megabytes). Set to None to ignore.
# This was set to 2000 in the paper.
SLURM_TRAIN_MEMORY = 2000
SCIP_TRAIN_MEMORY = 2000
SLURM_TEST_MEMORY = 90000
SCIP_TEST_MEMORY = 48000

# The random seeds used in all experiments
# These were [1,2,3, 4, 5] in the paper
RANDOM_SEEDS = [1, 2, 3, 4, 5]

# The permutation seeds used in all experiments
# These were [0] in the paper
PERMUTATION_SEEDS = [0]

# The time limit for all exclusive runs that were used. Time in seconds.
# This was set to 7200 in the paper
SCIP_TEST_TIME_LIMIT = 7200

# The time limit for all runs when selecting a training set (when ran without a solution)
TRAINING_SET_TIME_LIMIT_WITHOUT_SOLUTION = 120

# The time limit for all runs when selecting a training set (when ran with a solution)
TRAINING_SET_TIME_LIMIT_WITH_SOLUTION = 120

# The minimum time limit for an instance to be allowed into the training set
TRAINING_SET_MIN_TIME_LIMIT = 5

# The max allowed presolve time limit for all runs when selecting a training set
TRAINING_SET_PRESOLVE_TIME_LIMIT = 10

# The max and min nodes for an instance to be allowed in the training set
TRAINING_SET_MIN_NODES = 50
TRAINING_SET_MAX_NODES = 20000

# The maximum multiple of TRAINING_SET_TIME_LIMIT that can be hit during SMAC runs
MAX_MULTIPLE_DEFAULT_RUN_TIME = 2

# Directory information for SMAC
SMAC_DEFAULT_RESULTS_DIR = "results/Default-SMAC/"
SMAC_DEFAULT_OUTFILES_DIR = "outfiles/Default-SMAC/"

# The cut selector options. We have default and parallel
CUT_SELECTOR_OPTIONS = ["default", "ensemble"]
