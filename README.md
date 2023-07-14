# A Context-Aware Cutting Plane Selection Algorithm for Mixed-Integer Programming

If this software was used for academic purposes, please cite our paper with the below information:

`add something here`

## Install Guide
Requirements: Python 3.9 / Debian 11 (Also tested with 3.11).
We use SLURM https://slurm.schedmd.com/overview.html as a job manager. 
All calls go through a central function however, and in theory SLURM could be replaced by python's 
default multiprocessing package.

Run the bash script init_venv. If you don't use bash, configure the shebang (first line of the script) 
to be your shell interpreter.

`./init_venv`

After installing the virtual environment, make sure to always activate it with the script printed beneath. 
This is so your python path is appended and files at different directory levels can import from each other.

`source ./set_venv`

Now go and install SCIP from (https://www.scipopt.org/index.php#download / https://github.com/scipopt/scip)
For Ubuntu / Debian, the .sh installer is the easiest choice if you don't want to configure it yourself). 
The latest release version of SCIP is required for this project, or the development branch `mt/ensemble_cutsel`.

You can test if SCIP is installed by locating /bin/scip and calling it through the command line. 
SCIP should hopefully open.

One then needs to install PySCIPOpt https://github.com/scipopt/PySCIPOpt. 
I would recommend following the INSTALL.md guide. Make sure to have set your environment variable pointing to SCIP! 
You can test if this has been properly installed by running one of the tests, or by trying to import Model. 
This research was done using PySCIPOpt 4.3.0. 

## How to run the software
We use Nohup https://en.wikipedia.org/wiki/Nohup to run all of our jobs and to capture output 
of the main function calls. It also allows jobs to be started through a SHH connection. 
Using this is not necessary however, so feel free to use whatever software you prefer. 
An example call to redirect output to nohup/nohup.out and to run the process in the background would be

`nohup python dir/example.py > nohup/nohup.out &`

For our experiments we used [MIPLIB 2017](https://miplib.zib.de/), 
[StrIPLib](https://striplib.or.rwth-aachen.de/login/), and [SNDlib-MIPS](https://zenodo.org/record/8021237).
A combination of these instances were used for training, where we evaluated our final model on the MIPLIB 2017 
benchmark set.

To select instances in a similar to us, please download all the appropriate instances from the respective websites 
(or feel free to use instances that you have on hand). Place all instances in a single directory.
We reduce the amount of instances for training by running:

`python Slurm/generate_instance_subset.py instance_dir solution_dir instance_subset_dir results_dir outfiles_dir`

After the instances have been pre-filtered, we can now get their feature embeddings to perform a maximally diverse 
selection. To do this run

`python scripts/get_instance_embeddings.py instance_dir feature_dir`

Then to actually select a maximally diverse subset run

`python scripts/get_diverse_instance_subset.py instance_dir selected_instance_dir feature_dir plot_stuff`

here `plot_stuff` is an argument about whether you'd like the end result visualised. You can feel free to manually 
change between PCA or 2-SNE as the embedding options for visualisation. (You have to manually gzip all .sol files at 
this point)

## Running the Software

Now we can train a SMAC model. To do this run

`nohup python Slurm/main.py bool_1 seed instance_dir solution_dir results_dir outfiles_dir > nohup/test.out &`

`bool_1` here is whether ot not you want to first generate default runs to compare against (This is an option in 
case you already have generated your default runs and don't want to constantly reproduce identical results).

This main function trains a SMAC model, with output being redirected to `nohup/test.out`. In that file you will find 
the training progression along with the best parameter choice found at the end of training as well as how it 
performed. For more information, one can view the SMAC run history file that was also created. 

To test a given configuration on some test set, you can run the following:

`nohup python Slurm/test_configuration.py config_file bool instance_dir solution_dir results_dir outfiles_dir > 
nohup/test.out &`

#### Thanks for Reading!! I hope this code helps you with your research. Please feel free to send any issues.
