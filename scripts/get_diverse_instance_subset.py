import shutil
import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pyscipopt import Model

from utilities import is_dir, str_to_bool, remove_temp_files


def perform_pca_select_subset_and_plot(instance_dir, selected_instance_dir, feature_dir, plot_selection):
    """
    Main function for performing feature transformation, selecting a maximally diverse instance subset,
    and then plotting the result

    Args:
        instance_dir (dir): Directory containing all instances
        selected_instance_dir (dir): Directory where all selected instances will be moved to
        feature_dir (dir): Directory containing all feature information for each instance
        plot_selection (bool): Whether we want to actually plot the selected subset in the projected space.

    Returns:
        Nothing.
    """

    # Remove all previously selected instances
    remove_temp_files(selected_instance_dir)

    # Perform the feature transform (and load all the data)
    data, t_data, instances = perform_feature_transform(feature_dir)

    # Create a MILP and solve it (that chooses a diverse selection)
    selected_indices = select_subset(data)

    # Move all the selected files
    for i in selected_indices:
        shutil.copy(os.path.join(instance_dir, instances[i] + ".mps.gz"),
                    os.path.join(selected_instance_dir, instances[i] + ".mps.gz"))

    # Plot the selection in 2D if requested
    if plot_selection:
        plot_projection(t_data, selected_indices)

    print('Selected Instances: {}'.format([instances[i] for i in selected_indices]))


def select_subset(data):
    """
    Create a SCIP model for selecting instance subsets
    Args:
        data (np.ndarray): The list of feature vectors

    Returns:
        The selected indices
    """

    # Initialise the SCIP model
    scip = Model()

    # Now build the SCIP model variables and constraints
    x = {}
    y = {}
    distances = {}
    max_distance = 0
    for i in range(len(data)):
        x[i] = scip.addVar(vtype='B', name='instance_{}'.format(i))
        y[i] = {}
        distances[i] = {}
        for j in range(i + 1, len(data)):
            # Calculate the distance
            distances[i][j] = 0
            for dim in range(len(data[i])):
                distances[i][j] += abs(data[i][dim] - data[j][dim])**2
            distances[i][j] = float(np.sqrt(distances[i][j]))
            max_distance = max(distances[i][j], max_distance)
            y[i][j] = scip.addVar(vtype='C', name='connection_{},{}'.format(i, j))
    maxy = scip.addVar(vtype='C', name='max_y')
    # Add constraints
    scip.addCons(sum(x[i] for i in range(len(data))) == 40, name='select_x')
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            scip.addCons(y[i][j] <= distances[i][j] + (max_distance * (2 - x[i] - x[j])))
            scip.addCons(maxy <= y[i][j])
    scip.setObjective(maxy, sense='maximize')
    scip.optimize()

    # Load the selected indices from the optimal solution
    selected_indices = []
    for i in range(len(data)):
        if scip.getVal(x[i]) > 0.5:
            selected_indices.append(i)

    return selected_indices


def perform_feature_transform(feature_dir, pca=False):
    """
    Perform feature transform into a 2-dimensional space. Either do this through PCA or through 2-SNE
    Args:
        feature_dir (dir): Directory where all feature data is stored
        pca (bool): Whether PCA or 2-SNE should be performed

    Returns:
        The transformed 2-dimensional feature vectors
    """

    # Get the feature files and instances
    feature_files = os.listdir(feature_dir)
    instances = [feature_file.split('.yml')[0] for feature_file in feature_files]

    # Get the feature data
    data = []
    for feature_file in feature_files:
        data.append([])
        feature_file = os.path.join(feature_dir, feature_file)
        with open(feature_file, 'r') as s:
            instance_data = yaml.safe_load(s)
        for data_key in sorted(instance_data.keys()):
            data[-1].append(instance_data[data_key])

    # Transform the feature data to 2-dimensional embeddings
    data = np.array(data)
    if pca:
        pca = PCA(n_components=2)
        t_data = pca.fit_transform(data)
    else:
        sne = TSNE(n_components=2)
        t_data = sne.fit_transform(data)

    return data, t_data, instances


def plot_projection(t_data, selected_indices=None):
    """
    Function for plotting the instance distribution in the projected space
    Args:
        t_data (list): The transformed data
        selected_indices (list): The list of indices that were actually selected

    Returns:
        A plot
    """

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the test set as black dots. Their transparency is the relative performance of their prediction.
    for i in range(len(t_data)):
        if selected_indices is None or i in selected_indices:
            ax.scatter(t_data[:, 0][i], t_data[:, 1][i], c='black')
        else:
            ax.scatter(t_data[:, 0][i], t_data[:, 1][i], c='black', alpha=0.2)

    # Reveal your secrets!
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('selected_instance_dir', type=is_dir)
    parser.add_argument('feature_dir', type=is_dir)
    parser.add_argument('plot_selection', type=str_to_bool)
    args = parser.parse_args()

    perform_pca_select_subset_and_plot(args.instance_dir, args.selected_instance_dir,
                                       args.feature_dir, args.plot_selection)
