#! /usr/bin/env python
import os
import argparse
import yaml
from pyscipopt import Model, SCIP_PARAMSETTING
from utilities import is_dir, remove_temp_files
from BranchRules.RootNodeFeatureExtractorBranchRule import RootNodeFeatureExtractor


def presolve_and_get_embeddings(instance_dir, feature_dir):
    """
    Main function for extracting the feature representation of each instance
    Args:
        instance_dir (dir): Directory where all instance files are stored
        feature_dir (dir): Directory where all feature files will be dumped

    Returns:
        Creates feature files for all instances
    """

    # Get the instance files
    instance_files = sorted(os.listdir(instance_dir))
    instances = [instance_file.split('.mps')[0] for instance_file in instance_files]

    # Iterate over the instances and extract some feature representation and then save the feature file
    for i in range(len(instances)):
        scip = Model()
        scip.readProblem(os.path.join(instance_dir, instance_files[i]))
        feature_extractor = RootNodeFeatureExtractor(scip)
        scip.includeBranchrule(feature_extractor, "feature_extractor", "extract features of root LP relaxation",
                               priority=10000000, maxdepth=-1, maxbounddist=1)
        scip.setSeparating(SCIP_PARAMSETTING.OFF)
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)
        scip.optimize()
        feature_dict = feature_extractor.features
        scip.freeProb()

        with open(os.path.join(feature_dir, '{}.yml'.format(instances[i])), 'w') as s:
            yaml.dump(feature_dict, s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('feature_dir', type=is_dir)
    args = parser.parse_args()

    remove_temp_files(args.feature_dir)

    presolve_and_get_embeddings(args.instance_dir, args.feature_dir)
