import wget
import csv
import time
import os
import requests
import ssl
import argparse
from utilities import is_file, is_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('collection_set', type=is_file)
    parser.add_argument('instance_dir', type=is_dir)
    args = parser.parse_args()

    collection_set = args.collection_set
    instance_dir = args.instance_dir

    # Workaround for error with SSL certificate verification (Would not use for random websites)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Read the csv containing all instance names and tags
    instance_descriptions = []
    with open(collection_set, 'r') as s:
        reader = csv.reader(s, delimiter=',')
        for row in reader:
            instance_descriptions.append(row)

    # Get instances that do not contain tags: feasibility, numerics, infeasible, no_solution
    valid_rows = []
    num_instances = len(instance_descriptions)
    num_feasibility_instances = 0
    num_numerics_instances = 0
    num_infeasible_instances = 0
    num_no_solution_instances = 0
    num_unbounded_instances = 0
    for row_i, row in enumerate(instance_descriptions):
        if row_i == 0:
            continue
        if 'feasibility' in row[-1]:
            num_feasibility_instances += 1
            continue
        if 'numerics' in row[-1]:
            num_numerics_instances += 1
            continue
        if 'infeasible' in row[-1]:
            num_infeasible_instances += 1
            continue
        if 'no_solution' in row[-1]:
            num_no_solution_instances += 1
            continue
        if 'Unbounded' in row[-2]:
            num_unbounded_instances += 1
            continue
        valid_rows.append(row_i)

    instances = []
    for row_i in valid_rows:
        instances.append(instance_descriptions[row_i][0])

    # Download the instances
    for instance in instances:
        mps_url = 'https://miplib.zib.de/WebData/instances/{}.mps.gz'.format(instance)
        mps_file = '{}/{}.mps.gz'.format(instance_dir, instance)
        wget.download(mps_url, mps_file)
        time.sleep(0.1)

    print('num instances: {}'.format(num_instances))
    print('num filtered instances with feasibility flag: {}'.format(num_feasibility_instances))
    print('num filtered instances with numerics flag: {}'.format(num_numerics_instances))
    print('num filtered instances with infeasible flag: {}'.format(num_infeasible_instances))
    print('num filtered instances with no_solution flag: {}'.format(num_no_solution_instances))
    print('num unbounded instances with no valid solution: {}'.format(num_unbounded_instances))

