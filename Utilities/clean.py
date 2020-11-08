"""
Remove the Traj.csv files for each encounter on the given directory
"""

import argparse
import os

parser = argparse.ArgumentParser(description='Remove the Traj.csv files for each encounter on the given directory')
parser.add_argument('-p', action="store", dest='path', default=False)
parser.add_argument('-c', action="store", dest='count', default=0)

args = parser.parse_args()

for i in range(int(args.count)):
    path = f'''{args.path}/ENCOUNTER_{i}/Traj.csv'''
    if os.path.exists(path):
        os.remove(path)
