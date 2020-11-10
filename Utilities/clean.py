"""
Remove the Traj.csv files for each encounter on the given directory
"""
import argparse
import os

parser = argparse.ArgumentParser(description='Remove the Trajectory.csv files for each encounter on the given directory')
parser.add_argument('-p', action="store", dest='path', default=False)
parser.add_argument('-c', action="store", dest='count', default=0)

args = parser.parse_args()

for i in range(int(args.count)):
    path = f'''{args.path}/ENCOUNTER_{i}/Trajectory.csv'''
    if os.path.exists(path):
        print("Cleaned: ", path)
        os.remove(path)
    else:
        print("Path Not Found: ", path)
