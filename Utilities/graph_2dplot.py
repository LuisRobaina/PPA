"""
Generate a 2D plot of the position of the ownship and intruder.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', action="store", dest='own', default=False)
parser.add_argument('-i', action='store', dest='int', default=None)

args = parser.parse_args()
colors = ['b', 'r']

data = np.loadtxt(args.own, delimiter=',', unpack=False)
x, y = data.T
colors = ['b' for x in range(x.size)]

if args.int is not None:
    data = np.loadtxt(args.int, delimiter=',', unpack=False)
    x_i, y_i = data.T
    colors = ['b' for x in range(x.size)]
    colors = colors + ['r' for x in range(x_i.size)]
    x = np.concatenate((x, x_i), axis=None)
    y = np.concatenate((y, y_i), axis=None)

plt.xticks(np.arange(min(x), max(x)+1, 500.0))

plt.ylim(-21000, 5)
plt.scatter(x, y, c=colors)

# Output png file.
plt.savefig('2Dscatter.png')

