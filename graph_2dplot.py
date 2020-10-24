import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='***')
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

plt.scatter(x, y, c=colors)
plt.ylim(-21000, 5)
plt.savefig('scatter.png')



