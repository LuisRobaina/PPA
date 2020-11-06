import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kde
import numpy as np
import argparse


matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-n', action="store", dest='file_name', default=False)

args = parser.parse_args()

data = np.loadtxt(args.file_name, delimiter=',', unpack=False)
x, y = data.T

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
nbins = 300
k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
plt.colorbar()
plt.savefig('Density.png')

