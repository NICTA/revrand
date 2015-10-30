#! /usr/bin/env python3
""" GLM Demo """

import matplotlib.pyplot as pl
# import computers.gp as gp
import numpy as np
import logging

from revrand import basis_functions, glm, likelihoods
from revrand.validation import mll, smse
from revrand.utils.datasets import gen_gausprocess_se

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
#
# Settings
#

# Algorithmic properties
nbases = 100
lenscale = 0.7  # For all basis functions that take lengthscales
noise = 0.2
# rate = 0.9
# eta = 1e-6
# passes = 1000
# batchsize = 100
reg = 1

N = 500
Ns = 250

# Dataset properties
lenscale_true = 0.7  # For the gpdraw dataset
noise_true = 0.1

#
# Make Data
#

Xtrain, ytrain, Xtest, ftest = \
    gen_gausprocess_se(N, Ns, lenscale=lenscale_true, noise=noise_true)

#
# Make Bases and Likelihood
#

llhood = likelihoods.Gaussian()
basis = basis_functions.RandomRBF(nbases, Xtrain.shape[1])


#
# Inference
#

params = glm.glm_learn(ytrain, Xtrain, llhood, [noise**2], basis, [lenscale])
Ey_s = glm.glm_predict(Xtest, llhood, basis, *params)


#
# Plot
#

Xpl_t = Xtrain.flatten()
Xpl_s = Xtest.flatten()

# Regressor
pl.plot(Xpl_s, Ey_s, 'r-', label='SGD Bayes linear reg.')
# pl.fill_between(Xpl_s, Ey_s - 2 * Sy_s, Ey_s + 2 * Sy_s, facecolor='none',
#                 edgecolor='r', linestyle='--', label=None)

# Training/Truth
pl.plot(Xpl_t, ytrain, 'k.', label='Training')
pl.plot(Xpl_s, ftest, 'k-', label='Truth')

# pl.legend()

pl.grid(True)
pl.title('Regression demo')
pl.ylabel('y')
pl.xlabel('x')

m, C = params[0:2]
pl.figure()
K = m.shape[1]
cols = pl.cm.jet(np.linspace(0, 1, K))
for mk, Ck, c in zip(m.T, C.T, cols):
    pl.plot(range(len(mk)), mk, color=c)
    pl.fill_between(range(len(mk)), mk - 2 * np.sqrt(Ck), mk + 2 * np.sqrt(Ck),
                    alpha=0.1, edgecolor='none', facecolor=c, label=None)

pl.grid(True)
pl.title('Weight Posterior')
pl.ylabel('w')
pl.xlabel('basis index')

pl.show()
