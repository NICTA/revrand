#! /usr/bin/env python3
""" A La Carte Extended/Unscented GP demo.

    Author:     Daniel Steinberg
    Date:       18 May 2015
    Institute:  NICTA
"""

import numpy as np
import matplotlib.pyplot as pl
#from yavanna.supervised import bases
from scipy.spatial.distance import cdist

from linearizedGP import extendedGP
from linearizedGP import kernels


#
# Settings
#

N = 3e2
Ns = 1e3
lenscale_true = 0.5  # For the gpdraw dataset
noise = 0.1

nlfunc = lambda f: f**3
dnlfunc = lambda f: 3 * f**2


# GP settings
kfunc = kernels.kern_se
lenscale = 0.3  # For all basis functions that take lengthscales
nbases = 200


#
# Generate Data
#

X = np.linspace(-2*np.pi, 2*np.pi, N)
Xtrain = X[:, np.newaxis]
K = np.exp(-cdist(Xtrain, Xtrain, metric='sqeuclidean')
           / (2*lenscale_true**2))
U, S, V = np.linalg.svd(K)
L = U.dot(np.diag(np.sqrt(S))).dot(V)
ftrain = np.random.randn(N).dot(L)
ytrain = nlfunc(ftrain) + np.random.randn(N) * noise
Xtest = np.linspace(-4*np.pi, 4*np.pi, Ns)[:, np.newaxis]


#
#  Single task GPs Learning
#

gp = extendedGP.extendedGP(nlfunc, dnlfunc, kfunc)
gp.learnLB((1e-2, 1e-2), ynoise=1e-2)
lml = gp.learn(Xtrain.T, ytrain, (1, lenscale), ynoise=1, verbose=True)


#
#  Single task GPs Prediction
#

Eys, _, Ems, Vms = gp.predict(Xtest.T)
Sms = np.sqrt(Vms) * 2


#
#  Plot
#

Xs = Xtest.flatten()
pl.plot(X, ftrain, 'r', X, ytrain, 'k.')
pl.plot(Xs, Eys, 'g', Xs, Ems, 'b')
pl.fill_between(Xs, Ems+Sms, Ems-Sms, facecolor='blue', edgecolor='blue',
                alpha=0.3, label=None)
pl.legend(['True F', 'Training Y', 'E[Y]', 'E[F]'])
pl.show()
