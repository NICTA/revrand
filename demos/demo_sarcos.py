#! /usr/bin/env python3
""" A La Carte GP Application to SARCOS dataset. """

import os
import wget
import logging
import numpy as np
import computers.gp as gp
from scipy.io import loadmat
from pyalacarte import regression, bases
from pyalacarte.validation import smse, msll


#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

lenscale = 10
sigma = 100
noise = 1
regulariser = 1
nbases = 400
gp_Ntrain = 1024
maxit = 1e5
rate = 0.95
eta = 1e-6
batchsize = 100

useSGD = True
diagcov = True

# noise = 3.089897655618356
# regulariser = 18191.35201528529


#
# Load data
#

# Pull this down and process if not present
if not os.path.exists('sarcos_test.npy') or \
        not os.path.exists('sarcos_train.npy'):

    wget.download('http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat')
    wget.download('http://www.gaussianprocess.org/gpml/data/sarcos_inv_test'
                  '.mat')
    sarcostrain = loadmat('sarcos_inv.mat')['sarcos_inv']
    sarcostest = loadmat('sarcos_inv_test.mat')['sarcos_inv_test']
    np.save('sarcos_train.npy', sarcostrain)
    np.save('sarcos_test.npy', sarcostest)
    del sarcostest, sarcostrain

sarcos_train = np.load('sarcos_train.npy')
sarcos_test = np.load('sarcos_test.npy')

X_train = sarcos_train[:, 0:21]
X_test = sarcos_test[:, 0:21]
y_train = sarcos_train[:, 21]
y_test = sarcos_test[:, 21]

Ntrain, D = X_train.shape


#
# Whitening
#

# X_train, U, l, Xmean = whiten(X_train)
# X_test = whiten_apply(X_test, U, l, Xmean)


# Get random subset of data for training the GP
train_ind = np.random.choice(range(Ntrain), size=gp_Ntrain, replace=False)
X_train_sub = X_train[train_ind, :]
y_train_sub = y_train[train_ind]

#
# Train A la Carte
#

base = bases.RandomRBF_ARD(nbases, D)
lenARD = lenscale * np.ones(D)

# lenARD = np.array([1.15895334,
#                    4.90825038,
#                    12.39142359,
#                    1.00389066,
#                    9.81194189,
#                    11.43710948,
#                    7.58182018,
#                    9.97105491,
#                    15.21816297,
#                    16.94271258,
#                    1.59809687,
#                    14.44737729,
#                    12.52705036,
#                    7.92625432,
#                    33.34440466,
#                    55.46652518,
#                    52.02653434,
#                    56.79806777,
#                    55.84145868,
#                    27.73558116,
#                    40.22196858])

if useSGD:
    log.info("Using SGD regressor")
    params = regression.bayesreg_sgd(X_train, y_train, base, [lenARD],
                                     rate=rate, var=noise**2,
                                     regulariser=regulariser, maxit=maxit,
                                     batchsize=batchsize, eta=eta)
else:
    log.info("Using full variational regressor")
    params = regression.bayesreg_elbo(X_train, y_train, base, [lenARD],
                                      var=noise**2, diagcov=diagcov,
                                      regulariser=regulariser, maxit=maxit)

#
# Train GP
#

kdef = lambda h, k: h(1e-5, 1e5, sigma) \
    * k('gaussian', h(np.ones(D) * 1e-5, np.ones(D) * 1e5, lenARD))
kfunc = gp.compose(kdef)

# Set up optimisation
learning_params = gp.OptConfig()
learning_params.sigma = gp.auto_range(kdef)
learning_params.noise = gp.Range([1e-5], [1e5], [noise])
learning_params.walltime = 300


# Learn hyperparameters
hyper_params = gp.learn(X_train_sub, y_train_sub, kfunc, learning_params)
regressor = gp.condition(X_train_sub, y_train_sub, kfunc, hyper_params)


#
# Predict GP
#

query = gp.query(X_test, regressor)
Ey_gp = gp.mean(regressor, query)
Vf_gp = gp.variance(regressor, query) + np.array(hyper_params[1])**2
Vy_gp = Vf_gp + np.array(hyper_params[1])**2
Sy_gp = np.sqrt(Vy_gp)


#
# Predict A la Carte
#

Ey, Vf, Vy = regression.bayesreg_predict(X_test, base, *params)
Sy = np.sqrt(Vy)


#
# Validation
#

log.info("Subset GP smse = {}, msll = {},\n\thypers = {}, noise = {}."
         .format(smse(y_test, Ey_gp), msll(y_test, Ey_gp, Vy_gp, y_train),
                 hyper_params[0], hyper_params[1]))
log.info("A la Carte smse = {}, msll = {},\n\thypers = {}, noise = {}."
         .format(smse(y_test, Ey), msll(y_test, Ey, Vy, y_train),
                 params[2], np.sqrt(params[3])))
