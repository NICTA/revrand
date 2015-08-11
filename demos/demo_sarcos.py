#! /usr/bin/env python3
""" A La Carte GP Application to SARCOS dataset. """

import logging
import numpy as np
from yavanna.supervised import regression, bases
from yavanna.validate import smse, msll
from yavanna.unsupervised.transforms import whiten, whiten_apply
import computers.gp as gp


#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

lenscale = 10
sigma = 100
noise = 1
nbases = 400
gp_Ntrain = 1024


#
# Load data
#

# TODO: Put it into mithlond when ready
sarcos_train = np.load('sarcos_train.npy')
sarcos_test = np.load('sarcos_test.npy')

X_train = sarcos_train[:, 0:21]
X_test = sarcos_test[:, 0:21]
y_train = sarcos_train[:, 21]
y_test = sarcos_test[:, 21]

Ntrain, D = X_train.shape


#
# Whitening (as opposed to ARD kernel)
#

X_train, U, l, Xmean = whiten(X_train)
X_test = whiten_apply(X_test, U, l, Xmean)


#
# Train GP
#

kdef = lambda h, k: h(1e-5, 1e5, sigma) * k('gaussian', h(1e-5, 1e5, lenscale))
kfunc = gp.compose(kdef)

# Set up optimisation
learning_params = gp.LearningParams()
learning_params.sigma = gp.auto_range(kdef)
learning_params.noise = gp.Range([1e-5], [1e5], [noise])
learning_params.walltime = 60

# Get random subset of data for training
train_ind = np.random.choice(range(Ntrain), size=gp_Ntrain, replace=False)
X_train_sub = X_train[train_ind, :]
y_train_sub = y_train[train_ind]

# Learn hyperparameters
hyper_params = gp.learn(X_train_sub, y_train_sub, kfunc, learning_params)
regressor = gp.condition(X_train_sub, y_train_sub, kfunc, hyper_params)


#
# Train A la Carte
#

base = bases.RandomRBF_ARD(nbases, D) + bases.RandomRBF(nbases, D)
lenARD = lenscale * np.ones(D + 1)
params = regression.alacarte_learn(X_train, y_train, base, lenARD,
                                   var=noise**2, usegradients=False)

# base = bases.RandomRBF(nbases, D)
# params = regression.alacarte_learn(X_train, y_train, base, (lenscale,),
#                                    var=noise**2)


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

Ey, Vf, Vy = regression.alacarte_predict(X_test, X_train, y_train, base,
                                         *params)
Sy = np.sqrt(Vy)


#
# Validation
#

log.info("Subset GP smse = {}, msll = {},\n\thypers = {}, noise = {}."
         .format(smse(y_test, Ey_gp), msll(y_test, Ey_gp, Vy_gp, y_train),
                 hyper_params[0], hyper_params[1]))
log.info("A la Carte smse = {}, msll = {},\n\thypers = {}, noise = {}."
         .format(smse(y_test, Ey), msll(y_test, Ey, Vy, y_train),
                 params[0], np.sqrt(params[1])))
