#! /usr/bin/env python
""" A La Carte GP Application to SARCOS dataset. """

import logging
import numpy as np
import revrand.legacygp as gp
import revrand.legacygp.kernels as kern

from revrand import slm, glm
from revrand.basis_functions import RandomRBF
from revrand.likelihoods import Gaussian
from revrand.btypes import Parameter, Positive
from revrand.metrics import smse, msll
from revrand.utils.datasets import fetch_gpml_sarcos_data

#
# Settings
#

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

lenscale = 10
noise = 1
nbases = 200
gp_Ntrain = 1000
passes = 5
rho = 0.9
epsilon = 1e-6
batchsize = 10
useSGD = True


#
# Load data
#

gpml_sarcos = fetch_gpml_sarcos_data()

X_train = gpml_sarcos.train.data
y_train = gpml_sarcos.train.targets

X_test = gpml_sarcos.test.data
y_test = gpml_sarcos.test.targets

Ntrain, D = X_train.shape


# Get random subset of data for training the GP
train_ind = np.random.choice(range(Ntrain), size=gp_Ntrain, replace=False)
X_train_sub = X_train[train_ind, :]
y_train_sub = y_train[train_ind]


#
# Train A la Carte
#
lenARD = lenscale * np.ones(D)
base = RandomRBF(nbases, D, lenscale_init=Parameter(lenARD, Positive()))

if useSGD:
    log.info("Using SGD regressor")
    llhood = Gaussian()
    lparams = [noise**2]
    params = glm.learn(X_train, y_train, llhood, base, use_sgd=True, rho=rho,
                       epsilon=epsilon, batchsize=batchsize, maxit=passes)
else:
    log.info("Using full variational regressor")
    params = slm.learn(X_train, y_train, base,
                       var=Parameter(noise**2, Positive()))


#
# Train GP
#


def kdef(h, k):
    return (h(1e-5, 1., 0.5)
            * k(kern.gaussian, [h(1e-5, 1e5, l) for l in lenARD])
            + k(kern.lognoise, h(-4, 1, -3)))

hyper_params = gp.learn(X_train_sub, y_train_sub, kdef, verbose=True,
                        ftol=1e-15, maxiter=1000)


#
# Predict Revrand
#

Ey, Vf, _, _ = glm.predict_moments(X_test, llhood, base, *params)
Vy = Vf + params[2][0]
Sy = np.sqrt(Vy)


#
# Predict GP
#

regressor = gp.condition(X_train_sub, y_train_sub, kdef, hyper_params)
query = gp.query(regressor, X_test)
Ey_gp = gp.mean(query)
Vf_gp = gp.variance(query)
Vy_gp = gp.variance(query, noise=True)
Sy_gp = np.sqrt(Vy_gp)


#
# Validation
#

log.info("Subset GP smse = {}, msll = {},\n\thypers = {}, noise = {}."
         .format(smse(y_test, Ey_gp), msll(y_test, Ey_gp, Vy_gp, y_train),
                 hyper_params[0], hyper_params[1]))
log.info("Revrand smse = {}, msll = {},\n\thypers = {}, noise = {}."
         .format(smse(y_test, Ey), msll(y_test, Ey, Vy, y_train),
                 params[2], np.sqrt(params[3])))
