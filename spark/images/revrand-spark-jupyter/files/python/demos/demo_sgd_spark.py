#! /usr/bin/env python3
""" A Small demo to try out SGD using Spark on a Maximum likelihood regression problem. """

import numpy as np
import matplotlib.pyplot as pl
from revrand.optimize import (minimize, sgd_u, sgd_u_spark)
from revrand.optimize.sgd_updater import (AdaGrad, AdaDelta, Momentum)
from revrand.basis_functions import RadialBasis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

hasSparkContext = False
try:
    from pyspark import SparkConf, SparkContext
    conf = (SparkConf().setAppName("Spark SGD Demo"))
    sc = SparkContext(conf = conf)
    sc.addPyFile(__file__)
    hasSparkContext = True
except ImportError:
    pass


# Objective function
def f(w, Data, sigma=1.0):
    y, Phi = Data[:, 0], Data[:, 1:]
    logp = -1.0 / (2 * sigma * sigma) * ((y - Phi.dot(w))**2).sum()
    d = 1.0 / (sigma * sigma) * Phi.T.dot((y - Phi.dot(w)))
    return -logp, -d


def sgd_demo():
    # Settings

    batchsize = 100
    var = 0.05
    nPoints = 1000
    nQueries = 500
    passes = 200
    min_grad_norm = 0.01
    rate = 0.95
    eta = 1e-6

    # Create dataset
    X = np.linspace(0.0, 1.0, nPoints)[:, np.newaxis]
    Y = np.sin(2 * np.pi * X.flatten()) + np.random.randn(nPoints) * var
    centres = np.linspace(0.0, 1.0, 20)[:, np.newaxis]
    Phi = RadialBasis(centres)(X, 0.1)
    train_dat = np.hstack((Y[:, np.newaxis], Phi))

    Xs = np.linspace(0.0, 1.0, nQueries)[:, np.newaxis]
    Yt = np.sin(2 * np.pi * Xs.flatten())
    Phi_s = RadialBasis(centres)(Xs, 0.1)
    w = np.linalg.solve(Phi.T.dot(Phi), Phi.T.dot(Y))
    Ys = Phi_s.dot(w)

    # L-BFGS approach to test objective
    w0 = np.random.randn(Phi.shape[1])
    results = minimize(f, w0, args=(train_dat,), jac=True, method='L-BFGS-B')
    w_grad = results['x']
    Ys_grad = Phi_s.dot(w_grad)

    # SGD for learning w (AdaDelta)
    w0 = np.random.randn(Phi.shape[1])
    results = sgd_u(f, w0, train_dat, updater=AdaDelta(), passes=passes,
                  batchsize=batchsize, eval_obj=True, gtol=min_grad_norm)
    w_sgd_ad, gnorms_ad, costs_ad = results['x'], results['norms'], results['objs']
    Ys_sgd_ad = Phi_s.dot(w_sgd_ad)

    # SGD for learning w (AdaGrad)
    w0 = np.random.randn(Phi.shape[1])
    results = sgd_u(f, w0, train_dat, updater=AdaGrad(np.shape(w)), passes=passes,
                  batchsize=batchsize, eval_obj=True, gtol=min_grad_norm)
    w_sgd_ag, gnorms_ag, costs_ag = results['x'], results['norms'], results['objs']
    Ys_sgd_ag = Phi_s.dot(w_sgd_ag)

    # SGD for learning w (Momentum)
    w0 = np.random.randn(Phi.shape[1])
    results = sgd_u(f, w0, train_dat, updater=Momentum(np.shape(w)), passes=passes,
                  batchsize=batchsize, eval_obj=True, gtol=min_grad_norm)
    w_sgd_m, gnorms_m, costs_m = results['x'], results['norms'], results['objs']
    Ys_sgd_m = Phi_s.dot(w_sgd_m)

    if hasSparkContext:
        # Distributed SGD for learning w (AdaDelta)
        w0 = np.random.randn(Phi.shape[1])
        rdd = sc.parallelize(train_dat)
        results = sgd_u_spark(f, w0, rdd, updater=AdaDelta(), passes=passes,
                            batchsize=batchsize, eval_obj=True, gtol=min_grad_norm)
        w_sgd_dad, gnorms_dad, costs_dad = results['x'], results['norms'], results['objs']
        Ys_sgd_dad = Phi_s.dot(w_sgd_dad)

        # Distributed SGD for learning w (AdaGrad)
        w0 = np.random.randn(Phi.shape[1])
        rdd = sc.parallelize(train_dat)
        results = sgd_u_spark(f, w0, rdd, updater=AdaGrad(np.shape(w)), passes=passes,
                            batchsize=batchsize, eval_obj=True, gtol=min_grad_norm)
        w_sgd_dag, gnorms_dag, costs_dag = results['x'], results['norms'], results['objs']
        Ys_sgd_dag = Phi_s.dot(w_sgd_dag)

    # Print results
    def print_res(method, cost):
        print("{0: >20}: {1}".format(method,cost))

    print_res("AdaDelta", costs_ad[-1])
    print_res("AdaGrad", costs_ag[-1])
    print_res("Momentum", costs_m[-1])
    if hasSparkContext:
        print_res("Dist. AdaDelta", costs_dad[-1])
        print_res("Dist. AdaGrad", costs_dag[-1])

    # Visualise results
    fig = pl.figure()
    ax = fig.add_subplot(121)
    # truth
    pl.plot(X, Y, 'k.', Xs, Yt, 'k-', markersize=1)
    # exact weights
    pl.plot(Xs, Ys, 'c-')
    pl.plot(Xs, Ys_grad, 'b-')
    pl.plot(Xs, Ys_sgd_ad, 'g-')
    pl.plot(Xs, Ys_sgd_ag, 'r-')
    pl.plot(Xs, Ys_sgd_m, 'y-')
    labels = ['Training', 'Truth', 'Analytic', 'LBFGS', 'AdaDelta', 'AdaGrad', 'Momentum']
    if hasSparkContext:
        pl.plot(Xs, Ys_sgd_dad, 'g--')
        pl.plot(Xs, Ys_sgd_dag, 'r--')
        labels += [ 'Dist. AdaDelta', 'Dist. AdaGrad']

    pl.title('Function fitting')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.legend(labels)

    ax = fig.add_subplot(222)
    pl.xlabel('iteration')
    pl.title('SGD convergence')
    ax.plot(range(len(costs_ad)), costs_ad, 'g')
    ax.plot(range(len(costs_ag)), costs_ag, 'r')
    labels = ['AdaDelta', 'AdaGrad']
    if hasSparkContext:
        ax.plot(range(len(costs_dad)), costs_dad, 'g--')
        ax.plot(range(len(costs_dag)), costs_dag, 'r--')
        labels += [ 'Dist. AdaDelta', 'Dist. AdaGrad']
    ax.set_ylabel('cost')
    pl.legend(labels)

    ax = fig.add_subplot(224)
    pl.xlabel('iteration')
    pl.title('SGD convergence')
    ax.plot(range(len(gnorms_ad)), gnorms_ad, 'g')
    ax.plot(range(len(gnorms_ag)), gnorms_ag, 'r')
    labels = ['AdaDelta', 'AdaGrad']
    if hasSparkContext:
        ax.plot(range(len(gnorms_dad)), gnorms_dad, 'g--')
        ax.plot(range(len(gnorms_dag)), gnorms_dag, 'r--')
        labels += [ 'Dist. AdaDelta', 'Dist. AdaGrad']
    ax.set_ylabel('gradient norms')
    pl.legend(labels)

    pl.show()


if __name__ == "__main__":
    sgd_demo()
