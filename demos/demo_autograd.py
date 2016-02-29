#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import matplotlib.pyplot as plt
import autograd.numpy as np

from revrand.utils.datasets import make_regression
from revrand.basis_functions import make_radial_basis
from revrand.regression import learn, predict
from revrand.optimize import Positive


@click.command()
@click.option('--n-samples', default=500, type=int)
@click.option('--n-tests', default=250, type=int)
@click.option('--noise', default=.5, type=float)
@click.option('--lenscale', default=1., type=float)
@click.option('--reg', default=1., type=float)
@click.option('--use-autograd', is_flag=True)
def main(n_samples, n_tests, noise, lenscale, reg, use_autograd):

    X_train = np.expand_dims(np.linspace(-2. * np.pi, 2. * np.pi, n_samples),
                             axis=1)
    X_train, y_train = make_regression(np.sin, X_train, n_samples=n_samples,
                                       noise=noise, random_state=3)

    X_test = np.expand_dims(np.linspace(-2. * np.pi, 2. * np.pi, n_tests),
                            axis=1)
    X_test, f_test = make_regression(np.sin, X_test, n_samples=n_tests)

    basis = make_radial_basis(X_train)

    elbo_params = learn(X_train, y_train,
                        basis=basis,
                        basis_args=(lenscale,),
                        basis_args_bounds=[Positive()],
                        var=noise**2,
                        regulariser=reg,
                        use_autograd=use_autograd)

    Ey_e, Vf_e, Vy_e = predict(X_test, basis, *elbo_params)
    Sy_e = np.sqrt(Vy_e)

    fig, ax = plt.subplots()

    ax.plot(X_train.ravel(), y_train, 'k.', label='Training')
    ax.plot(X_test.ravel(), f_test, 'k-', label='Truth')

    ax.plot(X_test.ravel(), Ey_e, 'b--', label='Bayes linear reg')
    ax.fill_between(X_test.ravel(), Ey_e - 2. * Sy_e, Ey_e + 2. * Sy_e,
                    facecolor='lightblue')

    ax.legend()

    ax.grid(True)

    ax.set_ylabel('$y$')
    ax.set_xlabel('$x$')

    plt.show()

if __name__ == '__main__':
    main()