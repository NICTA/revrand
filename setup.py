#!/usr/bin/env python
""" Setup utility for the linearizedReg package. """

from distutils.core import setup

setup(
    name='linearizedReg',
    version='0.1',
    description='Implementation of the A la Carte extended and unscented '
                'Gaussian processes.',
    author='Daniel Steinberg',
    author_email='daniel.steinberg@nicta.com.au',
    url='',
    packages=['linearizedReg'],
    install_requires=[
        "scipy >= 0.15.0",  # These may be able to be lower, just not tested
        "numpy >= 1.8.2",
        # NLopt >= 2.4.2
        ]
)
