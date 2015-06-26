#!/usr/bin/env python
""" Setup utility for the pyalacarte package. """

from distutils.core import setup

setup(
    name='pyalacarte',
    version='0.1',
    description='Implementation of the A la Carte approximation of Gaussian'
                ' processes, amongst other things.',
    author='Daniel Steinberg',
    author_email='daniel.steinberg@nicta.com.au',
    url='',
    packages=['pyalacarte'],
    install_requires=[
        "scipy >= 0.15.0",  # These may be able to be lower, just not tested
        "numpy >= 1.8.2",
        # NLopt >= 2.4.2
        ]
)
