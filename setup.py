#!/usr/bin/env python
""" Setup utility for the pyalacarte package. """

from setuptools import setup

setup(
    name='pyalacarte',
    version='0.1',
    description='Implementation of the A la Carte approximation of Gaussian'
                'processes, amongst other things.',
    author='Daniel Steinberg',
    author_email='daniel.steinberg@nicta.com.au',
    url='',
    packages=['pyalacarte'],
    install_requires=[
        'scipy >= 0.14.1',
        'numpy >= 1.8.2',
        'wget >= 2.2',
        # NLopt >= 2.4.2
    ]
)
