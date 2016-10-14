#!/usr/bin/env python
"""Setup utility for the revrand package."""

from setuptools import setup, find_packages

from setuptools.command.test import test as TestCommand


class PyTest(TestCommand, object):

    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        super(PyTest, self).initialize_options()
        self.pytest_args = []

    def finalize_options(self):
        super(PyTest, self).finalize_options()
        self.test_suite = True
        self.test_args = []

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        exit(pytest.main(self.pytest_args))

setup(
    name='revrand',
    version='0.7.0',
    description='A library of scalable Bayesian generalised linear models with'
                ' fancy features',
    author='Daniel Steinberg',
    author_email='daniel.steinberg@nicta.com.au',
    url='http://github.com/nicta/revrand',
    packages=find_packages(),
    cmdclass={
        'test': PyTest
    },
    tests_require=['pytest'],
    install_requires=[
        'scipy >= 0.15.1',
        'numpy >= 1.8.2',
        'six >= 1.9.0',
        'decorator >= 4.0.6',
        'scikit-learn >= 0.18.0'
    ],
    extras_require={
        'demos': [
            'unipath',
            'requests',
            'matplotlib',
        ],
    },
    license="Apache Software License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
