#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setuptools-based setup module."""

from setuptools import setup, find_packages

INSTALL_REQUIRES = ['numpy',
                    'scipy']

SETUP_REQUIRE = ['pytest-runner']

TESTS_REQUIRE = ['pytest>=2.8']

EXTRAS_REQUIRE = {'analysis': ['matplotlib']}

setup(
    name='thrifty',
    version='0.0.1',
    description='Proof-of-concept SDR software for TDOA positioning',
    author='Schalk-Willem Krüger',
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRE,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(exclude=('tests', 'docs', 'old')),
    entry_points={
        'console_scripts': [
            'fastcard_capture = thrifty.fastcard_capture:_main'
        ]
    },
)
