#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from setuptools import setup, find_packages

setup(
    name="pypop",
    version="0.1.0",
    url="https://github.com/ptooley/ezpop.git",
    author="Phil Tooley",
    author_email="phil.tooley@nag.co.uk",
    description="Python notebook support for POP metrics and reports",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib >= 2", "pandas >= 0.24"],
    extras_require={'tqdm': ['tqdm']},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pypop-mpi-metrics = pypop.cli:mpi_cli_metrics",
            "pypop-hybrid-metrics = pypop.cli:hybrid_cli_metrics",
        ]
    },
)
