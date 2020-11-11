#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from setuptools import setup, find_packages
from versionate import versionate

setup(
    name="pypop",
    version=versionate(),
    url="https://github.com/numericalalgorithmsgroup/pypop.git",
    author="Numerical Algorithms Group",
    author_email="phil.tooley@nag.co.uk",
    description="Python notebook support for POP metrics and reports",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib >= 2",
        "pandas >= 0.24",
        "jupyter",
        "bokeh >= 1",
        "tables",
        "tqdm",
        "ipyfilechooser",
        "selenium",
        "nbformat",
    ],
    zip_safe=True,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pypop-mpi-metrics = pypop.cli.cli:mpi_cli_metrics",
            "pypop-hybrid-metrics = pypop.cli.cli:hybrid_cli_metrics",
            "pypop-openmp-metrics = pypop.cli.cli:openmp_cli_metrics",
            "pypop-preprocess = pypop.cli.cli:preprocess_traces",
            "pypop-idealise-prv = pypop.cli.cli:dimemas_idealise_cli",
            "pypop-copy-examples = pypop.cli.cli:copy_examples",
            "pypop-gui = pypop.cli.cli:pypop_gui",
        ]
    },
)
