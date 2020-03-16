#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from setuptools import setup, find_packages

setup(
    name="pypop",
    version="0.2.0rc8",
    url="https://github.com/numericalalgorithmsgroup/pypop.git",
    author="Numerical Algorithms Group",
    author_email="phil.tooley@nag.co.uk",
    description="Python notebook support for POP metrics and reports",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib >= 2",
        "pandas >= 0.24",
        "jupyter",
        "bokeh >= 1",
        "tables",
        "tqdm",
        "ipyfilechooser",
    ],
    zip_safe=True,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pypop-mpi-metrics = pypop.cli:mpi_cli_metrics",
            "pypop-hybrid-metrics = pypop.cli:hybrid_cli_metrics",
            "pypop-openmp-metrics = pypop.cli:openmp_cli_metrics",
            "pypop-preprocess = pypop.cli:preprocess_traces",
            "pypop-idealise-prv = pypop.cli:dimemas_idealise",
            "pypop-copy-examples = pypop.cli:copy_examples",
        ]
    },
)
