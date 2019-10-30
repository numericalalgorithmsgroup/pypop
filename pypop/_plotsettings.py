#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

pagedata = {"textwidth": 6}  # width checked for \documentclass[a4paper]{book}

figparams = {
    "single.figsize": (pagedata["textwidth"], pagedata["textwidth"] * 0.66),
    "single.width": 0.8,
    "single.height": 0.8,
    "single.leftpad": 0.1,
    "single.bottompad": 0.12,
}

figparams["single.axlayout"] = (
    (
        figparams["single.leftpad"],
        figparams["single.bottompad"],
        figparams["single.width"],
        figparams["single.height"],
    ),
)

pypop_mpl_params = {
    "text.usetex": False,
    "text.latex.preamble": [r"\usepackage{color}", r"\usepackage{amssymb}"],
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "font.size": 10,
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "lines.linewidth": 1,
    "legend.fontsize": 10,
    "legend.frameon": False,
}
