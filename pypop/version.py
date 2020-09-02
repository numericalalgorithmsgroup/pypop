#!/usr/bin/env python3

from warnings import warn
from pkg_resources import resource_filename
from subprocess import run, PIPE

_k_unknown_ver = "Unknown"


def get_git_verstring():
    git_result = run(
        ["git", "describe", "--tags", "--dirty", "--long"], stdout=PIPE, stderr=PIPE,
    )
    if git_result.returncode != 0:
        return _k_unknown_ver

    ver_string = git_result.stdout.decode().strip()
    ver_tokens = ver_string.split("-")

    dirty = None
    if ver_tokens[-1] == "dirty":
        dirty = ver_tokens.pop()

    sha = ver_tokens.pop()
    commitcount = ver_tokens.pop()
    tag = "-".join(ver_tokens)

    if commitcount == "0":
        return "-".join([tag, dirty]) if dirty else tag

    return ver_string


def get_version():

    gitver = get_git_verstring()
    if gitver != _k_unknown_ver:
        return gitver

    try:
        with open(resource_filename(__name__, "version"), "rt") as fh:
            fileversion = fh.readlines()[0].strip()
            return fileversion
    except FileNotFoundError:
        pass

    return _k_unknown_ver
