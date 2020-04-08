#!/usr/bin/env python3

from sys import hexversion
from warnings import warn
from subprocess import check_output, CalledProcessError, PIPE

_k_unknown_ver = "Unknown"


def get_git_verstring(silent=True):
    try:
        git_result = check_output(
            ['git', 'describe', '--tags', '--dirty', '--long'],
            stderr=PIPE,
        )
    except CalledProcessError as err:
        if not silent:
            warn("Git failed with error: {}".format(str(err.stderr)))
        return _k_unknown_ver

    ver_string = git_result.decode().strip()
    ver_tokens = ver_string.split('-')

    dirty = None
    if ver_tokens[-1] == "dirty":
      dirty = ver_tokens.pop()

    sha = ver_tokens.pop()
    commitcount = ver_tokens.pop()
    tag = '-'.join(ver_tokens)

    if commitcount == '0':
      return "+".join([tag, dirty]) if dirty else tag

    localver = "+".join([tag,sha])

    return ".".join([localver, dirty]) if dirty else localver


def versionate():

  ver_string = get_git_verstring()

  if ver_string == _k_unknown_ver:
    try:
        with open('pypop/version', 'rt') as fh:
            ver_string = fh.readlines()[0].strip()
    except:
        pass
    finally:
        return ver_string

  with open('pypop/version', 'wt') as fh:
    fh.write(ver_string)

  return ver_string


if __name__ == "__main__":
    print(get_git_verstring())
