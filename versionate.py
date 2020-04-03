#!/usr/bin/env python3

from warnings import warn
from subprocess import run, PIPE

_k_unknown_ver = "Unknown"


def get_git_verstring():
    git_result = run(
        ['git', 'describe', '--tags', '--dirty', '--long'],
        stdout=PIPE,
        stderr=PIPE,
    )
    if git_result.returncode != 0:
      warn("Git failed with error: {}".format(str(err.stderr)))
      return _k_unknown_ver

    ver_string = git_result.stdout.decode().strip()
    ver_tokens = ver_string.split('-')

    dirty = None
    if ver_tokens[-1] == "dirty":
      dirty = ver_tokens.pop()

    sha = ver_tokens.pop()
    commitcount = ver_tokens.pop()
    tag = '-'.join(ver_tokens)

    if commitcount == '0':
      return "-".join([tag, dirty]) if dirty else tag

    return ver_string


def versionate():

  ver_string = get_git_verstring()
  
  with open('pypop/version', 'wt') as fh:
    fh.write(ver_string)

  return ver_string


if __name__ == "__main__":
    print(get_git_verstring())
