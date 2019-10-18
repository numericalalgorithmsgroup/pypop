#!/bin/bash

(
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
basedir="$(readlink -f ${scriptdir}/..)"
docdir="${basedir}/doc/source"
ghpdir="${basedir}/gh-pages"

cd $ghpdir
git checkout gh-pages
touch .nojekyll

cd $docdir
sphinx-build -an . $ghpdir
cd $ghpdir

if ! [[ -d .git ]]; then
  echo "gh-pages repo not created, to push docs you need to:"
  echo "1. create an empty git repo in ${ghpdir},"
  echo "2. set upstream to git@github.com:numericalalgorithmsgroup/pypop,"
  echo "3. check out the gh-pages branch from upstream"
  echo "4. rerun this script"
  exit -1
fi
git add -A
git commit -m "docs build @ $(date)"
git push upstream gh-pages
)
