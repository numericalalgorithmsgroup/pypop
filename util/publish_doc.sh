#!/bin/bash

(
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
basedir="$(readlink -f ${scriptdir}/..)"
docdir="${basedir}/doc"
htmldir="${docdir}/build/html/"
ghpdir="${basedir}/gh-pages/"

cd $docdir
make clean; make html

cd $ghpdir
rsync -a --delete ${htmldir} ${ghpdir}
git init
git checkout -b gh-pages
git remote add upstream git@github.com:numericalalgorithmsgroup/pypop.git
git add -A
git commit -m "docs build @ $(date)"
git push --force upstream gh-pages
)
