#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
analysis_dir=$(readlink -f ${script_dir}/../basicanalysis)

for prvfile in "$@"; do
  ${script_dir}/chop_roi.sh ${prvfile}
done

cp -r ${analysis_dir}/modelfactors.py ${analysis_dir}/cfgs .

./modelfactors.py */roi*.prv


