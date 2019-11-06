#!/bin/sh

if [ ! -f "./hubble_orion.tiff" ]; then
  echo "Fetching test image"
  wget https://imgsrc.hubblesite.org/hvi/uploads/image_file/image_attachment/12728/full_tif.tif \
       --no-check-certificate \
       -O hubble_orion.tiff
fi

if [ -z "${EXTRAE_HOME}" ]; then
  echo "Error: \$EXTRAE_HOME is not set."
  exit -1
fi

mkdir -p imagemagick_example_traces

for num_threads in 1 $(seq 2 2 8); do
  export EXTRAE_CONFIG_FILE="./extrae-omp.xml"
  export OMP_NUM_THREADS=${num_threads}
  LD_PRELOAD="${EXTRAE_HOME}/libomptrace.so" magick hubble_orion.tiff -resize 25% hubble_orion_small.tiff
  mpi2prv -f TRACE.mpits -o imagemagick_example_traces/omp${num_threads}.prv -no-keep-mpits
  rm -r set-*
  rm hubble_orion_small.tiff
done

export EXTRAE_CONFIG_FILE="./extrae-omp.xml"
export OMP_NUM_THREADS=8
LD_PRELOAD="${EXTRAE_HOME}/libomptrace.so" magick hubble_orion.tiff -bench 2 -resize 25% hubble_orion_small.tiff
mpi2prv -f TRACE.mpits -o omp_detail.prv -no-keep-mpits
