#!/bin/sh

# Default values for certain parameters, can be overridden by environment

: ${EMP_EXTRAE_DIR:=AUTO}       # Extrae location (if AUTO, mpi2prv must be on $PATH)
: ${EMP_MERGE_MAX_MEM:=512}     # maximum memory (per process) for merge
: ${EMP_MPI_LAUNCHER:=mpirun}   # mpi launcher executable
: ${EMP_MPI2PRV_OPTS:=}         # Additional merge options to be passed to mpi2prv

function usage {
  cat <<EOF
Usage: $(basename $0) MPI_procs Extrae_xml MPI_program [program_args...]"

  MPI_procs            - number of MPI processes to use
  Extrae_xml           - Extrae xml configuration file
  MPI_program          - Path to MPI application
  program_args         - Extra arguments to be passed to MPI_program

Output tracefiles will be named \${MPI_program}_np\${nprocs}.prv
EOF
}

# First check we have the correct number of arguments
if [ "$#" -lt 3 ]; then
  usage
  exit 0
fi

# And consume input variables into suitable vars
nprocs=$1
shift
exml_skel=$1
shift
mpi_prog=$1
shift
prog_args=$@

# Now try to locate Extrae if location not provided
if [ ! -z ${EMP_EXTRAE_DIR} ]; then
  extrae_dir=$(dirname $(dirname $(which mpi2prv 2>/dev/null) 2>/dev/null) 2>/dev/null)
fi
# Either way should now be able to find Extrae files
if [ ! -e "$extrae_dir/etc/extrae.sh" ]; then
  echo -e "\nERROR: Cannot locate Extrae installation. Ensure mpi2prv etc. are in your PATH"\
          "or set EMP_EXTRAE_DIR\n" 1>&2
  exit -1
fi
extrae_version=$(${extrae_dir}/bin/extrae-cmd --version 2>/dev/null | grep -oE "[0-9.]*")
echo "Found Extrae version ${extrae_version} in ${extrae_dir}"

# Check that the xml skeleton exists
if [ ! -f "$exml_skel" ]; then
  echo -e "\nERROR: Configuration file ${exml_skel} not found.\n" 1>&2
  exit -1
fi

# Set the config path
export EXTRAE_CONFIG_FILE=${exml_skel}

# Finally create the tracing wrapper on the fly
cat > extraewrapper.sh <<EOF
#!/bin/bash

source ${extrae_dir}/etc/extrae.sh

export LD_PRELOAD=\${EXTRAE_HOME}/lib/libmpitracef.so # For Fortran apps

\$*
EOF
# Remember to make it executable
chmod +x extraewrapper.sh

## Run the program with the tracing wrapper
${EMP_MPI_LAUNCHER} -bind-to core -np ${nprocs} ./extraewrapper.sh ${mpi_prog} ${prog_args}

# Merging in serial or parallel?
if [ "${nprocs}" -eq 1 ]; then
  prvgen=mpi2prv
else
  prvgen=mpimpi2prv
fi

#No do the merge
${EMP_MPI_LAUNCHER} -np ${nprocs} $prvgen -f TRACE.mpits \
                             -e ${mpi_prog} \
                             -o ${mpi_prog}_np${nprocs}.prv \
                             -maxmem ${EMP_MERGE_MAX_MEM} \
                             ${EMP_MPI2PRV_OPTS}
