#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling
mpiicpc -cxx=icpx lab/prime_mpi_omp.cpp -fiopenmp -fopenmp-targets=spir64 -o bin/prime_mpi.x
