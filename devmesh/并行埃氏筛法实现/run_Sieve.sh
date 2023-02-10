#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling parallel_Sieve algorithm parallel_Sieve.cpp
# dpcpp lab/parallel_Sieve.cpp
icpx -fsycl lab/parallel_Sieve.cpp
if [ $? -eq 0 ]; then ./a.out; fi

