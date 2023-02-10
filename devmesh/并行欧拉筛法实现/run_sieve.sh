#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Parallel_Euler_Sieve
dpcpp lab/Parallel_Euler_Sieve.cpp
if [ $? -eq 0 ]; then ./a.out; fi
