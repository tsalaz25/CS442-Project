#!/usr/bin/env bash
# Simple runner for Easley: builds the MPI test binary and executes the five
# standard scaling runs. No CMake or other build systems are used.

set -euo pipefail
cd C
# Ensure the MPI toolchain is available on Easley
module purge
module load openmpi

# Build the test binary
mpicxx -O3 -std=c++17 main.cpp -o test

declare -a commands=(
  "srun -n 1 ./test 2048"
  "srun -n 4 ./test 2048"
  "srun -n 16 ./test 2048"
  "srun -n 4 ./test 4096"
  "srun -n 16 ./test 4096"
)

for cmd in "${commands[@]}"; do
  echo "Running: ${cmd}"
  eval "${cmd}"
  echo
  echo "Done"
  echo "----------------------------------------"
  echo
  sleep 1
done
