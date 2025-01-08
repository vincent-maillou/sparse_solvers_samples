#!/bin/bash -l

#SBATCH --job-name=mumps_bench
#SBATCH --partition=debug
#SBATCH --nodes=1  # The number of nodes, you will be currently limited to 1
#SBATCH --ntasks-per-node=1  # The number of (MPI) processes per node. Your total number of processes will be this number times the number of nodes above.
#SBATCH --cpus-per-task=16  # The number of CPU cores per (MPI) process.
#SBATCH --time=00:03:00  # The maximum duration your job will run in hours:minutes:seconds. It will be automatically killed if it takes longer than that. For your tests, 5-10 minutes should be enough.
#SBATCH --output=%x.%j.out # Name of the output file (whatever your application prints in STDOUT).
#SBATCH --error=%x.%j.err # Name of the error file (whatever your application prints in STDERR).
#SBATCH --account=lp16  
#SBATCH --hint=nomultithread
#SBATCH --exclusive

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Not very important for you, but it sets the number of OpenMP threads (per process) equal to the requested number of CPU cores (per process)

srun --cpu-bind=socket ./csr_potrf # Note that you use srun instead of mpirun/mpiexec and that you don't (or you don't have to) set the number of processes. This is already taken from the SBATCH parameters above
