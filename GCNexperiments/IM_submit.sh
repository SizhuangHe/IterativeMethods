#!/bin/bash
#SBATCH --job-name=iterativeMethodsSweeps
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=5:00:00
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log

cd /home/sizhuang/Research/IterativeMethods
module load python/3.9.12
source iterENV/bin/activate
cd /home/sizhuang/Research/IterativeMethods/GCNexperiments/experiments/sweeps


# Execute each Python file on a separate CPU using srun

srun -n 1 --exclusive python3 sweep_iGCN0.py &
srun -n 1 --exclusive python3 sweep_iGCN1.py &
srun -n 1 --exclusive python3 sweep_iGCN2.py &
srun -n 1 --exclusive python3 sweep_iGCN3.py &
srun -n 1 --exclusive python3 sweep_iGCN4.py &
srun -n 1 --exclusive python3 sweep_iGCN5.py &
srun -n 1 --exclusive python3 sweep_iGCN6.py &
srun -n 1 --exclusive python3 sweep_iGCN7.py &
srun -n 1 --exclusive python3 sweep_GCN7.py &
srun -n 1 --exclusive python3 sweep_GCN77.py &
srun -n 1 --exclusive python3 sweep_iGCNvar2.py &
srun -n 1 --exclusive python3 sweep_iGCNvar22.py 
# Wait for all tasks to finish
wait
