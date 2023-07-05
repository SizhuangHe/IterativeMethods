#!/bin/bash
#SBATCH --job-name=iterativeMethodsSweeps
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=12:00:00
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log

cd /home/sizhuang/Research/IterativeMethods
module load python/3.9.12
source iterENV/bin/activate
cd /home/sizhuang/Research/IterativeMethods/GCNexperiments/experiments/sweeps


# Execute each Python file on a separate CPU using srun

srun -n 1 --exclusive python3 sweep_la_iGCN0.py &
srun -n 1 --exclusive python3 sweep_la_iGCN1.py &
srun -n 1 --exclusive python3 sweep_la_iGCN2.py &
srun -n 1 --exclusive python3 sweep_la_iGCN3.py &
srun -n 1 --exclusive python3 sweep_la_iGCN4.py &
srun -n 1 --exclusive python3 sweep_la_iGCN5.py &
srun -n 1 --exclusive python3 sweep_la_iGCN6.py &
srun -n 1 --exclusive python3 sweep_la_iGCN7.py &
srun -n 1 --exclusive python3 sweep_et_iGCN0.py &
srun -n 1 --exclusive python3 sweep_et_iGCN1.py &
srun -n 1 --exclusive python3 sweep_et_iGCN2.py &
srun -n 1 --exclusive python3 sweep_et_iGCN3.py &
srun -n 1 --exclusive python3 sweep_et_iGCN4.py &
srun -n 1 --exclusive python3 sweep_et_iGCN5.py &
srun -n 1 --exclusive python3 sweep_et_iGCN6.py &
srun -n 1 --exclusive python3 sweep_et_iGCN7.py &
srun -n 1 --exclusive python3 sweep_iGCN0.py &
srun -n 1 --exclusive python3 sweep_iGCN1.py &
srun -n 1 --exclusive python3 sweep_iGCN2.py &
srun -n 1 --exclusive python3 sweep_iGCN3.py &
srun -n 1 --exclusive python3 sweep_iGCN4.py &
srun -n 1 --exclusive python3 sweep_iGCN5.py &
srun -n 1 --exclusive python3 sweep_iGCN6.py &
srun -n 1 --exclusive python3 sweep_iGCN7.py &
srun -n 1 --exclusive python3 sweep_GCN0.py &
srun -n 1 --exclusive python3 sweep_GCN1.py &
srun -n 1 --exclusive python3 sweep_GCN2.py &
srun -n 1 --exclusive python3 sweep_GCN3.py &
srun -n 1 --exclusive python3 sweep_GCN4.py &
srun -n 1 --exclusive python3 sweep_GCN5.py &
srun -n 1 --exclusive python3 sweep_GCN6.py &
srun -n 1 --exclusive python3 sweep_GCN7.py &
srun -n 1 --exclusive python3 sweep_iGCNvar0.py &
srun -n 1 --exclusive python3 sweep_iGCNvar1.py &
srun -n 1 --exclusive python3 sweep_iGCNvar2.py &
srun -n 1 --exclusive python3 sweep_iGCNvar3.py &
srun -n 1 --exclusive python3 sweep_iGCNvar4.py &
srun -n 1 --exclusive python3 sweep_iGCNvar5.py &
srun -n 1 --exclusive python3 sweep_iGCNvar6.py &
srun -n 1 --exclusive python3 sweep_iGCNvar7.py 
# Wait for all tasks to finish
wait
