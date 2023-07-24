#!/bin/bash
#SBATCH --job-name=iterativeMethodsSweeps
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=3
#SBATCH --mem-per-cpu=8000m 
#SBATCH --time=8:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log

cd /home/sizhuang/Research/IterativeMethods
module load python/3.9.12
source iterENV/bin/activate
cd /home/sizhuang/Research/IterativeMethods/GCNexperiments/experiments/sweeps


# Execute each Python file on a separate CPU using srun



srun -n 1 --exclusive python3 sweep_iGCNv_PubMed0.py &
srun -n 1 --exclusive python3 sweep_iGCNv_PubMed5.py &
srun -n 1 --exclusive python3 sweep_iGCNv_PubMed7.py 

# Wait for all tasks to finish
wait