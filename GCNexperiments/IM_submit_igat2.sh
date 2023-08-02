#!/bin/bash
#SBATCH --job-name=iterativeMethodsSweeps
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=6
#SBATCH --mem-per-cpu=6000m 
#SBATCH --time=15:00:00
#SBATCH --account=lsa1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log

cd /home/sizhuang/Research/IterativeMethods
module load python/3.9.12
source iterENV/bin/activate
cd /home/sizhuang/Research/IterativeMethods/GCNexperiments/experiments/sweeps


# Execute each Python file on a separate CPU using srun
srun -n 2 --exclusive python3 sweep_iGAT_PubMed0.py &
srun -n 2 --exclusive python3 sweep_iGAT_PubMed5.py &
srun -n 2 --exclusive python3 sweep_iGAT_PubMed7.py 

wait