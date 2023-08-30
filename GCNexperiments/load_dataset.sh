#!/bin/bash
#SBATCH --job-name=dld_dt
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=transfer
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=128gb
#SBATCH --time=00:05:00
#SBATCH --output=/home/sh2748/Logs/log_gnn_expt_%J.log
date;hostname;pwd
module load Python/3.8.6

cd /vast/palmer/home.mccleary/sh2748/vanDijkLab/IterativeMethods
source iterENV/bin/activate
cd GCNexperiments/experiments/sweeps/LRGBdatasets/Peptides-structdataset
export PYTHONPATH="/home/sh2748/vanDijkLab/IterativeMethods/GCNexperiments"

# Execute each Python file on a separate CPU using srun

srun -n 1 --exclusive python3 load_dataset.py



# Wait for all tasks to finish
wait
