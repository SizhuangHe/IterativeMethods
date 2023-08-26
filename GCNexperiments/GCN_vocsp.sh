#!/bin/bash
#SBATCH --job-name=GNN_swp
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=10:00:00
#SBATCH --output=/home/sh2748/Logs/log_gnn_expt_%J.log
date;hostname;pwd
module load Python/3.8.6

cd /vast/palmer/home.mccleary/sh2748/vanDijkLab/IterativeMethods
source iterENV/bin/activate
cd GCNexperiments/experiments/sweeps/LRGBdatasets/PascalVOC-SPdataset
export PYTHONPATH="/home/sh2748/vanDijkLab/IterativeMethods/GCNexperiments"

# Execute each Python file on a separate CPU using srun

srun -n 1 --exclusive python3 sweep_GCN_PVOC-SP.py 



# Wait for all tasks to finish
wait
