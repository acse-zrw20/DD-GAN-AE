#!/bin/bash
#PBS -N wandb_run_sf
#PBS -o wandb_run_sf.out
#PBS -j oe
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1
#PBS -lwalltime=24:00:00
set -vx

cd /rds/general/user/zrw20/home/DD-GAN-AE/ddganAE/wandb
module purge
module load anaconda3/personal
module load cuda/11.0.1
source activate tf 
nvidia-smi
python train_wandb_sf.py --model="$0" --datafile="$HOME/data/processed/sf_snapshots_200timesteps_rand.npy"

mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID

echo "... Run finished $(date) ..."
