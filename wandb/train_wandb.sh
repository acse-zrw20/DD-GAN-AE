#!/bin/bash
#PBS -N wandb_run
#PBS -o wandb.out
#PBS -j oe
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1
#PBS -lwalltime=24:00:00
set -vx

cd /rds/general/user/zrw20/home/files
module purge
module load anaconda3/personal
source activate tf210 
nvidia-smi
python GAN_train_optimise.py

mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID

echo "... Run finished $(date) ..."
