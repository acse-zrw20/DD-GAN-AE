#!/bin/bash
#PBS -N create_snapshots
#PBS -o create_snapshots.out
#PBS -lselect=1:ncpus=32:mem=62gb
#PBS -lwalltime=24:00:00
set -vx

cd /rds/general/user/zrw20/home/DD-GAN-AE/submodules/DD-GAN/preprocessing/src
module purge
module load anaconda3/personal
source activate py2
python reconstruct_3D.py --nTime=200 --offset=0 --out_file_base="$HOME/data/raw/slug_255_exp_projected_compressed_" --input_array="sf_prediction_cae.npy"

mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID

echo "... Run finished $(date) ..."