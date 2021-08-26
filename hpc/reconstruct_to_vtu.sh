#!/bin/bash
#PBS -N reconstruct_to_vtu
#PBS -o reconstruct_to_vtu.out
#PBS -lselect=1:ncpus=4:mem=16gb
#PBS -lwalltime=24:00:00
set -vx

cd /rds/general/user/zrw20/home/DD-GAN-AE/preprocessing/src
module purge
module load anaconda3/personal
source activate py2
python reconstruct_3D.py --nTime=200 --offset=0 --out_file_base="$HOME/data/raw/slug_255_exp_projected_compressed_" --input_array="sf_prediction_cae.npy"

mkdir $WORK/$PBS_JOBID
cp -r * $WORK/$PBS_JOBID

echo "... Run finished $(date) ..."