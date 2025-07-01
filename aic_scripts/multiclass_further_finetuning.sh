#!/bin/bash
#SBATCH -J further_multiclass_langID
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --mem 64G
#SBATCH -G 1
#SBATCH -o job_further_multiclass.out
#SBATCH -e job_further_multiclass.err

venv/bin/python3 src/multiclass.py --epochs 1 --batch-size 128 --learning-rate 1e-5 --warmup-ratio 0.05 --model-path further_multiclass_aic_output existing long_multiclass_output
mkdir results
venv/bin/python3 src/flores_evaluation.py --model-path further_multiclass_aic_output --type multiclass --encoder-path trainer_output/label_encoder.pkl --output "results/flores_further_multiclass_long.txt"

