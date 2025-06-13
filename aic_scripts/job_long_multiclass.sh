#!/bin/bash
#SBATCH -J multiclass_langID
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --mem 64G
#SBATCH -G 1
#SBATCH -o job_long_multiclass.out
#SBATCH -e job_long_multiclass.err

venv/bin/python3 src/multiclass.py --epochs 1 --batch-size 128 --model-path long_multiclass_output
venv/bin/python3 src/flores_evaluation.py --model-path long_multiclass_output --type multiclass --encoder-path trainer_output/label_encoder.pkl --output "results/flores_multiclass_long.txt"
