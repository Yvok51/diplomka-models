#!/bin/bash
#SBATCH -J negative_langID
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -G 1
#SBATCH -o job_long_negative.out
#SBATCH -e job_long_negative.err

venv/bin/python3 src/multilabel.py --epochs 1 --batch-size 64 --model-path long_negative_output --synthetic-proportion 0.5 --negative-sampling
mkdir results
venv/bin/python3 src/flores_evaluation.py --model-path long_negative_output --type multilabel --encoder-path trainer_output/multilabel_encoder.pkl --output "results/flores_multilabel_long.txt"
