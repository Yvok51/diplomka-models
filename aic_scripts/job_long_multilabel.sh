#!/bin/bash
#SBATCH -J multilabel_langID
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --mem 64G
#SBATCH -G 1
#SBATCH -o job_long_multilabel.out
#SBATCH -e job_long_multilabel.err

venv/bin/python3 src/multilabel.py --epochs 1 --batch-size 128 --model-path long_multilabel_output --synthetic-proportion 0.75
venv/bin/python3 src/flores_evaluation.py --model-path long_multilabel_output --type multilabel --encoder-path trainer_output/multilabel_encoder.pkl --output "results/flores_multilabel_long.txt"


