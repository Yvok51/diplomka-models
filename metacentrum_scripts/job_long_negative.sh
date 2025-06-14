#!/bin/bash
#PBS -N negative_langID
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb:scratch_local=10gb:gpu_mem=16gb
#PBS -l walltime=500:00:00

# define a DATADIR variable: directory where the input files are taken from and where the output will be copied to
MYHOME=/storage/brno2/home/michal-tichy
DATADIR="${MYHOME}/diplomka/diplomka-models" # substitute username and path to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python/python-3.10.4-intel-19.0.4-sc7snnf

export HOME="${MYHOME}"
export TMPDIR="${SCRATCHDIR}"
cd $DATADIR

git pull
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r requirements-pytorch.txt

venv/bin/python3 src/multilabel.py --epochs 1 --batch-size 128 --model-path long_negative_output --synthetic-proportion 0.5 --negative-sampling
venv/bin/python3 src/flores_evaluation.py --model-path long_negative_output --type multilabel --encoder-path trainer_output/multilabel_encoder.pkl --output "results/flores_multilabel_long.txt"
