 #!/bin/bash
#PBS -N batch_job_example
#PBS -l select=1:ncpus=8:mem=256gb:scratch_local=10gb
#PBS -l walltime=16:00:00

# define a DATADIR variable: directory where the input files are taken from and where the output will be copied to
DATADIR=/storage/brno2/home/michal-tichy/diplomka/diplomka-models # substitute username and path to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python/python-3.10.4-intel-19.0.4-sc7snnf

export HOME=$DATADIR
cd $DATADIR

git pull
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r requirements-pytorch.txt

venv/bin/python3 src/multilabel --epochs 2 --samples-per-language 10000 --batch-size 96
