#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_night
#SBATCH --time=12:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1


current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 68afab5eb937d690376e9d6d1e2a336ee439e7d6"
cd $TMPDIR/code
git checkout 68afab5eb937d690376e9d6d1e2a336ee439e7d6


# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main /usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/configs/tmp50l7i4mo-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
