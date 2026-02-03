#!/bin/bash

#SBATCH --job-name=tentative
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --exclude=sh[00,03,05,10-19] 
#SBATCH --array=1-1


current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 49cb0cfac06c775d5e9c71ca77a236b50eaf1fac"
cd $TMPDIR/code
git checkout 49cb0cfac06c775d5e9c71ca77a236b50eaf1fac


# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main /usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/configs/tmpc32bqqnj-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
