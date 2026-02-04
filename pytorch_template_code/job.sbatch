#!/bin/bash

#SBATCH --job-name=autosend
#SBATCH --nodes=1
#SBATCH --partition=gpu_inter
#SBATCH --time=2:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --exclude=sh
#SBATCH --exclude=cam
#SBATCH --array=1-1


current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude '__pycache__' --exclude '*.egg-info' --exclude 'build' --exclude 'dist' . $TMPDIR/code

export TMPDIR

#echo "Copying the dataset to have faster access to the samples"
#mkdir $TMPDIR/dataset
#rsync -a --info=progress2 /mounts/datasets/datasets/2025-ZooCamChallenge $TMPDIR/dataset/
#envsubst </usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/configs/tmpj9ohreko-config.yml> $TMPDIR/config.yaml

#echo "Verifying that the right configuration has been generated" 
#cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id f8e4e8484354d8bb9bfa12a69cf12c983378c978"
cd $TMPDIR/code
git checkout f8e4e8484354d8bb9bfa12a69cf12c983378c978

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main /usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/config-no-tmp.yaml train

if [[ $? != 0 ]]; then
    exit -1
fi
