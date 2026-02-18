#!/bin/bash

#SBATCH --job-name=pretrained
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --exclude=sh[00,01,05,03,10-19,20]    
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
export PYTORCH_ALLOC_CONF=expandable_segments:True 

echo "Copying the dataset to have faster access to the samples"
mkdir $TMPDIR/dataset
rsync -aph --info=progress2 /mounts/datasets/datasets/2025-ZooCamChallenge $TMPDIR/dataset/
envsubst </usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/configs/tmpebwec2rb-config.yml> $TMPDIR/config.yaml

echo "Verifying that the right configuration has been generated" 
cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id 5cd2a516103a1d2ca13f54c901f020c875b8c83f"
cd $TMPDIR/code
git checkout 5cd2a516103a1d2ca13f54c901f020c875b8c83f

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main $TMPDIR/config.yaml train

if [[ $? != 0 ]]; then
    exit -1
fi
