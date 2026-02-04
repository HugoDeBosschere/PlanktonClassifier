#!/bin/bash

#SBATCH --job-name=gpuusage
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
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

export TMPDIR

echo "Copying the dataset to have faster access to the samples"
mkdir $TMPDIR/dataset
rsync -a --info=progress2 /mounts/datasets/datasets/2025-ZooCamChallenge $TMPDIR/dataset/
envsubst </usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/configs/tmp9akkvl_e-config.yml> $TMPDIR/config.yaml

echo "Verifying that the right configuration has been generated" 
cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id 7e6d41e602beb6b65838870656ed473b72f22abc"
cd $TMPDIR/code
git checkout 7e6d41e602beb6b65838870656ed473b72f22abc

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
