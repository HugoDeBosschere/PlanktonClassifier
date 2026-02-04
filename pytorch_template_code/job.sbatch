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
mkdir $TMPDIR/dataset/train 
mkdir $TMPDIR/dataset/test
cp /mounts/datasets/datasets/2025-ZooCamChallenge/train $TMPDIR/dataset/train
cp /mounts/datasets/datasets/2025-ZooCamChallenge/test $TMPDIR/dataset/test
envsubst </usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/configs/tmp3onppaz9-config.yml> $TMPDIR/config.yaml

echo "Verifying that the right configuration has been generated" 
cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id fdf6ed3f041b04ac974a14ba84697973269b22d3"
cd $TMPDIR/code
git checkout fdf6ed3f041b04ac974a14ba84697973269b22d3

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
