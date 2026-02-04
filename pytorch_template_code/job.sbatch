#!/bin/bash

#SBATCH --job-name=autosend
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

#echo "Copying the dataset to have faster access to the samples"
#mkdir $TMPDIR/dataset
#rsync -a --info=progress2 /mounts/datasets/datasets/2025-ZooCamChallenge $TMPDIR/dataset/
#envsubst </usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/configs/tmpih1fbv2w-config.yml> $TMPDIR/config.yaml

#echo "Verifying that the right configuration has been generated" 
#cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id 5daa01e85ca24731ae048a54750798d34f712d36"
cd $TMPDIR/code
git checkout 5daa01e85ca24731ae048a54750798d34f712d36

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
