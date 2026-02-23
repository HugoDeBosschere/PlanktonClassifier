#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb
#SBATCH --job-name=pretrained
#SBATCH --nodes=1
#SBATCH --partition=prod40
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1


current_dir=`pwd`
export PATH=$PATH:~/.local/bin
export HF_TOKEN=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
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
rsync -aph --info=progress2 ~/dataset/ $TMPDIR/dataset/
envsubst </raid/home/students/hugodebosschere/deep_learning_2025_2026_debosschere_delaby_huhardeaux/pytorch_template_code/configs/tmpvlimaob6-config.yml> $TMPDIR/config.yaml

echo "Verifying that the right configuration has been generated" 
cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id 744b6f27a868c0624a35fa709633f51cb3959a3e"
cd $TMPDIR/code
git checkout 744b6f27a868c0624a35fa709633f51cb3959a3e

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main $TMPDIR/config.yaml create_sweep

if [[ $? != 0 ]]; then
    exit -1
fi
