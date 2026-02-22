#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --job-name=hyper10
#SBATCH --nodes=1
#SBATCH --partition=prod10
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-2


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
rsync -aph --info=progress2 ~/dataset/ $TMPDIR/dataset/
envsubst </raid/home/students/hugodebosschere/deep_learning_2025_2026_debosschere_delaby_huhardeaux/pytorch_template_code/configs/tmpj3g3ce2_-config.yml> $TMPDIR/config.yaml

echo "Verifying that the right configuration has been generated" 
cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id 721196551b691e0e5cb653eb1038bc14175da7ae"
cd $TMPDIR/code
git checkout 721196551b691e0e5cb653eb1038bc14175da7ae

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main $TMPDIR/config.yaml launch_agent

if [[ $? != 0 ]]; then
    exit -1
fi
