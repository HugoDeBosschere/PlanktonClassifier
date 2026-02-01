#!/bin/bash

#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1
#SBATCH --exclude=sh[00,10-19]                                                                                                          
 


current_dir="pwd"
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
echo "New sbatch because I'm getting tired of doing shit"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs $HOME/DeepLearning/pytorch_template_code/ $TMPDIR/code

#echo "Checking out the correct version of the code commit_id 388074fa8530adadf5e17d48650a162cf788d183"
cd $TMPDIR/code
#git checkout 388074fa8530adadf5e17d48650a162cf788d183


echo "Not Setting up the virtual environment (maybe my conda environment will suffice)"
#python3 -m venv venv
#source venv/bin/activate

# Install the library
# Note hugo : I think that the -e is very important because if you don't put it, you risk running an old version of the code 
python -m pip install -e .

echo "Testing"
python -m torchtmpl.main /usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/config-sample.yaml test

if [[ $? != 0 ]]; then
    exit -1
fi
 