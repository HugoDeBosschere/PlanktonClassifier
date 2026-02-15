#!/bin/bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3.40gb:1
#SBATCH --job-name=focalloss
#SBATCH --nodes=1
#SBATCH --partition=prod40
#SBATCH --time=24:00:00
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
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude '__pycache__' --exclude '*.egg-info' --exclude 'build' --exclude 'dist' . $TMPDIR/code

export TMPDIR
export PYTORCH_ALLOC_CONF=expandable_segments:True 

echo "Copying the dataset to have faster access to the samples"
mkdir $TMPDIR/dataset
rsync -aph --info=progress2 ~/dataset/ $TMPDIR/dataset/
envsubst </raid/home/memconflictsllm/hugodebosschere/deep_learning_2025_2026_debosschere_delaby_huhardeaux/pytorch_template_code/configs/tmp6sznirpe-config.yml> $TMPDIR/config.yaml

echo "Verifying that the right configuration has been generated" 
cat $TMPDIR/config.yaml

echo "Checking out the correct version of the code commit_id aadf56e9a5cd19fc8a1336dd6e19d542bb993876"
cd $TMPDIR/code
git checkout aadf56e9a5cd19fc8a1336dd6e19d542bb993876

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
