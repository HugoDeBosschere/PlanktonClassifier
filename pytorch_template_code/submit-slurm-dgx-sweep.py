#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile

def makejob(commit_id, configpath, nruns, func):
    return f"""#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --job-name=sweep
#SBATCH --nodes=1
#SBATCH --partition=prod10
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

export PATH=$PATH:~/.local/bin
export HF_TOKEN=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 # Attention à ne pas commit ce token en clair en production

echo "Session ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "Running on $(hostname)"

# 1. Définition d'un espace de travail unique sur le disque local (rapide) du noeud
export JOB_WORKSPACE="$TMPDIR/$USER/job_${{SLURM_ARRAY_JOB_ID}}_task_${{SLURM_ARRAY_TASK_ID}}"

# 2. Nettoyage automatique à la fin du script (succès ou échec)
trap 'echo "Cleaning up workspace..."; rm -rf "$JOB_WORKSPACE"' EXIT

echo "Creating workspace at $JOB_WORKSPACE"
mkdir -p "$JOB_WORKSPACE/code"
mkdir -p "$JOB_WORKSPACE/dataset"

# Isolation des dossiers WandB pour les sweeps
export WANDB_DIR="$JOB_WORKSPACE/wandb"
mkdir -p "$WANDB_DIR"

echo "Copying the source directory and data"
date
# 3. Copie du code vers l'espace isolé
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude '__pycache__' \\
         --exclude '*.egg-info' --exclude 'build' --exclude 'dist' --exclude 'venv' \\
         --exclude '.venv' . "$JOB_WORKSPACE/code"

export PYTORCH_ALLOC_CONF=expandable_segments:True 

# Variables d'optimisation CPU commentées (à réactiver si besoin de profiler les Dataloaders)
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1

echo "Copying the dataset to have faster access to the samples"
rsync -aph --info=progress2 ./dataset/ "$JOB_WORKSPACE/dataset/"

# 4. Génération de la configuration (envsubst utilisera le bon JOB_WORKSPACE local)
envsubst < "{configpath}" > "$JOB_WORKSPACE/config.yaml"

echo "Verifying that the right configuration has been generated" 
cat "$JOB_WORKSPACE/config.yaml"

echo "Checking out the correct version of the code commit_id {commit_id}"
cd "$JOB_WORKSPACE/code"

echo "=== CHECKING SHARED MEMORY LIMIT ==="
df -h /dev/shm
echo "===================================="

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library inside the virtual env
python -m pip install .

echo "Training"
# Lancement de l'entrainement via le module
python -m torchtmpl.main "$JOB_WORKSPACE/config.yaml" {func}

if [[ $? != 0 ]]; then
    exit -1
fi
"""

def submit_job(job):
    with open("job-dgx.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job-dgx.sbatch")

# Ensure all the modified files have been staged and commited
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode().strip()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [2, 3, 4]:
    print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
    sys.exit(-1)

configpath = sys.argv[1]
if len(sys.argv) <= 2:
    nruns = 1
else:
    nruns = int(sys.argv[2])

if len(sys.argv) == 4:
    func = sys.argv[3]
else:
    func = "create_sweep"

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f"cp {configpath} {tmp_configfilepath}")

# Launch the batch jobs
submit_job(makejob(commit_id, tmp_configfilepath, nruns, func))