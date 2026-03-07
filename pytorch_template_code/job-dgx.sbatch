#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --job-name=sweep
#SBATCH --nodes=1
#SBATCH --partition=prod40  
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

export PATH=$PATH:~/.local/bin
export HF_TOKEN=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 # Attention à ne pas commit ce token en clair en production

echo "Session ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Running on $(hostname)"

# 1. Définition d'un espace de travail unique sur le disque local (rapide) du noeud
export JOB_WORKSPACE="$TMPDIR/$USER/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"

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
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude '__pycache__' \
         --exclude '*.egg-info' --exclude 'build' --exclude 'dist' --exclude 'venv' \
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
envsubst < "/raid/home/projects/pfe-inserm-2026/Projet_Deep/deep_learning_2025_2026_debosschere_delaby_huhardeaux/pytorch_template_code/configs/tmph3we9stt-config.yml" > "$JOB_WORKSPACE/config.yaml"

echo "Verifying that the right configuration has been generated" 
cat "$JOB_WORKSPACE/config.yaml"

echo "Checking out the correct version of the code commit_id 3d854733094726c0b7e7cfed383bb94312db5f9e"
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
python -m torchtmpl.main "$JOB_WORKSPACE/config.yaml" create_sweep

if [[ $? != 0 ]]; then
    exit -1
fi
