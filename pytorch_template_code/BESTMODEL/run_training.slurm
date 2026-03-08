#!/bin/bash -l
#SBATCH --job-name=plankton_mobilenet     # Nom du job
#SBATCH --output=logs/train_%x_%j.out     # Fichier de sortie standard (logs/train_nom_jobID.out)
#SBATCH --error=logs/train_%x_%j.err      # Fichier d'erreur standard
#SBATCH --time=24:00:00                   # Temps alloué (HH:MM:SS) - À ajuster selon ta limite
#SBATCH --partition=prod80                   # Nom de la partition GPU (vérifie la doc de ton cluster)
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1                 # Demande 1 GPU A100
#SBATCH --cpus-per-task=32               # Nombre de cœurs CPU (doit correspondre à num_workers)
#SBATCH --nodes=1                         # Un seul nœud de calcul
#SBATCH --ntasks=1

# --- 1. Préparation de l'environnement ---
# (Décommente et adapte selon les modules de ton cluster)
#module purge
#module load python/3.10
#module load cuda/11.8

# Activation de ton environnement virtuel si tu en utilises un
# source source .venv/bin/activate

# --- 1. Préparation de l'environnement ---

# On se place dans le dossier du projet (optionnel mais recommandé)
cd /raid/home/projects/pfe-inserm-2026/Time_attack/

# Activation de ton environnement virtuel
source .venv/bin/activate

# Vérification optionnelle pour s'assurer que c'est le bon Python qui tourne
which python

# Affiche les informations sur le GPU alloué pour vérifier que tout est OK
echo "--- GPU alloué ---"
nvidia-smi

# --- 2. Lancement de l'entraînement ---
echo "--- Début de l'entraînement ---"

# On utilise srun pour exécuter le script Python avec les paramètres optimisés pour l'A100
srun python -u step1_finetune.py \
    --data_dir /raid/home/projects/pfe-inserm-2026/Time_attack/dataset/train \
    --batch_size 512 \
    --num_workers 32 \
    --lr 0.0014 \
    --epochs 20 \
    --model_name mobilenetv2_140_clipping_CosineSched

echo "--- Entraînement terminé ---"