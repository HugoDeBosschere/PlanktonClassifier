# Projet de deep learning 2025-2026

Ceci est le repo de Hugo de Bosschere, Antoine Delaby et Louis Huhardeaux.
Il est constitué de 3 branches:
- `main`
- `dev_Louis`
- `tta`

La branche `main` est celle qui contient le code le plus aboutit et qui reprend le template. Les deux autres branches sont plus exploratoires et ne reprennent pas nécessairement la structure du template.

## Organisation du travail

Le travail fourni peut se diviser en 3 axes :
- entraîner des modèles from scratch, en essayant de trouver les meilleurs hyperparamètres et en surveillant l'évolution des sweeps via Weights and Biais 
- finetuner des modèles préentraînés en s'inspirant de l'article donné en bibliographie
- implémenter test time augmentation, de l'ensembling de modèles et du zero shot de modèle comme CLIP


On a utilisé des outils d'intelligence artificielle générative à certaines étapes du projet :  debugging, amélioration du code existant etc...

## Commande

Il est recommandé de lancer ces commandes sur le dgx (et non sur le dce) car les données sont téléchargées sur dans un dossier temporaire avant de lancer l'entrainement / le test, ce qui est très long sur le dce mais rapide sur le dgx
La commandes pour lancer des inférnces, entraînements est 

Pour lancer un entrainement isolé (création de sweep automatique pour la visualisation et le logging sur wandb)

```
python -m submit-slurm-dgx-sweep.py config-pretrained.yaml nruns create_sweep
```

Pour créer un wandb sweep, il faut d'abord exécuter en ligne de commande

```
wandb sweep config-wandb-sweep-dgx.yaml 
```

Il faut ensuite mettre le sweep-id obtenu dans le fichier sweep-id-dgx 
puis pour qu'un (ou plusieurs) gpu rejoigne le sweep:

```
python -m submit-slurm-dgx-sweep.py sweep-id-dgx.yaml nruns create_sweep
```

pour tester les modèles, il faut exécuter: 

```
python -m submit-slurm-dgx-sweep.py best_model.yaml 1 test_ensemble
```

ou l'on aura mis dans best_model.yaml les configurations idoines (dans model_path le path vers les poids du modèle, dans model_config le path vers la config du modèle)

## Bibliographie 

Panaïotis et al. (2026). *Benchmark of plankton images classification: emphasizing features extraction over classifier complexity*. https://essd.copernicus.org/articles/18/945/2026/essd-18-945-2026-discussion.html