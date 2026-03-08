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

La commandes pour lancer des inférnces, entraînements est 

```
python -m torchtmpl.main config_file.yaml [train,test,test_ensemble..]
```


## Bibliographie 

Panaïotis et al. (2026). *Benchmark of plankton images classification: emphasizing features extraction over classifier complexity*. https://essd.copernicus.org/articles/18/945/2026/essd-18-945-2026-discussion.html