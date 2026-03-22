# Deep Learning Project 2025-2026

This repository belongs to Hugo de Bosschere, Antoine Delaby, and Louis Huhardeaux.
It consists of 3 branches:
- `main`: Contains the most finalized code and follows the project template.
- `dev_Louis`: Exploratory branch; does not necessarily follow the template structure.
- `tta`: Exploratory branch focused on Test-Time Augmentation.

## Work Organization

The project is divided into three main axes:
- **Training from scratch**: Optimizing hyperparameters and monitoring sweeps via **Weights & Biases**.
- **Fine-tuning**: Adapting pre-trained models based on the methodology provided in the bibliography.
- **Advanced Techniques**: Implementing Test-Time Augmentation (TTA), model ensembling, and zero-shot classification using CLIP.

Generative AI tools were utilized during development for debugging and code optimization.

## Usage

### Isolated Training
To launch an isolated training run (automatic sweep creation for visualization and logging on W&B):

```bash
python -u submit-slurm-dgx-sweep.py config-pretrained.yaml nruns create_sweep
```

### Weights & Biases Sweeps
To create a W&B sweep, first execute:

```bash
pip install wandb 
wandb sweep config-wandb-sweep-dgx.yaml 
```

1. Copy the resulting `sweep-id` into the `sweep-id-dgx` file.
2. To have one or more GPUs join the sweep, run:

```bash
python -u submit-slurm-dgx-sweep.py sweep-id-dgx.yaml nruns create_sweep
```

### Testing and Inference
To test the models, run:

```bash
python -u submit-slurm-dgx-sweep.py best_model.yaml 1 test_ensemble
```
*Note: Ensure `best_model.yaml` is configured with the correct `model_path` (weights) and `model_config` (architecture).*

## Bibliography 

Panaïotis et al. (2026). *Benchmark of plankton images classification: emphasizing features extraction over classifier complexity*. [Link to Article](https://essd.copernicus.org/articles/18/945/2026/essd-18-945-2026-discussion.html)

## Explainer Video

A video (in French) detailing our methodology and results can be found here: 
[Watch on YouTube](https://www.youtube.com/watch?v=1JTjhFDgrtw)
