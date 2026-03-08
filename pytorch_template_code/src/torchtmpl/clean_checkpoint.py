import torch
import argparse
from pathlib import Path

def clean_compiled_checkpoint(input_path: str, output_path: str = None):
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {in_path}")

    # Génération automatique du nom de sortie
    if output_path is None:
        output_path = in_path.with_name(f"{in_path.stem}_clean{in_path.suffix}")
    else:
        output_path = Path(output_path)

    # Chargement sur CPU (standard pour l'I/O de fichiers .pth)
    state_dict = torch.load(in_path, map_location='cpu', weights_only=False)

    # Logique strictement identique au script d'inférence : 
    # Remplacement aveugle de '_orig_mod.' partout dans la clé
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Sauvegarde directe en format plat
    torch.save(clean_state_dict, output_path)

    print(f"Succès : Checkpoint nettoyé sauvegardé vers {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supprime '_orig_mod.' des checkpoints compilés.")
    parser.add_argument("input", type=str, help="Chemin vers le fichier .pth original")
    parser.add_argument("-o", "--output", type=str, default=None, help="Chemin de sortie (optionnel)")
    
    args = parser.parse_args()
    clean_compiled_checkpoint(args.input, args.output)