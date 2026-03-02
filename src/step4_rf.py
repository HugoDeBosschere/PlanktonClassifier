import argparse
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from tqdm import tqdm

def get_tested_configs(log_file):
    """Lit le fichier de log et retourne la liste des configurations RF déjà testées."""
    tested = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    tested.append(data.get("rf_params", {}))
    return tested

def main():
    parser = argparse.ArgumentParser(description="Tuning Random Forest sur features PCA")
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Chemin vers le fichier npz (ex: ../features/run_name/run_name_pca_features_50d.npz)")
    parser.add_argument('--logs_dir', type=str, default='../logs_rf', 
                        help="Dossier racine pour stocker les logs du Random Forest")
    args = parser.parse_args()

    # Récupération automatique du nom du run pour les logs
    run_name = os.path.basename(os.path.dirname(os.path.normpath(args.data_path)))
    if not run_name or run_name == '.':
        filename = os.path.basename(args.data_path)
        run_name = filename.split('_pca_features')[0]

    os.makedirs(args.logs_dir, exist_ok=True)
    log_file = os.path.join(args.logs_dir, f"{run_name}_tuning_logs.jsonl")
    
    print(f"📝 Les logs seront sauvegardés dans : {log_file}")

    print(f"📂 Chargement des données partitionnées depuis {args.data_path}...")
    data = np.load(args.data_path)
    
    # ÉVALUATION RIGOUREUSE : Train et Val sont strictement séparés depuis le début
    X_train = data['train_features']
    y_train = data['train_labels']
    X_val = data['val_features']
    y_val = data['val_labels']
    
    print(f"✅ Dimensions - Train: {X_train.shape}, Val: {X_val.shape}")

    # GRILLE D'HYPERPARAMÈTRES (Définie par l'utilisateur)
    rf_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [25, 35],             
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt'], 
        'class_weight': ['balanced']       
    }

    tested_configs = get_tested_configs(log_file)
    print(f"📊 Configurations déjà testées pour ce modèle : {len(tested_configs)}")

    grid = list(ParameterGrid(rf_param_grid))
    print(f"🚀 Lancement de la recherche sur {len(grid)} configurations totales...")

    for rf_params in tqdm(grid, desc="Tuning RF"):
        
        # Ignorer les configurations déjà calculées (Tolérance aux pannes)
        if rf_params in tested_configs:
            continue

        # Entraînement
        rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Évaluation (Macro-F1 pour correspondre à la métrique Kaggle)
        preds = rf.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average='macro')
        
        # Sauvegarde itérative
        log_entry = {
            'rf_params': rf_params,
            'macro_f1': macro_f1
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        tqdm.write(f"✅ F1-Macro: {macro_f1:.4f} | Params: {rf_params}")

    print("\n🏁 Tuning terminé ! Le fichier de logs est à jour.")

if __name__ == "__main__":
    main()