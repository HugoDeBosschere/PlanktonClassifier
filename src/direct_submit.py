import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importation de tes modules
from dataset import get_transforms, ZooCamTestDataset
from model import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Génération directe d'une soumission Kaggle depuis le CNN (MobileNetV2)")
    parser.add_argument('--run_dir', type=str, required=True, 
                        help="Dossier du run contenant best_model.pth (ex: ../models/mobilenetv2_140_ep10...)")
    parser.add_argument('--data_dir_test', type=str, default='../data/Test/imgs', 
                        help="Dossier contenant les images de test non labellisées")
    parser.add_argument('--out_dir', type=str, default='../submissions', 
                        help="Dossier cible pour le fichier CSV Kaggle")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Création du dossier cible pour les soumissions
    os.makedirs(args.out_dir, exist_ok=True)
    
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    csv_path = os.path.join(args.out_dir, f"{run_name}_direct_submission.csv")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 Génération de la soumission directe ('Baseline CNN')")
    print(f"⚙️ Périphérique cible : {device}")

    # 2. Préparation du DataLoader de Test
    print(f"📂 Chargement des images de test depuis {args.data_dir_test}...")
    test_dataset = ZooCamTestDataset(
        img_dir=args.data_dir_test, 
        transform=get_transforms(is_train=False) # Résolution native, pad carré
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # 3. Chargement de l'architecture et des poids
    # Caractéristiques : MobileNetV2 (width_mult=1.4) + Tête personnalisée (Dropout -> Linear 600 -> BatchNorm1d -> ReLU -> Dropout -> Linear 86)
    print("🧠 Chargement du réseau MobileNetV2 finetuné...")
    model_path = os.path.join(args.run_dir, "best_model.pth")
    
    # On suppose 86 classes comme défini par ton ImageFolder d'origine
    num_classes = 86 
    
    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Fige les statistiques de BatchNorm et désactive le Dropout

    # 4. Inférence sur le jeu de test
    print("\n🔍 Lancement de l'inférence...")
    predictions = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Prédiction Test"):
            images = images.to(device)
            
            # Forward pass avec précision mixte
            with torch.amp.autocast('cuda'):
                logits = model(images)
            
            # On récupère l'indice de la classe ayant la plus forte probabilité (logique argmax)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # On stocke le couple (nom_du_fichier, entier_de_la_classe)
            for fname, pred in zip(filenames, batch_preds):
                # On s'assure de ne garder que le nom du fichier brut (ex: "0.jpg")
                clean_name = os.path.basename(fname)
                predictions.append((clean_name, pred))

    # 5. Formatage et Export Kaggle
    print("\n📝 Écriture du fichier de soumission...")
    df_submission = pd.DataFrame(predictions, columns=['imgname', 'label'])
    
    # Sauvegarde en CSV (sans index, avec le header exact imposé par la plateforme)
    df_submission.to_csv(csv_path, index=False)
    
    print(f"✅ Soumission générée avec succès : {csv_path}")
    print("Tu peux maintenant uploader ce fichier sur Kaggle !")

if __name__ == "__main__":
    main()