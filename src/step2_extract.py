import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import h5py
import numpy as np

# Importation de nos modules
from dataset import get_transforms, datasets, ZooCamTestDataset
from model import get_model
from utils import get_or_create_split

def parse_args():
    parser = argparse.ArgumentParser(description="Extraction des Deep Features avec MobileNetV2")
    parser.add_argument('--run_dir', type=str, required=True, help="Dossier du run contenant best_model.pth (ex: ../models/mobilenetv2_140_ep10_...)")
    parser.add_argument('--data_dir_train', type=str, default='../data/Train', help="Dossier des images d'entraînement")
    parser.add_argument('--data_dir_test', type=str, default='../data/Test/imgs', help="Dossier des images de test")
    parser.add_argument('--out_dir', type=str, default='../features', help="Dossier racine pour sauvegarder les HDF5")
    parser.add_argument('--batch_size', type=int, default=128, help="Taille du batch")
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

def append_to_h5(dataset, new_data):
    """Fonction utilitaire pour redimensionner et ajouter des données au fichier HDF5."""
    current_size = dataset.shape[0]
    new_size = current_size + new_data.shape[0]
    dataset.resize(new_size, axis=0)
    dataset[current_size:new_size] = new_data

def main():
    args = parse_args()
    
    # Récupération propre du nom du dossier du modèle
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    run_out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Périphérique cible : {device}")

    # 1. SCAN UNIQUE ET SÉPARATION DES DONNÉES SANS FUITE
    print("📂 Préparation des DataLoaders...")
    full_train_dataset = datasets.ImageFolder(
        root=args.data_dir_train, 
        transform=get_transforms(is_train=False) # Pas d'augmentation à l'extraction
    )
    
    # Récupération du registre de split statique
    split_path = os.path.join(args.data_dir_train, "split_registry.json")
    split_registry = get_or_create_split(full_train_dataset.targets, split_file=split_path)
    
    # Création des sous-ensembles garantis étanches
    train_subset = Subset(full_train_dataset, split_registry["train"])
    val_subset = Subset(full_train_dataset, split_registry["val"])

    # Dataset de test
    test_dataset = ZooCamTestDataset(
        img_dir=args.data_dir_test, 
        transform=get_transforms(is_train=False)
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 2. CHARGEMENT DU MODÈLE
    print(f"🧠 Chargement du modèle depuis {args.run_dir}...")
    model_path = os.path.join(args.run_dir, "best_model.pth")
    num_classes = len(full_train_dataset.classes)
    
    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Indispensable pour figer BatchNorm et Dropout

    # 3. PRÉPARATION DU FICHIER HDF5
    h5_path = os.path.join(run_out_dir, "extracted_features.h5")
    print(f"📦 Création de la base de données HDF5 partitionnée : {h5_path}")
    
    with h5py.File(h5_path, 'w') as h5f:
        
        # Structure Train
        train_feat_ds = h5f.create_dataset('train_features', shape=(0, 1792), maxshape=(None, 1792), dtype='float32', chunks=True)
        train_lbl_ds  = h5f.create_dataset('train_labels', shape=(0,), maxshape=(None,), dtype='int64', chunks=True)
        
        # Structure Val
        val_feat_ds = h5f.create_dataset('val_features', shape=(0, 1792), maxshape=(None, 1792), dtype='float32', chunks=True)
        val_lbl_ds  = h5f.create_dataset('val_labels', shape=(0,), maxshape=(None,), dtype='int64', chunks=True)
        
        # Structure Test
        test_feat_ds = h5f.create_dataset('test_features', shape=(0, 1792), maxshape=(None, 1792), dtype='float32', chunks=True)
        str_dtype = h5py.string_dtype(encoding='utf-8')
        test_file_ds = h5f.create_dataset('test_filenames', shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=True)

        # 4. EXTRACTION (TRAIN)
        print("\n🚀 Extraction des features du jeu d'entraînement...")
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extraction Train"):
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    features = model(images, extract_features=True)
                append_to_h5(train_feat_ds, features.cpu().numpy())
                append_to_h5(train_lbl_ds, labels.cpu().numpy())

        # 5. EXTRACTION (VAL)
        print("\n🚀 Extraction des features du jeu de validation...")
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Extraction Val"):
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    features = model(images, extract_features=True)
                append_to_h5(val_feat_ds, features.cpu().numpy())
                append_to_h5(val_lbl_ds, labels.cpu().numpy())

        # 6. EXTRACTION (TEST)
        print("\n🚀 Extraction des features du jeu de test...")
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="Extraction Test"):
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    features = model(images, extract_features=True)
                append_to_h5(test_feat_ds, features.cpu().numpy())
                append_to_h5(test_file_ds, np.array(filenames, dtype=object))

    # Sauvegarde des noms de classes pour référence
    classes_path = os.path.join(run_out_dir, "class_names.txt")
    with open(classes_path, 'w') as f:
        for c in full_train_dataset.classes:
            f.write(f"{c}\n")

    print("\n✅ Extraction terminée avec succès !")
    print(f"💾 Espace latent stocké dans : {h5_path}")

if __name__ == "__main__":
    main()