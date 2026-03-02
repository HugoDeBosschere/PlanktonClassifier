import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

def parse_args():
    parser = argparse.ArgumentParser(description="Étape 3 : Réduction de dimension par IPCA")
    parser.add_argument('--features_path', type=str, required=True, help="Chemin du HDF5 produit à l'étape 2 (ex: ../features/modele_X/extracted_features.h5)")
    parser.add_argument('--n_components', type=int, default=50, help="Dimensions à conserver")
    parser.add_argument('--chunk_size', type=int, default=4096, help="Taille des blocs pour l'ajustement séquentiel")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # On détermine le dossier de sauvegarde (le même que celui du HDF5 source)
    out_dir = os.path.dirname(args.features_path)
    run_name = os.path.basename(os.path.normpath(out_dir))
    out_file = os.path.join(out_dir, f"{run_name}_pca_features_{args.n_components}d.npz")
    
    print(f"📦 Ouverture de la base HDF5 partitionnée : {args.features_path}")
    
    with h5py.File(args.features_path, 'r') as h5f:
        # Récupération des pointeurs vers les datasets (sans tout charger en RAM)
        train_features = h5f['train_features']
        train_labels = h5f['train_labels'][:] 
        
        val_features = h5f['val_features']
        val_labels = h5f['val_labels'][:]
        
        test_features = h5f['test_features']
        test_filenames = h5f['test_filenames'][:]
        
        n_train = train_features.shape[0]
        n_val = val_features.shape[0]
        n_test = test_features.shape[0]
        
        # 1. IPCA Fit (STRICTEMENT SUR TRAIN)
        # On s'assure qu'aucune information de la validation ne contamine la matrice de covariance
        print(f"\n📉 Ajustement de l'IPCA (Réduction à {args.n_components} dimensions)...")
        ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.chunk_size)
        
        for i in tqdm(range(0, n_train, args.chunk_size), desc="IPCA Fit (Train)"):
            ipca.partial_fit(train_features[i : i + args.chunk_size])
            
        print(f"✨ Variance expliquée conservée : {np.sum(ipca.explained_variance_ratio_) * 100:.2f}%")

        # 2. IPCA Transform
        # On projette les 3 ensembles dans le même espace latent
        print("\n🧮 Projection des données d'entraînement...")
        train_pca = np.empty((n_train, args.n_components), dtype=np.float32)
        for i in tqdm(range(0, n_train, args.chunk_size), desc="Transform Train"):
            train_pca[i : i + args.chunk_size] = ipca.transform(train_features[i : i + args.chunk_size])
            
        print("🧮 Projection des données de validation...")
        val_pca = np.empty((n_val, args.n_components), dtype=np.float32)
        for i in tqdm(range(0, n_val, args.chunk_size), desc="Transform Val"):
            val_pca[i : i + args.chunk_size] = ipca.transform(val_features[i : i + args.chunk_size])
            
        print("🧮 Projection des données de test...")
        test_pca = np.empty((n_test, args.n_components), dtype=np.float32)
        for i in tqdm(range(0, n_test, args.chunk_size), desc="Transform Test"):
            test_pca[i : i + args.chunk_size] = ipca.transform(test_features[i : i + args.chunk_size])

    # 3. Sauvegarde ultra-rapide en RAM (NumPy Zipped)
    # L'objet NPZ contiendra désormais proprement train, val et test pour l'étape 4
    print(f"\n💾 Sauvegarde des matrices réduites dans : {out_file}")
    np.savez_compressed(
        out_file,
        train_features=train_pca,
        train_labels=train_labels,
        val_features=val_pca,
        val_labels=val_labels,
        test_features=test_pca,
        test_filenames=test_filenames
    )
    print("✅ Étape 3 terminée avec succès !")

if __name__ == "__main__":
    main()