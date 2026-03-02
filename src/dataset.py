import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import shutil

class ResizeAndPadToSquare:
    """
    Redimensionne le côté le plus long à la taille cible (224), 
    puis pad le côté le plus court avec la valeur médiane pour obtenir un carré[cite: 204, 206].
    """
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, image):
        w, h = image.size
        
        # 1. Redimensionnement homothétique
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        if new_w == self.target_size and new_h == self.target_size:
            return image
            
        # 2. Extraction des bordures et calcul de la médiane [cite: 206]
        img_np = np.array(image)
        if img_np.ndim == 3: # 3 canaux
            top, bottom = img_np[0, :, :], img_np[-1, :, :]
            left, right = img_np[:, 0, :], img_np[:, -1, :]
            borders = np.concatenate([top, bottom, left, right], axis=0)
            median_val = tuple(np.median(borders, axis=0).astype(int))
        else: # Niveaux de gris
            top, bottom = img_np[0, :], img_np[-1, :]
            left, right = img_np[:, 0], img_np[:, -1]
            borders = np.concatenate([top, bottom, left, right])
            median_val = int(np.median(borders))

        # 3. Calcul du padding pour centrer l'image
        pad_left = (self.target_size - new_w) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_top = (self.target_size - new_h) // 2
        pad_bottom = self.target_size - new_h - pad_top
        
        return ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=median_val)


def get_transforms(is_train=True, rotation_degrees=0):
    """
    Génère la pipeline de transformation.
    - is_train : active l'augmentation de données[cite: 220].
    - rotation_degrees : permet d'ajouter une rotation (0 pour désactiver)[cite: 222].
    """
    # Etape 1 : Le resize strict et le padding [cite: 204, 206]
    base_transforms = [ResizeAndPadToSquare(224)]
    
    # Etape 2 : L'augmentation (si entraînement) [cite: 220]
    if is_train:
        base_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5), # [cite: 220]
            transforms.RandomVerticalFlip(p=0.5),   # [cite: 220]
            transforms.RandomAffine(
                degrees=rotation_degrees,           # L'article n'en met pas, mais dispo ici [cite: 222]
                scale=(0.8, 1.2),                   # Zoom in/out jusqu'à 20% [cite: 220]
                shear=[-15, 15]                     # Cisaillement jusqu'à 15° [cite: 220]
            )
        ])
    
    # Etape 3 : Conversion en tenseur et normalisation
    base_transforms.extend([
        transforms.Grayscale(num_output_channels=3), # Duplication du canal pour le CNN [cite: 205, 208]
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(base_transforms)


class ZooCamTestDataset(Dataset):
    """Dataset sur mesure pour le répertoire Test (retourne l'image et son filename)."""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name


def get_dataloaders(train_dir, test_dir, batch_size=64, num_workers=4, rotation_degrees=0):
    """Fonction utilitaire pour instancier rapidement les DataLoaders."""
    train_dataset = datasets.ImageFolder(
        root=train_dir, 
        transform=get_transforms(is_train=True, rotation_degrees=rotation_degrees)
    )

    test_dataset = ZooCamTestDataset(
        img_dir=test_dir, 
        transform=get_transforms(is_train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_dataset.classes

# ==========================================
# BLOC DE TEST AUTONOME
# ==========================================
if __name__ == "__main__":
    print("Mise en place de l'environnement de test temporaire...")
    
    # 1. Création d'une fausse arborescence
    test_base_dir = "./tmp_data"
    tmp_train = os.path.join(test_base_dir, "Train")
    tmp_test = os.path.join(test_base_dir, "Test", "imgs")
    
    os.makedirs(os.path.join(tmp_train, "Calanoida"), exist_ok=True)
    os.makedirs(os.path.join(tmp_train, "Nannosquillidae"), exist_ok=True)
    os.makedirs(tmp_test, exist_ok=True)
    
    # 2. Création de fausses images (rectangulaires pour tester le padding)
    Image.new('L', (300, 100), color=128).save(os.path.join(tmp_train, "Calanoida", "img1.jpg"))
    Image.new('L', (50, 400), color=200).save(os.path.join(tmp_train, "Nannosquillidae", "img2.jpg"))
    Image.new('L', (200, 200), color=50).save(os.path.join(tmp_test, "test1.jpg"))
    
    # 3. Test des DataLoaders (on simule ta config 8Go VRAM avec un petit batch_size)
    print("\nInitialisation des DataLoaders...")
    train_loader, test_loader, classes = get_dataloaders(
        train_dir=tmp_train, 
        test_dir=tmp_test, 
        batch_size=2, 
        num_workers=0, # 0 pour éviter les soucis de multiprocessing sous Windows dans un script rapide
        rotation_degrees=15
    )
    
    print(f"Classes détectées : {classes}")
    
    # 4. Vérification des dimensions
    for images, labels in train_loader:
        print(f"\n[Train Batch] Forme du tenseur d'images : {images.shape} (Attendu : [N, 3, 224, 224])")
        print(f"[Train Batch] Labels : {labels}")
        break # On teste juste le premier batch
        
    for images, filenames in test_loader:
        print(f"\n[Test Batch] Forme du tenseur d'images : {images.shape}")
        print(f"[Test Batch] Fichiers correspondants : {filenames}")
        break
        
    # 5. Nettoyage
    print("\nNettoyage des fichiers temporaires...")
    shutil.rmtree(test_base_dir)
    print("Test terminé avec succès !")