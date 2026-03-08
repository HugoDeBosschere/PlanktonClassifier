import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import copy

# Importation de nos modules personnalisés
from dataset import get_transforms, datasets
from model import get_model
from utils import compute_class_weights, evaluate_macro_f1, get_or_create_split

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning de MobileNetV2")
    parser.add_argument('--model_name', type=str, default='mobilenetv2_140')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_decay', type=float, default=0.97)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='./data/train')
    parser.add_argument('--out_dir', type=str, default='./models')
    return parser.parse_args()

def main():
    args = parse_args()
    
    run_name = f"{args.model_name}_ep{args.epochs}_lr{args.lr}_dec{args.lr_decay}_bs{args.batch_size}"
    run_save_dir = os.path.join(args.out_dir, run_name)
    checkpoint_dir = os.path.join(run_save_dir, "checkpoints")
    
    os.makedirs(run_save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(os.path.join(run_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Démarrage du run : {run_name}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. SCAN UNIQUE DU DOSSIER (Sauve +15min de chargement)
    print(f"📂 Scan unique des données depuis {args.data_dir}...")
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=get_transforms(is_train=True))
    
    # 2. SAUVEGARDE DU DICTIONNAIRE DES CLASSES (Pour l'étape 5)
    classes_dict = {i: c for i, c in enumerate(train_dataset.classes)}
    with open(os.path.join(run_save_dir, 'class_names.json'), 'w') as f:
        json.dump(classes_dict, f, indent=4)
    
    # 3. COPIE SUPERFICIELLE POUR LA VALIDATION (Pas de re-scan, juste la transfo change)
    val_dataset = copy.copy(train_dataset)
    val_dataset.transform = get_transforms(is_train=False)
    
    # 4. SPLIT STATIQUE SANS RE-SCANNER
    split_path = os.path.join(args.data_dir, "split_registry.json")
    split_registry = get_or_create_split(train_dataset.targets, split_file=split_path)
    
    train_subset = Subset(train_dataset, split_registry["train"])
    val_subset = Subset(val_dataset, split_registry["val"])

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("⚖️ Calcul des poids de classes...")
    class MockDatasetForWeights:
        def __init__(self, full_targets, subset_indices):
            self.targets = [full_targets[i] for i in subset_indices]
    
    train_targets_mock = MockDatasetForWeights(train_dataset.targets, split_registry["train"])
    class_weights = compute_class_weights(train_targets_mock).to(device)

    model = get_model(num_classes=len(train_dataset.classes), device=device)

    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_macro_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", mininterval=60.0, ascii=True)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_subset)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        val_macro_f1 = evaluate_macro_f1(all_targets, all_preds)
        print(f"📈 Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | LR: {current_lr:.6f}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_macro_f1': val_macro_f1
        }, checkpoint_path)
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model_path = os.path.join(run_save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ Nouveau meilleur modèle sauvegardé (F1: {best_macro_f1:.4f})")

    print("\n✅ Entraînement terminé !")

if __name__ == "__main__":
    main()