import torch
import torch.nn as nn
import timm

class PlanktonMobileNet(nn.Module):
    def __init__(self, num_classes=86):
        super(PlanktonMobileNet, self).__init__()
        
        # 1. Extracteur de caractéristiques (MobileNetV2 avec width multiplier 1.4)
        # num_classes=0 retire la tête de classification d'origine.
        self.feature_extractor = timm.create_model('mobilenetv2_140', pretrained=True, num_classes=0)
        
        # Récupération dynamique de la dimension du vecteur
        in_features = self.feature_extractor.num_features # 1792
        
        # 2. Classifieur hybride (Auteur + BatchNorm)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),                  # Premier Dropout de l'autrice
            nn.Linear(in_features, 600),        # Transformation affine
            nn.BatchNorm1d(600),                # <-- AJOUT : Normalisation des pré-activations
            nn.ReLU(inplace=True),              # Activation non-linéaire
            nn.Dropout(p=0.2),                  # Second Dropout de l'autrice
            nn.Linear(600, num_classes)         # Couche de classification finale
        )

    def forward(self, x, extract_features=False):
        """
        Passe avant du réseau.
        Si extract_features=True, renvoie le vecteur profond de dimension 1792.
        """
        # Extraction des features : [Batch, 3, 224, 224] -> [Batch, 1792]
        deep_features = self.feature_extractor(x)
        
        if extract_features:
            return deep_features
            
        # Passage dans le classifieur final : [Batch, 1792] -> [Batch, num_classes]
        logits = self.classifier(deep_features)
        return logits


def get_model(num_classes, device='cpu'):
    """
    Instancie le modèle et le place sur le périphérique cible (CPU ou GPU).
    """
    model = PlanktonMobileNet(num_classes=num_classes)
    return model.to(device)


# ==========================================
# BLOC DE TEST AUTONOME
# ==========================================
if __name__ == "__main__":
    print("Initialisation du modèle de test avec timm (mobilenetv2_140) et BatchNorm1d...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Périphérique utilisé : {device}")
    
    # Création du modèle pour les 86 classes
    model = get_model(num_classes=86, device=device)
    
    # On simule un batch : l'argument 2 représente la taille du batch, nécessaire pour la BatchNorm1d
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    # Test 1 : Entraînement complet
    print("\n--- Test du Forward Pass (Entraînement) ---")
    logits = model(dummy_input)
    print(f"Forme des logits : {logits.shape} -> Attendu : [2, 86]")
    
    # Test 2 : Extraction
    print("\n--- Test de l'extraction des Deep Features ---")
    features = model(dummy_input, extract_features=True)
    print(f"Forme des features : {features.shape} -> Attendu : [2, 1792]")
    
    # Vérification du décompte des paramètres mis à jour
    # La BatchNorm1d ajoute 1200 paramètres (600 pour gamma, 600 pour beta)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNombre total de paramètres : {total_params:,}")