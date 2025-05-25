
# Projet de Classification des Tumeurs Cérébrales

Ce projet a été réalisé dans le cadre du cours de **Computer Vision** à AIMS Sénégal (2024–2025). Il utilise l'apprentissage profond pour classifier automatiquement des images IRM cérébrales en quatre catégories : **gliome**, **méningiome**, **sans tumeur**, et **adénome hypophysaire**.

## Contenu du projet

- Modèles CNN pré-entraînés : **ResNet18 (PyTorch)** et **VGG16 (TensorFlow)**
- Application web avec **Flask** permettant le téléversement et la prédiction d'images
- Ensemble de données d'IRM classifiées
- Visualisation des métriques (accuracy, recall, F1-score)

##  Structure du dataset

- **Entraînement :** 5712 images
- **Test :** 1311 images
- Répartition dans des sous-dossiers par classe

##  Modèles

- **PyTorch - ResNet18 :**
  - Couches convolutives gelées
  - Dernière couche dense à 4 sorties
  - Optimisation avec `Adam`, `CrossEntropyLoss`

- **TensorFlow - VGG16 :**
  - Couches gelées + tête dense personnalisée
  - `Flatten → Dense(256, ReLU) → Dropout → Dense(4, Softmax)`
  - Optimisation avec `Adam`, `categorical_crossentropy`

##  Entraînement

- Résolution des images : **224x224**
- Batch size : **32**
- 20 époques pour chaque modèle

##  Résultats sur le jeu de test

| Modèle              | Accuracy | Recall | F1-score |
|---------------------|----------|--------|----------|
| PyTorch - ResNet18  | 87%      | 84.5%  | 85.0%    |
| TensorFlow - VGG19  | 92%      | 91%    | 89.5%    |

##  Application Web

- Développée en **Flask**
- Upload d'image, sélection du modèle, affichage des résultats
- Encodage des images en base64 pour visualisation
- Calcul des métriques via `scikit-learn`

##  Perspectives

- Déploiement prévu sur **Heroku**, **Render** ou **Google Cloud Platform**
- Amélioration du dataset et affinement des modèles

##  Auteur

**Victor MBUYI BIDIKUCHANCE**  
AIMS Sénégal – Master en Intelligence Artificielle  
📅 Mai 2025
