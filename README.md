# 🌿 Pipeline MLOps – Détection de maladies des plantes (PlantVillage)

Projet : Pipeline MLOps pour la détection automatique des maladies des plantes à l’aide de Deep Learning et de Cloud Computing  
Encadrant : Dr. Anass Deroussi  
Année universitaire : 2025/2026

## 🎯 Objectif
Construire une chaîne MLOps complète (DataOps → ModelOps → DeploymentOps) pour :
- entraîner un modèle de classification d’images de feuilles (saines / malades),
- tracer les expériences avec MLflow (paramètres, métriques, artefacts),
- déployer une API d’inférence (FastAPI),
- fournir une interface utilisateur interactive (Streamlit).

## 🧱 Architecture (implémentée)
Data (PlantVillage) → Training (PyTorch Lightning) → Tracking (MLflow) → Inference API (FastAPI) → UI (Streamlit)

## 📦 Dataset
- Source : PlantVillage (Kaggle)
- Structure utilisée : `train / val / test`
- Classes : healthy + maladies (tomate, poivron, pomme de terre…)

Chemin attendu :
data/raw/PlantVillageDataset/train_val_test/
train/
val/
test/

## 🛠️ Stack technique
- Python 3.11
- PyTorch + TorchVision
- PyTorch Lightning
- MLflow
- FastAPI + Uvicorn
- Streamlit
- (à venir dans la suite) Docker, Kubernetes, Monitoring

## ✅ Résultats (exemple)
- ResNet18 fine-tuning
- test_acc ≈ 0.98–0.99 (selon run)
- inférence locale < 2s

## 📁 Structure du projet
plant-disease-mlops/
api/
main.py
src/
train_cnn.py
train_cnn_mlflow.py
notebooks/
01_exploration.ipynb
models/
best_cnn_model.pt
artifacts/
classes.json
ui_app.py
mlflow.db
requirements.txt

## 🚀 Installation
Créer et activer un environnement virtuel :

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Entraîner + tracker avec MLflow

Lancer l’entraînement (log params + metrics + model) :
.\.venv\Scripts\python.exe src\train_cnn_mlflow.py

Lancer MLflow UI :
mlflow ui --backend-store-uri sqlite:///mlflow.db


Ouvrir : http://127.0.0.1:5000

Lancer l’API d’inférence (FastAPI):
uvicorn api.main:app --reload

Docs Swagger : http://127.0.0.1:8000/docs

Endpoint principal :

POST /predict (upload image)

🖥️ Lancer l’interface interactive (Streamlit)

Dans un 2e terminal :
streamlit run ui_app.py

Ouvrir : http://localhost:8501

🧪 Tester

Ouvrir l’UI Streamlit

Uploader une image depuis :
data/raw/PlantVillageDataset/train_val_test/test/<class_name>/

Cliquer sur Predict

Vérifier la classe + confidence

📌 Auteurs:
Étudiant : Hahou Anas

Groupe : 5iir6


---