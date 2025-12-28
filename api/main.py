from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
import os
import json
from collections import OrderedDict

app = FastAPI(title="Plant Disease Detection API")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ===== Paths =====
MODEL_PATH = os.path.join("models", "best_cnn_model.pt")

# Option 1 (recommandé): lire classes depuis un fichier généré par training
CLASSES_JSON = os.path.join("artifacts", "classes.json")

# Option 2 (fallback): lire classes depuis le dossier train
CLASSES_DIR = os.path.join("data", "raw", "PlantVillageDataset", "train_val_test", "train")


def load_class_names():
    # ✅ Priorité: classes.json (plus stable)
    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or len(classes) == 0:
            raise RuntimeError("classes.json invalide")
        return classes

    # ✅ Fallback: dossier train
    if not os.path.isdir(CLASSES_DIR):
        raise RuntimeError(f"Dossier classes introuvable: {CLASSES_DIR}")

    classes = sorted([
        d for d in os.listdir(CLASSES_DIR)
        if os.path.isdir(os.path.join(CLASSES_DIR, d))
    ])
    if len(classes) == 0:
        raise RuntimeError("Aucune classe trouvée dans le dossier train")
    return classes


CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# ✅ même preprocessing que training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> torch.nn.Module:
    # weights=None car on recharge nos poids
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_model() -> torch.nn.Module:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = build_model(NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location="cpu")

    # ✅ Corrige le cas où le state_dict vient d'un LightningModule (préfixe "model.")
    if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
        new_state = OrderedDict((k.replace("model.", "", 1), v) for k, v in state.items())
        state = new_state

    # Charge les poids
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Si ça reste incohérent, on fail clairement
    if len(missing) > 0 and len(unexpected) > 0:
        raise RuntimeError(
            "State dict incompatible.\n"
            f"Missing keys (extrait): {missing[:5]}\n"
            f"Unexpected keys (extrait): {unexpected[:5]}"
        )

    model.eval()
    return model


# ✅ charge au démarrage
model = load_model()


@app.get("/")
def root():
    return {"status": "ok", "num_classes": NUM_CLASSES}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": NUM_CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # sécurité: types acceptés
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Format non supporté (jpg/png uniquement)")

    image_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Image invalide ou corrompue")

    x = transform(img).unsqueeze(0)  # (1,3,224,224)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence
    }
