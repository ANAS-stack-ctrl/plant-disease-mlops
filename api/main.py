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

# =====================
# App FastAPI
# =====================
app = FastAPI(title="Plant Disease Detection API")

# ✅ Prometheus instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# =====================
# Paths
# =====================
MODEL_PATH = os.path.join("models", "best_cnn_model.pt")
CLASSES_JSON = os.path.join("artifacts", "classes.json")
CLASSES_DIR = os.path.join(
    "data", "raw", "PlantVillageDataset", "train_val_test", "train"
)

# =====================
# Utils
# =====================
def load_class_names():
    # priorité : classes.json
    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or len(classes) == 0:
            raise RuntimeError("classes.json invalide")
        return classes

    # fallback : dossier train
    if not os.path.isdir(CLASSES_DIR):
        raise RuntimeError(f"Dossier classes introuvable: {CLASSES_DIR}")

    classes = sorted([
        d for d in os.listdir(CLASSES_DIR)
        if os.path.isdir(os.path.join(CLASSES_DIR, d))
    ])
    if len(classes) == 0:
        raise RuntimeError("Aucune classe trouvée")
    return classes


CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# =====================
# Preprocessing
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# =====================
# Model
# =====================
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model() -> nn.Module:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = build_model(NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location="cpu")

    # correction Lightning "model."
    if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
        state = OrderedDict(
            (k.replace("model.", "", 1), v) for k, v in state.items()
        )

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


model = load_model()

# =====================
# Routes
# =====================
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
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Format non supporté")

    image_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Image invalide")

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence
    }
