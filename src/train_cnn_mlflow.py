import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
import mlflow.pytorch

# === CONFIG ===
DATA_ROOT = r"data/raw/PlantVillageDataset/train_val_test"
BATCH_SIZE = 32
NUM_WORKERS = 0          # ✅ Windows safe
IMG_SIZE = 224
MAX_EPOCHS = 4
LEARNING_RATE = 1e-4

EXPERIMENT_NAME = "PlantDiseaseExperiment"
RUN_NAME = "ResNet18_CNN"

MODEL_OUT = os.path.join("models", "best_cnn_model.pt")


def create_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=eval_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_dl, val_dl, test_dl, train_ds.classes


class PlantDiseaseModel(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        self.model = backbone
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    train_dl, val_dl, test_dl, class_names = create_dataloaders()
    num_classes = len(class_names)

    # ✅ MLflow Logger (un seul run)
    mlf_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        tracking_uri="sqlite:///mlflow.db"
    )

    model = PlantDiseaseModel(num_classes=num_classes)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",   # ✅ simple/fiable
        devices=1,
        logger=mlf_logger,
        log_every_n_steps=10,
        enable_checkpointing=False
    )

    trainer.fit(model, train_dl, val_dl)
    test_metrics = trainer.test(model, test_dl)[0]

    # ✅ Log extras dans LE MÊME RUN
    run_id = mlf_logger.run_id
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "lr": LEARNING_RATE,
            "epochs": MAX_EPOCHS,
            "num_classes": num_classes
        })

        # log classes.json (utile pour API)
        classes_path = os.path.join("artifacts", "classes.json")
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(classes_path, artifact_path="metadata")

        # métriques test (si pas déjà dans logger)
        for k, v in test_metrics.items():
            mlflow.log_metric(k, float(v))

        # ✅ log le vrai torch model
        mlflow.pytorch.log_model(model.model, artifact_path="models")

    # ✅ sauvegarde locale (API)
    torch.save(model.model.state_dict(), MODEL_OUT)
    print(f"✅ Model saved: {MODEL_OUT}")
    print("✅ Model + metrics logged to MLflow")


if __name__ == "__main__":
    main()
