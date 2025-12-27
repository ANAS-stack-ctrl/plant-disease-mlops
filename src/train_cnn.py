import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pytorch_lightning as pl

DATA_ROOT = r"data/raw/PlantVillageDataset/train_val_test"
BATCH_SIZE = 32
NUM_WORKERS = 0     # ✅ Windows
IMG_SIZE = 224
MAX_EPOCHS = 5
LEARNING_RATE = 1e-4

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

    train_dl, val_dl, test_dl, classes = create_dataloaders()
    num_classes = len(classes)

    model = PlantDiseaseModel(num_classes=num_classes)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False
    )

    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl)

    torch.save(model.model.state_dict(), MODEL_OUT)
    print(f"✅ Modèle sauvegardé: {MODEL_OUT}")
    print(f"✅ Classes: {classes}")


if __name__ == "__main__":
    main()
