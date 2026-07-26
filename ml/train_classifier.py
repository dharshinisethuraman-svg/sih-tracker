"""
Task 3 - Build the Classifier
===============================
Fine-tunes MobileNetV2 (small, pretrained on ImageNet) to classify a
certificate image as "genuine" or "tampered".

Run this LOCALLY (not inside the assessment sandbox) since it needs
internet access to download the pretrained ImageNet weights the first
time:

    pip install torch torchvision pillow
    python ml/generate_images.py      # if not already run
    python ml/train_classifier.py

Split policy - read this before changing anything:
    The manifest records which of the 8 templates each image came from.
    We split by TEMPLATE, not by individual file: templates 0-5 -> train,
    templates 6-7 -> test. This guarantees no template (and therefore no
    near-duplicate image) appears on both sides. A random image-level
    split would leak near-identical certificates into both sets and the
    reported test accuracy would be meaningless (see Task 3 note in the
    assessment brief).
"""
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "images")
MANIFEST = os.path.join(BASE, "manifest.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "certificate_model.pt")

LABELS = ["genuine", "tampered"]
TEST_TEMPLATES = {6, 7}  # held-out templates -> test set

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CertDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label, _ = self.rows[idx]
        img = Image.open(os.path.join(BASE, path)).convert("RGB")
        return transform(img), LABELS.index(label)


def load_manifest():
    with open(MANIFEST) as f:
        rows = [(r["path"], r["label"], int(r["template_id"]))
                for r in csv.DictReader(f)]
    train = [r for r in rows if r[2] not in TEST_TEMPLATES]
    test = [r for r in rows if r[2] in TEST_TEMPLATES]
    return train, test


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad = False  # freeze the pretrained backbone
    model.classifier[1] = nn.Linear(model.last_channel, 2)  # fine-tune head only
    return model


def main():
    train_rows, test_rows = load_manifest()
    print(f"train images: {len(train_rows)}  (templates "
          f"{sorted(set(r[2] for r in train_rows))})")
    print(f"test images:  {len(test_rows)}  (templates "
          f"{sorted(set(r[2] for r in test_rows))})")

    train_loader = DataLoader(CertDataset(train_rows), batch_size=8, shuffle=True)
    test_loader = DataLoader(CertDataset(test_rows), batch_size=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 6
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"epoch {epoch+1}/{EPOCHS}  train_loss={total_loss/len(train_rows):.4f}")

    # --- evaluate on held-out templates ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"TEST accuracy on held-out templates {sorted(TEST_TEMPLATES)}: "
          f"{correct}/{total} = {100*correct/total:.1f}%")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
