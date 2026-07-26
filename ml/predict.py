"""
Task 5 integration point.
Loads the certificate-authenticity model (Task 3) once and exposes
verify_certificate(image_bytes) for the Flask app to call when a
coordinator confirms a placement and uploads a certificate photo.

If the model file doesn't exist (train_classifier.py hasn't been run
yet - it needs internet access to download ImageNet weights, which the
assessment sandbox may not have), importing this module raises
ImportError/FileNotFoundError, and app.py catches that and disables the
feature gracefully instead of breaking the page.

Low-confidence rule: if the model isn't clearly sure (< CONFIDENCE_FLOOR),
we report "uncertain" instead of forcing a genuine/tampered label - a
wrong confident-looking label is worse than admitting uncertainty here.
"""
import io
import os

import torch
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "certificate_model.pt")
LABELS = ["genuine", "tampered"]
CONFIDENCE_FLOOR = 0.65  # below this, we refuse to give a verdict

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "certificate_model.pt not found - run ml/train_classifier.py first "
        "(requires internet access to fetch pretrained weights)."
    )

_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    model.to(_device)
    model.eval()
    return model


_model = _load_model()


def verify_certificate(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    conf = float(conf)
    if conf < CONFIDENCE_FLOOR:
        return {"verdict": "uncertain", "confidence": round(conf * 100, 1)}
    return {"verdict": LABELS[int(idx)], "confidence": round(conf * 100, 1)}
