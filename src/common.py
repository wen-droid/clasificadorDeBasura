"""Arquitectura propia y preprocesamiento compartido por entrenamiento e interfaz."""
from pathlib import Path
import io
import json
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Baterías', 'Biológico', 'Cartón', 'Ropa', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Zapatos', 'Otros residuos']
SIZE = 96


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def preprocess(image):
    """Misma transformación determinista para validación, prueba e inferencia."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    image = ImageOps.pad(image, (SIZE, SIZE), method=Image.Resampling.BILINEAR,
                         color=(128, 128, 128))
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.)


class ResiduoCNN(nn.Module):
    """Sin torchvision, descargas, transfer learning ni pesos de terceros."""
    def __init__(self, classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AvgPool2d(3))  # entrada 12x12 → 4x4; gradiente determinista en CUDA
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(96*4*4, 64),
                                        nn.ReLU(), nn.Dropout(.4), nn.Linear(64, classes))

    def forward(self, x):
        return self.classifier(self.features(x))  # logits; CrossEntropyLoss aplica log-softmax


def load_model(device='cpu'):
    checkpoint = torch.load(ROOT/'models/mejor_modelo.pt', map_location=device, weights_only=True)
    if checkpoint['classes'] != CLASSES or checkpoint['size'] != SIZE:
        raise ValueError('Clases o preprocesamiento incompatibles con este código.')
    model = ResiduoCNN().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
