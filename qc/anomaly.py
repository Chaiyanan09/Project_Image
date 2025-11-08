from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from sklearn.neighbors import NearestNeighbors
import joblib

def _build_backbone(backbone="resnet18", layer="avgpool"):
    if backbone == "resnet18":
        m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
        feat_dim = 512
    else:
        raise ValueError("Unsupported backbone")
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m, feat_dim

def _preprocess_tf():
    return tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_feature(img_bgr: np.ndarray, model, device="cpu") -> np.ndarray:
    img_rgb = img_bgr[:,:,::-1]
    pil = Image.fromarray(img_rgb)
    tf = _preprocess_tf()
    x = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.forward_features(x) if hasattr(model, "forward_features") else model.forward(x)
        # For ResNet, take global avgpool output (before fc)
        if isinstance(feats, torch.Tensor) and feats.ndim == 2 and feats.shape[1] == 1000:
            # extract penultimate features by removing fc: use layer attributes
            backbone = model
            modules = list(backbone.children())[:-1]  # up to avgpool
            body = torch.nn.Sequential(*modules)
            y = body(x)
            y = torch.flatten(y, 1)
        else:
            # Otherwise, attempt to flatten
            y = torch.flatten(feats, 1)
    y = y.cpu().numpy().astype("float32")
    y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
    return y[0]

class AnomalyKNN:
    def __init__(self, k=5):
        self.k = k
        self.nn = None
        self.mean = None
        self.std = None

    def fit(self, feats: np.ndarray):
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(feats)), metric="euclidean")
        self.nn.fit(feats)
        dists, _ = self.nn.kneighbors(feats, n_neighbors=min(self.k, len(feats)))
        self.mean = float(np.mean(dists))
        self.std = float(np.std(dists) + 1e-8)

    def score(self, f: np.ndarray) -> float:
        dists, _ = self.nn.kneighbors([f], n_neighbors=self.nn.n_neighbors)
        return float(np.mean(dists))

    def save(self, path: str):
        joblib.dump({"k": self.k, "nn": self.nn, "mean": self.mean, "std": self.std}, path)

    @staticmethod
    def load(path: str) -> "AnomalyKNN":
        obj = joblib.load(path)
        a = AnomalyKNN(k=obj["k"])
        a.nn = obj["nn"]
        a.mean = obj["mean"]
        a.std = obj["std"]
        return a
