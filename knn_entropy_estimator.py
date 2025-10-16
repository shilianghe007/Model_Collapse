import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, gammaln
import json

# ====== DINOv2 init ======
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
processor = AutoProcessor.from_pretrained("facebook/dinov2-small")
model.eval()

# ====== Dataset from npy (images) ======
class NPYPixelDataset(Dataset):
    """Return PIL.Image so the DINOv2 processor can handle it."""
    def __init__(self, npy_path):
        self.data = np.load(npy_path)  # [N, H, W, C] or [N, C, H, W]
        if self.data.ndim != 4:
            raise ValueError(f"Expected 4D array, got shape {self.data.shape}")
        # Ensure [N, H, W, C]
        if self.data.shape[-1] != 3 and self.data.shape[1] == 3:
            self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.data = np.clip(self.data, 0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        return Image.fromarray(self.data[idx])

def _load_npy_as_flat_pixels(npy_path):
    """
    Load npy and return flattened pixel vectors in [0,1] float32 of shape [N, D].
    Accepts [N, H, W, C] or [N, C, H, W].
    """
    arr = np.load(npy_path)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {arr.shape}")
    # To [N, H, W, C]
    if arr.shape[-1] != 3 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    arr = np.clip(arr, 0, 255).astype(np.float32)
    N = arr.shape[0]
    flat = arr.reshape(N, -1)  # [N, D]
    return flat
 
# ====== DINOv2 feature extraction ======
def extract_dinov2_features_from_npy(npy_path, batch_size=64, num_workers=4):
    dataset = NPYPixelDataset(npy_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=lambda x: x  # list[PIL.Image]
    )

    all_features = []
    with torch.no_grad():
        for imgs in tqdm(dataloader, desc=f"DINOv2 extracting {npy_path}"):
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]  # CLS token
            all_features.append(feats.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    return all_features  # [N, 384] for dinov2-small

# ====== kNN entropy (Kozachenko–Leonenko) ======
def knn_entropy(X, k=5):
    """
    KL estimator for differential entropy under Euclidean metric.
    X: [N, d] array
    """
    n_samples, n_features = X.shape
    # Fit kNN (k+1 because the first neighbor is the point itself)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)
    r = distances[:, k]  # distance to k-th neighbor

    # Volume of unit ball in R^d: c_d = pi^{d/2} / Gamma(d/2 + 1)
    log_c_d = (n_features / 2.0) * np.log(np.pi) - gammaln(n_features / 2.0 + 1.0)
    # KL estimator:
    # H ≈ ψ(n) - ψ(k) + log c_d + (d/n) * sum(log r_k)
    entropy = digamma(n_samples) - digamma(k) + log_c_d + (n_features / n_samples) * np.sum(np.log(r + 1e-10))
    return entropy

# ====== Unified API: choose feature space ======
def compute_entropy_from_npy(
    npy_path,
    k=1,
    space="dino",          # "dino" or "pixel"
    batch_size=64,
    num_workers=4
):
    """
    Compute entropy in either DINO feature space ("dino") or raw pixel space ("pixel").
    """
    if space == "dino":
        feats = extract_dinov2_features_from_npy(npy_path, batch_size, num_workers)
    elif space == "pixel":
        feats = _load_npy_as_flat_pixels(npy_path)
    else:
        raise ValueError(f'Unknown space="{space}". Use "dino" or "pixel".')
    return knn_entropy(feats, k=k)

# ====== Backward-compat shim (keeps your original function name) ======
def compute_entropy_dino_from_npy(npy_path, k=1, batch_size=64, num_workers=4):
    return compute_entropy_from_npy(npy_path, k=k, space="dino", batch_size=batch_size, num_workers=num_workers)

# ====== Example ======
if __name__ == "__main__":
    entro_list = []
    for iter in range(1, 10):
        folder_path = "Your_Path/images.npy"

        # DINO space
        # entropy_dino = compute_entropy_from_npy(folder_path, k=1, space="dino", batch_size=64, num_workers=8)
        # print(f"[iter {iter}] Entropy (DINOv2 CLS): {entropy_dino:.4f}")

        # Pixel space
        entropy_pixel = compute_entropy_from_npy(folder_path, k=1, space="pixel")
        print(f"[iter {iter}] Entropy (Pixel space): {entropy_pixel:.4f}")
        entro_list.append(entropy_pixel)
    
    with open("results.json", "w") as f:
        json.dump(entro_list, f)
    # print(f"[done] results saved to {out_dir}")