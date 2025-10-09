

import torch
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_iters = 15
trace_list = []

for i in tqdm(range(1, n_iters), desc="Iterations"):
    path = f"Your/Path/To/Generated/Images/iter_{i}/images.npy"

    if path.endswith(".npy"):
        X = np.load(path).astype('float32')
        X = torch.tensor(X)
        X = X.reshape(X.shape[0], -1)
    elif os.path.isdir(path):
        img_files = glob.glob(os.path.join(path, "*.png"))
        if not img_files:
            print(f"[warn] no images found in {path}")
            continue

        # ---------- load & flatten images as Tensor ----------
        imgs = [
            torch.tensor(
                np.array(Image.open(fp).convert("RGB"), dtype=np.float32)
            ).view(-1)  # flatten
            for fp in img_files
        ]
        X = torch.stack(imgs, dim=0).to(device)  # shape (N, D)

    # ---------- Covariance in original space ----------
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean  # shape (N, D)

    cov = (X_centered.t() @ X_centered) / (X_centered.shape[0] - 1)  # shape (D, D)

    # Total variance (trace of covariance)
    total_var = torch.trace(cov).item()
    trace_list.append(total_var)

# ---------- plot those metrics ----------
iterations = list(range(1, n_iters))

plt.figure(figsize=(8, 5))
# trace_list = [a / max(trace_list) for a in trace_list]
# avg_far_list = [a / max(avg_far_list) for a in avg_far_list]
# breakpoint()
plt.plot(iterations, trace_list, marker='o', label='Trace of Covariance')
# plt.plot(iterations, avg_far_list, marker='s', label='Avg Farthest Distance (after SVD Projection)')
# plt.yscale('log')  # Uncomment if you prefer log scale
plt.xlabel("Iteration")
plt.ylabel("Trace of Covariance")
plt.title("Trace of Covariance")
plt.grid(True, which="both", linestyle="--")
# plt.legend()
plt.tight_layout()
plt.savefig("cifar_sample2048_UNet48_epoch8000_last.png")
import json
with open('cifar_sample2048_UNet48_epoch8000_last.json', 'w') as f:
    json.dump(trace_list, f)
