import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, gammaln
from PIL import Image
import os
import math
from tqdm import tqdm

def knn_entropy(X, k=5):
    # breakpoint()
    n_samples, n_features = X.shape
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    r = distances[:, k]

    log_volume_unit_ball = (n_features / 2) * np.log(np.pi) - gammaln(n_features / 2 + 1)

    entropy = (n_features / n_samples) * np.sum(np.log(r + 1e-10)) + digamma(n_samples) - digamma(k) + log_volume_unit_ball
    # breakpoint()
    return entropy

def load_images_as_vectors(folder_path, resize_to=None):
    vectors = []
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for path in tqdm(image_files, desc=f"Loading {folder_path}"):
        img = Image.open(path).convert('RGB')
        if resize_to is not None:
            img = img.resize(resize_to)
        img_array = np.array(img)

        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0.0, 1.0)
            img_array = (img_array * 255).astype(np.uint8)

        img_vector = img_array.flatten()
        vectors.append(img_vector)

    vectors = np.stack(vectors, axis=0)
    return vectors

def compute_entropy_from_folder(folder_path, k=1, resize_to=None):
    if folder_path.endswith('.npy'):
        samples = np.load(folder_path)
        samples = samples.reshape(samples.shape[0], -1)
    elif os.path.isdir(folder_path):
        samples = load_images_as_vectors(folder_path, resize_to=resize_to)
    else:
        raise ValueError("Wrong folder path or file format. Please provide a directory or a .npy file.")
    samples = samples.astype(np.float32)
    entropy = knn_entropy(samples, k=k)
    return entropy

from tqdm import tqdm
import matplotlib.pyplot as plt
import json

entropys = []
datasizes = [1024 * (2 ** i) for i in range(6)]  # 1024, 2048, 4096, 8192, 16384, 32768

for iteration in tqdm(range(1, 16)):
    folder_path = f"Your/Path/To/Generated/Images/iter_{iteration}/images.npy"
    entropy = compute_entropy_from_folder(folder_path, k=1, resize_to=None)
    entropys.append(entropy)
    print(f"Iteration {iteration}, Estimated entropy: {entropy:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(range(1, 16), entropys, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.title('Entropy vs Iteration, K=1')
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar_sample2048_UNet48_epoch8000_last_k=1.png")
plt.close()


with open("cifar_sample2048_UNet48_epoch8000_last_k=1.json", "w") as f:
    json.dump(entropys, f)


