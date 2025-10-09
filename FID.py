import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)

class NpyDataset(Dataset):
    def __init__(self, npy_path, transform):
        self.images = np.load(npy_path)  # shape: (N, H, W, C)
        # perm = torch.randperm(self.images.shape[0])
        # self.images = self.images[perm[:16384]]
        # breakpoint()
        if self.images.max() > 1:
            self.images = self.images.astype(np.uint8) 
        if self.images.shape[0] > 32768:
            self.images = self.images[:32768]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img)


def get_activations(dataloader, model, device):
    model.eval()
    features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            pred = model(batch)  # InceptionV3 outputs a tuple (features, aux_features)
            features.append(pred.cpu().numpy())
    # breakpoint()
    return np.concatenate(features, axis=0)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("FID calculation produced singular product; adding eps to diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

    
def calculate_fid(path1, path2, batch_size=1024, device='cuda' if torch.cuda.is_available() else 'cpu'):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    if path1.endswith(".npy"):
        dataset1 = NpyDataset(path1, transform)
    else:
        dataset1 = ImageFolderDataset(path1, transform)

    dataset2 = ImageFolderDataset(path2, transform)
    # breakpoint()

    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=False)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=False)

    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()

    act1 = get_activations(loader1, inception, device)
    act2 = get_activations(loader2, inception, device)
    print(f"act1 shape: {act1.shape}, act2 shape: {act2.shape}")
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

if __name__ == "__main__":
    fids = []
    for iter in tqdm(range(1, 10)):
        path1 = f"Your/Path/To/Generated/Images/iter_{iter}/images.npy"
        path2 = "Your/Path/To/Real/Images"
        
        fid = calculate_fid(path1, path2)
        print(f"Calculating FID for iter {iter}: {fid}")
        fids.append(fid)
        # break
    print(f"FID scores for all iterations: {fids}")
    import matplotlib.pyplot as plt
    plt.plot(range(1, 10), fids, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("FID Score")
    plt.title("FID over Iterations")
    plt.grid(True)
    plt.savefig("cifar_32768_UNet48_epoch500_accumulated_filteredfeature_decayrate0.98.png")
    # save it as json
    import json
    with open("cifar_32768_UNet48_epoch500_accumulated_filteredfeature_decayrate0.98.json", "w") as f:
        json.dump(fids, f)
