import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from scipy.stats import gaussian_kde
from tqdm import tqdm


target_resolution = 32


resize_transform = transforms.Compose([
    transforms.Resize(target_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor() 
])

def loadfromdir_and_flatten_images(folder):
    vectors = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(folder, filename)
            img = Image.open(path)
            img_tensor = resize_transform(img)  # [C, H, W]
            img_tensor = img_tensor.permute(1,2,0)
            vector = img_tensor.reshape(-1).numpy()  
            vectors.append(vector)
    return np.stack(vectors, axis=0)

def loadfromnpy(npy_path):
    images_array = np.load(npy_path)
    images_array = images_array.astype(np.float32) / 255.0
    return images_array.reshape(images_array.shape[0], -1)


def compute_svd_top2(matrix):
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[[1, 2], :], mean  # shape: (2, D)

def project_to_2d(matrix, top2_vectors, mean):
    centered = matrix - mean
    return centered @ top2_vectors.T  # shape: (N, 2)

for iter in tqdm(range(1, 10)):

    base_folder = '/cifar_subset_16384'
    other_folder = f'/iter_{iter}_epoch_1000/images.npy'
    save_dir = "./cifar_svd/cifar16384_last-greedyfeature"
    base_data = loadfromdir_and_flatten_images(base_folder)
    other_data = loadfromnpy(other_folder)


    top2_vectors, mean = compute_svd_top2(base_data)


    base_proj = project_to_2d(base_data, top2_vectors, mean)
    other_proj = project_to_2d(other_data, top2_vectors, mean)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(base_proj[:, 0], base_proj[:, 1], label='Base Images', alpha=0.6)
    plt.scatter(other_proj[:, 0], other_proj[:, 1], label='Other Images', alpha=0.6)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Projection of Images onto Top 2 SVD Components')
    plt.legend()
    plt.grid(True)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    file_name = f"{save_dir}/iter0-{iter}_scatter.jpg"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name)
    plt.close()

    if iter==1:
        kde = gaussian_kde(base_proj.T)

        x_min, x_max = base_proj[:, 0].min(), base_proj[:, 0].max()
        y_min, y_max = base_proj[:, 1].min(), base_proj[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])

        zz = kde(grid_coords).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.imshow(zz.T, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='hot', aspect='auto')
        plt.title("KDE Heatmap of Base Image Projections")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.colorbar(label="Density")
        plt.savefig(f"{save_dir}/kde_heatmap.jpg")
        plt.close()