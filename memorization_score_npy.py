import os
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model
import torch
import json

def mse(imageA, imageB):
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def visualize_results(results, gen_images, train_images, file_name, grid_rows=10, pairs_per_row=5):
    total_pairs = min(len(results), grid_rows * pairs_per_row)
    n_rows = grid_rows * 2 
    n_cols = pairs_per_row  
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))

    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1) 

    for idx in range(total_pairs):
        pair_row = (idx // pairs_per_row) * 2  
        col = idx % pairs_per_row  
        gen_idx, train_idx, distance = results[idx]

        try:
            gen_img = Image.fromarray(gen_images[gen_idx])
        except Exception as e:
            print(f"{gen_idx}: {e}")
            continue
        try:
            train_img = Image.fromarray(train_images[train_idx])
            train_img = train_img.resize(gen_img.size, Image.BILINEAR)
        except Exception as e:
            print(f"{train_idx}: {e}")
            continue

        axs[pair_row][col].imshow(gen_img)
        axs[pair_row][col].set_title('Generated\n' + str(gen_idx))
        axs[pair_row][col].axis('off')

        axs[pair_row + 1][col].imshow(train_img)
        axs[pair_row + 1][col].set_title('Training\n' + str(train_idx) + '\nMSE: {:.4f}'.format(distance))
        axs[pair_row + 1][col].axis('off')

    total_subplots = n_rows * n_cols
    for idx in range(total_pairs, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        axs[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(file_name)

def get_dino_feature(images, image_processor, model):
    if isinstance(images, list):
        images = [torch.tensor(img) if isinstance(img, np.ndarray) else img for img in images]
    else:
        images = [torch.tensor(images)] if isinstance(images, np.ndarray) else [images]
    # breakpoint()
    inputs = image_processor(images[0], return_tensors="pt").to("cuda")
    # breakpoint()
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state
    return features.cpu().numpy()

def main():
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = Dinov2Model.from_pretrained("facebook/dinov2-small").to("cuda")
    mem_distances = []
    for iter in range(1, 12):
        iter_dir = f"/your/path/to/generated/images/iter_{iter}"

        train_images = np.load(os.path.join(iter_dir, "training_dataset.npy"))
        gen_images = np.load(os.path.join(iter_dir, "training_dataset.npy"))
        
        random_index = np.random.permutation(gen_images.shape[0])
        train_images = train_images[random_index]
        gen_images = gen_images[random_index]

        # feature of generated images 
        gen_data = []
        batch_size = 256
        for i in range(0, len(gen_images), batch_size):
            batch_imgs = gen_images[i:min(i+batch_size, len(gen_images))]
            batch_features = get_dino_feature(batch_imgs, image_processor, model) # use features
            for feature in batch_features:
                gen_data.append(feature)

        # feature of training images 
        train_data = []
        if gen_data:
            for i in range(0, len(train_images), batch_size):
                batch_imgs = train_images[i:min(i+batch_size, len(train_images))]
                batch_features = get_dino_feature(batch_imgs, image_processor, model) # use features
                for feature in batch_features:
                    train_data.append(feature)
                if i > 1000:
                    break

        results = []
        ave_min_dis = 0
        # breakpoint()
        for gen_idx, gen_arr in enumerate(gen_data):
            min_distance = None
            best_match = None

            for train_idx, train_arr in enumerate(train_data):
                distance = mse(gen_arr, train_arr)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    best_match = train_idx

            results.append((gen_idx, best_match, min_distance))
            ave_min_dis += min_distance

            if gen_idx + 1 >= 1000:
                break
        file_name = f"./pixel/cifar16384_UNet48_epoch1000_last-filteredfeature_dino/Train_vs_Train/iter{iter}_avedis{ave_min_dis / len(results):.4f}.png"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        visualize_results(results[:50], gen_images, train_images, file_name, grid_rows=10, pairs_per_row=5)
        # Save results to json

        print(f"average min_distance is {ave_min_dis / len(results):.4f}")
        mem_distances.append(ave_min_dis / len(results))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mem_distances) + 1), mem_distances, marker='o', linewidth=2)
    plt.title("Average Min MSE vs Iteration", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Min MSE", fontsize=12)
    plt.grid(True)
    plt.xticks(range(1, len(mem_distances) + 1))
    plt.tight_layout()
    plt.savefig("./pixel/cifar16384_UNet48_epoch1000_last-filteredfeature_dino/Train_vs_Train/average_mse_plot.png")
    
    # Save the average distances to a json file
    with open("./pixel/cifar16384_UNet48_epoch1000_last-filteredfeature_dino/Train_vs_Train/average_distances.json", "w") as f:
        json.dump(mem_distances, f)


if __name__ == "__main__":
    main()
    # test()