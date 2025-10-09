import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

grid_size = 8     
num_iters = 6      
cols = 3
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(12, 12)) 
axes = axes.flatten()  

for iter in range(1, num_iters + 1):
    image_dir = f"Your/Path/To/Generated/Images/iter_{iter}/images.npy"

    if os.path.isdir(image_dir):
        image_files = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])[:64]
        images = [Image.open(f).convert('RGB') for f in image_files]
    elif image_dir.endswith(".npy"):
        images_npy = np.load(image_dir)
        if images_npy.max() > 1:
            images_npy = images_npy.astype(np.uint8)
        images = [Image.fromarray(image_npy) for image_npy in images_npy[:64]]

    img_width, img_height = images[0].size

    grid_img = Image.new('RGB', (img_width * grid_size, img_height * grid_size))
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid_img.paste(img, (col * img_width, row * img_height))

    axes[iter - 1].imshow(grid_img)
    axes[iter - 1].axis('off')
    axes[iter - 1].set_title(f'Iteration {iter}', fontsize=22)

plt.tight_layout()
plt.savefig("cifar_sample32768_last.png", dpi=300)
# plt.show()
