
import os
import torch.utils.data
from PIL import Image
import numpy as np


class WikiArt(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size):
        subdirectories = os.listdir(img_dir)
        self.num_categories = len(subdirectories)

        self.filenames = []
        for i, subdirectory in enumerate(subdirectories):
            self.filenames += map(
                lambda x: (i, f"{img_dir}/{subdirectory}/{x}"),
                os.listdir(f"{img_dir}/{subdirectory}"),
            )
        self.img_size = img_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        category, img_path = self.filenames[index]
        orig_img = Image.open(img_path).convert('RGB')
        orig_img = orig_img.resize((self.img_size, self.img_size))
        img = np.array(orig_img, dtype=np.float32)
        img = img / 255.0
        img = np.moveaxis(img, -1, 0)  # HWC to CHW

        class_vec = np.zeros(self.num_categories)
        class_vec[category] = 1
        class_vec = torch.from_numpy(class_vec).float()

        return img, class_vec
