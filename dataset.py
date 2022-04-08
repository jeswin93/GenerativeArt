
from ast import Raise
import os
import torch.utils.data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class WikiArt(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        subdirectories = os.listdir(img_dir)
        self.num_categories = len(subdirectories)

        self.filenames = []
        for i, subdirectory in enumerate(subdirectories):
            self.filenames += map(
                lambda x: (i, f"{img_dir}/{subdirectory}/{x}"),
                os.listdir(f"{img_dir}/{subdirectory}"),
            )
        self.to_tensor = transforms.ToTensor()


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        category, img_path = self.filenames[index]
        try:
            orig_img = Image.open(img_path).convert('RGB')
            img = self.to_tensor(orig_img)
            class_vec = np.zeros(self.num_categories)
            class_vec[category] = 1
            class_vec = torch.from_numpy(class_vec).float()
        except Exception as e:
            print(img_path)
            raise Exception('some error')

        return img, class_vec


def gen_hardcoded_noise_class(batch_size, class_num=14, latent_dim = 128):
    noise = torch.randn([batch_size, latent_dim])
    classes = torch.zeros([batch_size, class_num])
    batch_ind = torch.arange(batch_size)
    class_ind = torch.randint(0, class_num, [batch_size])
    classes[batch_ind, class_ind] = 1
    print(noise.shape)
    print(classes.shape)
    np.save(f'hardcoded_class_{batch_size}.npy', classes)
    np.save(f'hardcoded_noise_{batch_size}.npy', noise)

def get_class_num(data_dir = 'wikiart_res_64'):
    return len(os.listdir(data_dir))

if __name__ == '__main__':
    gen_hardcoded_noise_class(8*8, get_class_num())
