

import torch.utils.data

class WikiArt(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size):
        pass

    def __len__(self):
        return 0

    def __geitem__(self, index):
        image = []
        class_vec = []
        return image, class_vec