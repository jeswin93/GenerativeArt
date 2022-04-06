import torch
from torch.utils.data import DataLoader

from dataset import WikiArt
from gan import GANDiscriminator, Nonlinearity, GANGenerator

# Temp testing code
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
dataset = WikiArt("D:\\Datasets\\WikiArt2\\images", 64)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)

if __name__ == '__main__':
    disc = GANDiscriminator(14, 64, Nonlinearity.LeakyRelu, True).to(DEVICE)
    gen = GANGenerator(14, 64, Nonlinearity.LeakyRelu, True).to(DEVICE)

    for images, class_vecs in loader:
        images = images.to(DEVICE)
        class_vecs = class_vecs.to(DEVICE)

        disc(images)

        noise = torch.randn([32, 128])
        x = gen(class_vecs, noise)
        print(x)
