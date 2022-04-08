import torch
import torch.optim as optim
from gan import GANGenerator, GANDiscriminator, Nonlinearity
from train import Trainer
from dataset import WikiArt, get_class_num
from torch.utils.data.dataloader import DataLoader

batch_size = 100
model_dim = 128
num_classes = get_class_num()
latent_dim = 128


if __name__ == '__main__':

    img_size = (32, 32, 1)

    generator = GANGenerator(num_classes, model_dim, True, latent_dim)
    discriminator = GANDiscriminator(num_classes, model_dim, Nonlinearity.LeakyRelu, True)

    lr = 1e-4
    betas = (.5, .999)
    gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    disc_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    epochs = 200
    trainer = Trainer(generator, discriminator, gen_optim, disc_optim)
    dataset = WikiArt('D:\\Datasets\\wikiartsmall128')
    data_loader = DataLoader(dataset, batch_size = batch_size)
    trainer.train(data_loader, epochs, save_training_gif=True)

    


