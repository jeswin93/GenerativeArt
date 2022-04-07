import imageio
import numpy as np
import torch
import torch.functional as F
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid
from tqdm import tqdm

from gan import *


class Trainer:
    def __init__(self, generator: GANGenerator, discriminator: GANDiscriminator, gen_optim, disc_optim,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 device='cuda'):
        self.generator: GANGenerator = generator
        self.gen_optim = gen_optim
        self.discriminator = discriminator
        self.disc_optim = disc_optim
        self.losses = {'G': [], 'D_real': [], 'D_fake': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.device = device
        self.gp_weight = gp_weight
        self.disc_iterations = critic_iterations
        self.print_every = print_every

        self.generator.to(device)
        self.discriminator.to(device)

    def disc_train_step(self, real_imgs, real_classes):
        self.disc_optim.zero_grad()
        batch_size = real_imgs.size()[0]
        fake_imgs, fake_classes = self.sample_generator(batch_size)

        real_imgs = real_imgs.to(self.device)
        real_classes = real_classes.to(self.device)

        disc_source_real, disc_class_real = self.discriminator(real_imgs)
        disc_source_real_loss = F.binary_cross_entropy(disc_source_real,
                                                       torch.ones_like(disc_source_real, dtype=torch.float32).to(
                                                           self.device))
        disc_class_real_loss = F.binary_cross_entropy(disc_class_real, real_classes)

        disc_real_loss = disc_source_real_loss + disc_class_real_loss
        disc_real_loss.backward()
        disc_source_fake, disc_class_fake = self.discriminator(fake_imgs)
        disc_source_fake_loss = F.binary_cross_entropy(disc_source_fake,
                                                       torch.zeros_like(disc_source_fake, dtype=torch.float32).to(
                                                           self.device))
        disc_class_fake_loss = F.binary_cross_entropy(disc_class_fake, fake_classes)

        gradient_penalty = self.gradient_penalty(real_imgs, fake_imgs)
        self.losses['GP'].append(gradient_penalty.data)

        disc_fake_loss = disc_class_fake_loss + disc_source_fake_loss + gradient_penalty
        disc_fake_loss.backward()

        self.disc_optim.step()

        self.losses['D_real'].append(disc_real_loss.data)
        self.losses['D_fake'].append(disc_fake_loss.data)

    def gen_train_step(self, data):
        """ """
        self.gen_optim.zero_grad()

        batch_size = data.size()[0]
        fake_image, fake_classes = self.sample_generator(batch_size)

        disc_source_fake, disc_class_fake = self.discriminator(fake_image)
        gen_source_loss = F.binary_cross_entropy(disc_source_fake, torch.ones_like(disc_source_fake, dtype=torch.float32))
        gen_class_loss = F.binary_cross_entropy(disc_class_fake, fake_classes)
        gen_loss = gen_source_loss + gen_class_loss
        gen_loss.backward()
        self.gen_optim.step()

        self.losses['G'].append(gen_loss.data)

    def gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = interpolated.to(self.device)
        interpolated.requires_grad_(True)
        prob_interpolated, _ = self.discriminator(interpolated)

        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, progress):
        for i, data in enumerate(data_loader):
            images, class_vecs = data
            self.num_steps += 1
            self.generator.eval()
            self.discriminator.train()
            self.disc_train_step(images, class_vecs)
            if self.num_steps % self.disc_iterations == 0:
                self.discriminator.eval()
                self.generator.train()
                self.gen_train_step(images)
            progress.set_description(f'{i+1}/{len(data_loader)}')

            # if i+1 % self.print_every == 0:
            #     print("Iteration {}".format(i + 1))
            #     print("D: {}".format(self.losses['D_real'][-1]))
            #     print("GP: {}".format(self.losses['GP'][-1]))
            #     print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
            #     if self.num_steps > self.disc_iterations:
            #         print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):

        training_progress_images = []
        with tqdm(range(epochs)) as progress:
            for epoch in progress:
                self._train_epoch(data_loader, progress)

                if save_training_gif:
                    sample_count = 8*8
                    fixed_latents = torch.from_numpy(np.load(f'hardcoded_noise_{sample_count}.npy')).to(self.device)
                    fixed_classes = torch.from_numpy(np.load(f'hardcoded_class_{sample_count}.npy')).to(self.device)
                    data = self.generator(fixed_classes, fixed_latents).cpu().data
                    img_grid = make_grid(data)
                    img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                    imageio.imwrite(f'output/epoch_{epoch+1}.png', img_grid)
                    training_progress_images.append(img_grid)

                # Save models
                if (epoch) % 10 == 0:
                    torch.save(self.generator.state_dict(), f'./gen_{epoch+1}.pt')
                    torch.save(self.discriminator.state_dict(), f'./dis_{epoch+1}.pt')

            if save_training_gif:
                imageio.mimsave(f'output/training_{epochs}_epochs.gif',
                                training_progress_images)

    def sample_generator(self, batch_size):
        sample_noise = self.generator.sample_noise(batch_size)
        class_vec = self.generator.gen_class_vec(batch_size)
        sample_noise = sample_noise.to(self.device)
        class_vec = class_vec.to(self.device)
        generated_data = self.generator(class_vec, sample_noise)
        return generated_data, class_vec

