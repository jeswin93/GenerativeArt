from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn


class Nonlinearity(Enum):
    LeakyRelu = 1,


def get_conv(in_channels, out_channels, init=False):
    conv2d = nn.Conv2d(in_channels, out_channels, 5, 2, 2)
    if init:
        nn.init.normal(conv2d.weight, mean=0, std=0.02)
    return conv2d


def get_conv_trans(in_channels, out_channels, init=False):
    conv_trans = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
    if init:
        nn.init.normal(conv_trans.weight, mean=0, std=0.02)
    return conv_trans



def get_linear(in_features, out_features, init=False):
    linear = nn.Linear(in_features, out_features)
    if init:
        nn.init.normal(linear.weight, mean=0, std=0.02)
    return linear


def get_nonlinearity(non_linearity: Nonlinearity):
    if non_linearity == Nonlinearity.LeakyRelu:
        return nn.LeakyReLU(0.02)
    return nn.LeakyReLU(0.02)


def get_batch_norm_2d(batch_norm, feature_count):
    return nn.BatchNorm2d(feature_count) if batch_norm else nn.Identity()


def get_batch_norm_1d(batch_norm, feature_count):
    return nn.BatchNorm1d(feature_count) if batch_norm else nn.Identity()


class GatedNonLinearity(nn.Module):
    def __init__(self):
        super(GatedNonLinearity, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh= nn.Tanh()

    def forward(self, input, condition):
        a = input[:, ::2]
        b = input[:, 1::2]
        c = condition[:, ::2]
        d = condition[:, 1::2]
        if c is not None and d is not None:
            a = a + c
            b = b + d
        return self.sigmoid(a) * self.tanh(b)


class GANGenerator(nn.Module):
    def __init__(self, num_classes, model_dim, batch_normal, latent_dim=128):
        super(GANGenerator, self).__init__()
        self.batch_normal = batch_normal
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.input_layer = get_linear(latent_dim + num_classes, 8 * 4 * 4 * model_dim * 2)
        self.num_classes = num_classes

        self.batch_norm_1 = get_batch_norm_2d(self.batch_normal, 8 * self.model_dim * 2)
        self.conditioning_1 = nn.Linear(num_classes, 8 * 4 * 4 * model_dim * 2)
        self.gated_nonlinearity = GatedNonLinearity()
        self.upsampling_1 = nn.Sequential(
            get_conv_trans(8 * model_dim, 4 * model_dim * 2),
            get_batch_norm_2d(self.batch_normal, 4 * model_dim * 2))

        self.conditioning_2 = get_linear(num_classes, 4 * 8 * 8 * model_dim * 2)
        self.upsampling_2 = nn.Sequential(
            get_conv_trans(4 * model_dim, 2 * model_dim * 2),
            get_batch_norm_2d(self.batch_normal, 2 * model_dim * 2)
        )

        self.conditioning_3 = get_linear(num_classes, 2 * 16 * 16 * model_dim * 2)
        self.upsampling_3 = nn.Sequential(
            get_conv_trans(2 * model_dim, model_dim * 2),
            get_batch_norm_2d(self.batch_normal, model_dim * 2)
        )

        self.conditioning_3_5 = get_linear(num_classes, 32 * 32 * model_dim * 2)
        self.upsampling_3_5 = nn.Sequential(
            get_conv_trans(model_dim, model_dim),
            get_batch_norm_2d(self.batch_normal, model_dim)
        )

        self.conditioning_4 = get_linear(num_classes, 64 * 64 * model_dim)
        self.upsampling_4 = get_conv_trans(model_dim // 2, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, class_vec, noise):
        noise = torch.cat([noise, class_vec], dim=-1)

        input = self.input_layer(noise)
        input = input.reshape(-1, 8 * self.model_dim * 2, 4, 4)

        input = self.batch_norm_1(input)
        conditioning = self.conditioning_1(class_vec).reshape(-1, 8 * self.model_dim * 2, 4, 4)
        input = self.gated_nonlinearity(input, conditioning)
        input = self.upsampling_1(input)

        conditioning = self.conditioning_2(class_vec).reshape(-1, 4 * self.model_dim * 2, 8, 8)
        input = self.gated_nonlinearity(input, conditioning)
        input = self.upsampling_2(input)

        conditioning = self.conditioning_3(class_vec).reshape(-1, 2 * self.model_dim * 2, 16, 16)
        input = self.gated_nonlinearity(input, conditioning)
        input = self.upsampling_3(input)

        conditioning = self.conditioning_3_5(class_vec).reshape(-1, self.model_dim * 2, 32, 32)
        input = self.gated_nonlinearity(input, conditioning)
        input = self.upsampling_3_5(input)

        conditioning = self.conditioning_4(class_vec).reshape(-1, self.model_dim, 64, 64)
        input = self.gated_nonlinearity(input, conditioning)
        input = self.upsampling_4(input)
        output = self.sigmoid(input)
        return output

    def sample_noise(self, batch_size):
        return torch.randn([batch_size, self.latent_dim])

    def gen_class_vec(self, batch_size):
        class_vec = torch.zeros([batch_size, self.num_classes], dtype=torch.float32)
        batch_ind = torch.arange(batch_size)
        clas_ind = torch.randint(0, self.num_classes, [batch_size])
        class_vec[batch_ind, clas_ind] = 1
        return class_vec


class GANDiscriminator(nn.Module):
    def __init__(self, num_class, model_dim, non_linearity: Nonlinearity, batch_norm: bool):
        super(GANDiscriminator, self).__init__()

        self.model_dim = model_dim
        self.non_linearity = non_linearity
        self.batch_norm = batch_norm
        self.conv_layer = nn.Sequential(
            # block 1
            get_conv(3, model_dim),
            get_nonlinearity(non_linearity),
            # block 2
            get_conv(model_dim, 2 * model_dim),
            get_batch_norm_2d(batch_norm, 2 * model_dim),
            get_nonlinearity(non_linearity),
            # block 3
            get_conv(2 * model_dim, 4 * model_dim),
            get_batch_norm_2d(batch_norm, 4 * model_dim),
            get_nonlinearity(non_linearity),
            # block 4
            get_conv(4 * model_dim, 8 * model_dim),
            get_batch_norm_2d(batch_norm, 8 * model_dim),
            # block 5
            get_conv(8 * model_dim, 16 * model_dim),
            get_batch_norm_2d(batch_norm, 16 * model_dim),
        )

        self.source_output = nn.Linear(2 * 4 * 4 * 8 * model_dim, 1)
        self.class_output = nn.Linear(2 * 4 * 4 * 8 * model_dim, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layer(x)
        flatten = nn.Flatten()
        x = flatten(x)
        out_source = self.source_output(x)
        out_class = self.class_output(x)
        out_class = self.sigmoid(out_class)
        out_source = self.sigmoid(out_source)
        return out_source, out_class


if __name__ == '__main__':
    pass
    # disc = GANDiscriminator(10, 64, Nonlinearity.LeakyRelu, True)
    # input = torch.randn([1, 3, 64, 64])
    # disc(input)
    # gen = GANGenerator(10, 64, Nonlinearity.LeakyRelu, True)
    # class_vec = torch.zeros([1, 10])
    # class_vec[0, 0] = 1
    # noise = torch.randn([1, 128])
    # x = gen(class_vec, noise)
    # print(x)
