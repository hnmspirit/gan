import os
from os.path import isdir, join
import numpy as np
import torch
import torch.nn.functional as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# param
mb_size = 64
Z_dim = 100

mnist = input_data.read_data_sets('../mnist', one_hot=True)
savedir = 'out/'


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# G
G_W1 = xavier_init(size=[100, 128])
G_b1 = Variable(torch.zeros(128), requires_grad=True)

G_W2 = xavier_init(size=[128, 784])
G_b2 = Variable(torch.zeros(784), requires_grad=True)

theta_G = [G_W1, G_W2, G_b1, G_b2]

# D
D_W1 = xavier_init(size=[784, 128])
D_b1 = Variable(torch.zeros(128), requires_grad=True)

D_W2 = xavier_init(size=[128, 1])
D_b2 = Variable(torch.zeros(1), requires_grad=True)

theta_D = [D_W1, D_W2, D_b1, D_b2]
params = theta_D + theta_G

# GAN
def sample_Z(m, n):
    return Variable(torch.randn(m, n))


def generator(z):
    h = torch.relu(z @ G_W1 + G_b1)
    X = torch.sigmoid(h @ G_W2 + G_b2)
    return X


def discriminator(X):
    h = torch.relu(X @ D_W1 + D_b1)
    y = torch.sigmoid(h @ D_W2 + D_b2)
    return y


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


def plot(samples):
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.1, hspace=0.1)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='pink_r')
    return fig


G_solver = optim.Adam(theta_G, lr=1e-3)
D_solver = optim.Adam(theta_D, lr=1e-3)

# main
ones_label = Variable(torch.ones(mb_size, 1))
zeros_label = Variable(torch.zeros(mb_size, 1))

if not isdir(savedir):
    os.makedirs(savedir)

i = 0
z_vis = sample_Z(25, Z_dim)

for it in range(100000):
    z = sample_Z(mb_size, Z_dim)
    G_sample = generator(z)
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    D_real = discriminator(X)
    D_fake = discriminator(G_sample)

    D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
    D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake
    D_loss.backward()
    D_solver.step()
    reset_grad()

    z = sample_Z(mb_size, Z_dim)
    G_sample = generator(z)
    D_fake = discriminator(G_sample)

    G_loss = nn.binary_cross_entropy(D_fake, ones_label)
    G_loss.backward()
    G_solver.step()
    reset_grad()

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('+ D_loss: {:.4}'.format(D_loss.data.numpy()))
        print('+ G_loss: {:.4}'.format(G_loss.data.numpy()))
        print()

    if it % 1000 == 0:
        samples = generator(z_vis).data.numpy()
        fig = plot(samples)
        plt.savefig(join(savedir, '{:06d}.png'.format(i)), bbox_inches='tight')
        i += 1
        plt.close(fig)
