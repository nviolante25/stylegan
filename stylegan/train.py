import torch
from timm.optim import Lamb


from networks.stylegan import MappingNetwork, Generator, Discriminator
from losses import generator_loss, discriminator_loss, r1_loss
from dataset.dataset import ImageDataset, Transform, InfiniteSampler
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader


seed=0
torch.manual_seed(0)
z_dim=512
w_dim=512
resolution=128
num_layers=8
batch_size = 4
r1_regularization_interval=16 #mini-batches
r1_gamma = 5.0

mapping_network = MappingNetwork(z_dim, w_dim, num_layers)
generator = Generator(w_dim, resolution)
discriminator = Discriminator(resolution)

transform = Transform(Compose([ToTensor(), Resize((128, 128))]))
dataset = ImageDataset('/home/nviolante/datasets/coco/test2017', transform)
sampler = InfiniteSampler(dataset, seed)
dataloader = iter(DataLoader(dataset, batch_size=4, sampler=sampler))

M_optimizer = Lamb(mapping_network.parameters(), lr=2e-4, betas=(0.9, 0.9))
G_optimizer = Lamb(generator.parameters(), lr=2e-3, betas=(0.9, 0.9))
D_optimizer = Lamb(discriminator.parameters(), lr=2e-3, betas=(0.9, 0.9))



total_images=100e3
num_images=0
num_batches=0


while True:
    # train discriminator
    discriminator.zero_grad()
    z = torch.randn((batch_size, z_dim))
    real_images = next(dataloader)
    num_images += batch_size
    num_batches += 1

    with torch.no_grad():
        w = mapping_network(z)
        fake_images = generator(w)
    fake_logits = discriminator(fake_images) 
    real_logits = discriminator(real_images)

    D_loss = discriminator_loss(real_logits, fake_logits)
    D_loss.backward()
    D_optimizer.step()
    print('D_loss', D_loss.item())

    if (num_batches % r1_regularization_interval == 0):
        discriminator.zero_grad()
        real_images = next(dataloader)
        real_images.requires_grad = True
        real_logits = discriminator(real_images)
        Reg_loss = 0.5 * r1_gamma * r1_loss(real_logits, real_images) * r1_regularization_interval
        Reg_loss.backward()
        D_optimizer.step()

    # Train Generator
    generator.zero_grad()
    mapping_network.zero_grad()
    z = torch.randn((batch_size, z_dim))
    w = mapping_network(z)
    fake_images = generator(w)


    fake_logits = discriminator(fake_images)
    G_loss = generator_loss(fake_logits)
    G_loss.backward()
    G_optimizer.step()
    M_optimizer.step()
    print('G_loss', G_loss.item())

    done_training = (num_images >= total_images)
    if done_training:
        break

