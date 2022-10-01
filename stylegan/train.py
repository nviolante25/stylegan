import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from timm.optim import Lamb
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from networks.stylegan import Generator, Discriminator
from losses import generator_loss, discriminator_loss, r1_loss
from dataset.dataset import ImageDataset, InfiniteSampler, create_output_folder


class Trainer:
    def __init__(
        self,
        data,
        dest,
        batch_size,
        z_dim,
        w_dim,
        num_layers_mapping=2,
        device="cuda",
        seed=0,
    ):
        torch.manual_seed(seed)
        self.outdir = create_output_folder(dest, data)
        self.tf_events = SummaryWriter(log_dir=self.outdir)

        dataset = ImageDataset(data)
        sampler = InfiniteSampler(dataset, seed)
        self.dataloader = iter(DataLoader(dataset, batch_size=batch_size, sampler=sampler))
        print(dataset)

        assert dataset.info.image_shape[1] == dataset.info.image_shape[2]
        resolution = dataset.info.image_shape[1]

        self.generator = Generator(z_dim, w_dim, resolution, num_layers_mapping, device)
        self.discriminator = Discriminator(resolution, device)
        self.G_optimizer = Lamb(self.generator.parameters(), lr=0.0025, betas=(0.0, 0.9))
        self.D_optimizer = Lamb(self.discriminator.parameters(), lr=0.002, betas=(0.0, 0.9))

        self.batch_size = batch_size
        self.r1_regularization_every = 16
        self.r1_gamma = 10.0
        self.device = device

    def fit(self):
        self.total_images = int(100e3)
        self.step = 0

        self.grid_z = self.generator.sample_z(16)
        self.tick = 0
        self.tick_every = 5000

        with tqdm(initial=self.step, total=self.total_images) as pbar:
            while self.step < self.total_images:
                self.discriminator_step()
                self.generator_step()

                if self.do_r1_regularization():
                    self.r1_regularization_step()

                if self.do_save_checkpoint():
                    self.save_checkpoint()

                self.step += self.batch_size
                pbar.update(self.batch_size)

    def do_save_checkpoint(self):
        return self.step // self.tick_every > self.tick

    def do_r1_regularization(self):
        return self.step * self.batch_size % self.r1_regularization_every == 0

    def r1_regularization_step(self):
        self.discriminator.zero_grad()
        real_images = next(self.dataloader).to(self.device)
        real_images.requires_grad = True
        real_logits = self.discriminator(real_images)
        Reg_loss = 0.5 * self.r1_gamma * r1_loss(real_logits, real_images) * self.r1_regularization_every
        Reg_loss.backward()
        self.D_optimizer.step()
        self.tf_events.add_scalar("Loss/R1 regularization", Reg_loss.item(), global_step=self.step)

    def discriminator_step(self):
        self.discriminator.zero_grad()
        real_images = next(self.dataloader).to(self.device)
        z = self.generator.sample_z(self.batch_size)
        fake_images = self.generator(z)
        fake_logits = self.discriminator(fake_images)
        real_logits = self.discriminator(real_images)

        D_loss = discriminator_loss(real_logits, fake_logits)
        D_loss.backward()
        self.D_optimizer.step()
        self.tf_events.add_scalar("Loss/Discriminator", D_loss.item(), global_step=self.step)

    def generator_step(self):
        self.generator.zero_grad()
        z = self.generator.sample_z(self.batch_size)
        fake_images = self.generator(z)
        fake_logits = self.discriminator(fake_images)
        G_loss = generator_loss(fake_logits)
        G_loss.backward()
        self.G_optimizer.step()
        self.tf_events.add_scalar("Loss/Generator", G_loss.item(), global_step=self.step)

    @torch.no_grad()
    def save_images(self):
        display_images = self.generator(self.grid_z)
        display_images = make_grid(display_images, 4)
        display_images = torch.clip(display_images, -1, 1).detach().cpu().numpy().transpose(1, 2, 0)
        display_images = (255 * (0.5 * display_images + 0.5)).astype(np.uint8)
        display_images = Image.fromarray(display_images)
        display_images.save(os.path.join(self.outdir, f"fake_images_{str(self.tick).zfill(6)}.png"))

    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.outdir, f"network_{str(self.tick).zfill(6)}.pth")
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "G_optimizer": self.G_optimizer.state_dict(),
                "D_optimizer": self.D_optimizer.state_dict(),
                "step": self.step,
            },
            checkpoint_path,
        )
        self.save_images()
        self.tick += 1


if __name__ == "__main__":
    trainer = Trainer(
        data="/data/nviolant/data_eg3d/lsun-cars-100k-256x256/",
        dest="./training-runs",
        batch_size=16,
        z_dim=512,
        w_dim=512,
    )

    trainer.fit()
