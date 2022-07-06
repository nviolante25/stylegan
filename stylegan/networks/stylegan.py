from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StyleConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        w_dim=512,
        demodulate=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.affine = nn.Linear(w_dim, in_channels, device=device)
        self.demodulate = demodulate

    def forward(self, input, w):
        """Modulated convolutional 2D forward pass without bias

        Args:
            - input: [batch_size, channels_in, height, width]
            - w    : [batch_size, w_dim]

        Modulation:
            - For each feature map in input, the weights self.weight of shape
            [channels_out, channels_in, size, size] are modulated using one style
            vector of shape [1, channels_in, 1, 1]

        """
        style = self.affine(w)
        batch_size, in_channels, height, width = input.shape

        # outputs = []
        # for i in range(batch_size):
        #     weight = self._modulate_weight(style[i])
        #     outputs.append(self._conv_forward(input[i], weight, None))
        # return torch.stack(outputs) + self.bias.view(-1, 1, 1)

        # weight [batch_size, out_channels, in_channels, h, w]
        weight = self.weight.repeat(batch_size, 1, 1, 1, 1) * style.view(
            batch_size, 1, in_channels, 1, 1
        )
        if self.demodulate:
            weight = weight / torch.sqrt(
                torch.sum(weight**2, dim=(-3, -2, -1), keepdim=True) + 1e-8
            )
        self.groups = batch_size
        output = self._conv_forward(
            input.view(1, -1, height, width),
            weight.view(-1, in_channels, *self.kernel_size),
            self.bias.repeat(batch_size, 1).view(-1),
        ).view(batch_size, -1, height, width)
        return output

    def _modulate_weight(self, style):
        # weight [channels_out, channels_in, kernel_size[0], kernel_size[1]]
        weight = style.view(1, -1, 1, 1) * self.weight
        if self.demodulate:
            weight /= torch.sqrt(
                torch.sum(weight**2, dim=(1, 2, 3), keepdim=True) + 1e-8
            )
        return weight


class GeneratorBlock(nn.Module):
    """
    1. Upsample input by a factor of 2
    2. Apply two StyleConvolutions, each followed by a LeakyReLu
    3. Apply one 1x1 StyleConvolution (with no demodulation) to generate RGB
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2.0, mode="bilinear")
        self.convs = nn.ModuleList(
            [
                StyleConv2d(in_channels, out_channels, 3, 1, 1, w_dim=w_dim),
                StyleConv2d(out_channels, out_channels, 3, 1, 1, w_dim=w_dim),
            ]
        )
        self.to_rgb = StyleConv2d(
            out_channels, 3, 1, w_dim=w_dim, demodulate=False
        )

    def forward(self, x, w):
        x = self.upsample(x)
        for conv in self.convs:
            x = F.leaky_relu(conv(x, w), 0.2)
        rgb_image = self.to_rgb(x, w)
        return x, rgb_image


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = [nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, w_dim=512, resolution=1024):
        super().__init__()
        in_channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 256,
            256: 128,
            512: 64,
            1024: 32,
        }
        res = 4
        self.init_const = nn.Parameter(
            torch.zeros((1, in_channels[res], res, res)), requires_grad=True
        )
        nn.init.normal_(self.init_const)

        self.init_conv = StyleConv2d(
            in_channels[res], in_channels[res], 3, 1, 1, w_dim=w_dim
        )
        self.style_blocks = nn.ModuleList()
        while res < resolution:
            next_res = 2 * res
            self.style_blocks.append(
                GeneratorBlock(in_channels[res], in_channels[next_res], w_dim)
            )
            res = next_res

    def forward(self, w):
        assert w.dim() == 2
        batch_size = w.shape[0]
        x = self.init_const.repeat(batch_size, 1, 1, 1)
        x = self.init_conv(x, w)

        rgb_image = torch.zeros((batch_size, 3, 4, 4)).to(w.device)
        for block in self.style_blocks:
            x, intermediate_rgb_image = block(x, w)
            rgb_image = intermediate_rgb_image + F.interpolate(
                rgb_image, scale_factor=2, mode="bilinear"
            )

        return rgb_image


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=0.5, mode="bilinear"),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.convs(x) + self.skip(x)
        return out / math.sqrt(2.0)


class Discriminator(nn.Module):
    def __init__(self, resolution=1024) -> None:
        super().__init__()
        in_channels = {
            1024: 32,
            512: 64,
            256: 128,
            128: 256,
            64: 512,
            32: 512,
            16: 512,
            8: 512,
            4: 512,
        }
        res = resolution
        self.from_rgb = nn.Conv2d(3, in_channels[res], 3, 1, 1)
        blocks = []
        while res > 4:
            prev_res = res // 2
            blocks.append(
                DiscriminatorBlock(in_channels[res], in_channels[prev_res])
            )
            res = prev_res

        self.blocks = nn.Sequential(*blocks)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels[4], in_channels[4], 4, 1, 0),
            nn.LeakyReLU(0.2),
        )
        self.final_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels[4], 1),
        )

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.final_linear(x)
        return x


class StyleGAN2(nn.Module):
    def __init__(
        self,
        z_dim=512,
        w_dim=512,
        mapping_layers=8,
        resolution=1024,
    ) -> None:
        super().__init__()
        self.mapping_network = MappingNetwork(z_dim, w_dim, mapping_layers)
        self.generator = Generator(w_dim, resolution)
        self.discriminator = Discriminator(resolution)


if __name__ == "__main__":
    torch.manual_seed(0)
    mapping_network = MappingNetwork()
    generator = Generator()
    discriminator = Discriminator()
    z = torch.randn((2, 512))
    w = mapping_network(z)
    image = generator(w)
    a = discriminator(image)
    print()
    for name, param in generator.named_parameters():
        if param.requires_grad:
            print(name)

    sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print()
