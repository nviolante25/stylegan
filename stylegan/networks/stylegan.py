import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedConv2d(nn.Conv2d):
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

    def forward(self, input, w):
        """Modulated convolutional 2D forward pass without bias

        Args:
            - input: [batch_size, channels_in, height, width]
            - w    : [batch_size, channels_in, 1, 1]

        Modulation:
            - For each feature map in input, the weights self.weight of shape
            [channels_out, channels_in, size, size] are modulated using one style
            vector of shape [1, channels_in, 1, 1]

        """
        style = self.affine(w)
        batch_size = style.shape[0]
        outputs = []
        for i in range(batch_size):
            weight = self._modulate_weight(style[i])
            outputs.append(self._conv_forward(input[i], weight, None))
        return torch.stack(outputs) + self.bias.view(-1, 1, 1)

    def _modulate_weight(self, style):
        weight = style.view(1, -1, 1, 1) * self.weight
        weight /= torch.sqrt(
            torch.sum(weight**2, dim=(1, 2, 3), keepdim=True) + 1e-8
        )
        return weight


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self,
        w_dim=512
    ):
        super().__init__()
        self.init_const = nn.Parameter(
            torch.zeros((1, 512, 4, 4)), requires_grad=True
        )
        nn.init.kaiming_uniform_(self.init_const)
        self.init_conv = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv_layers = nn.ModuleList(
            [
                ModulatedConv2d(256, 128, 3, 1, 1, w_dim=w_dim),
                ModulatedConv2d(128, 64, 3, 1, 1, w_dim=w_dim),
            ]
        )
        self.to_rgb_layers = nn.ModuleList(
            [
                nn.Conv2d(128, 3, 1),
                nn.Conv2d(64, 3, 1)
            ]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, w):
        assert w.dim() == 2
        batch_size = w.shape[0]
        x = self.init_const
        x = self.init_conv(x)
        x = F.leaky_relu(x, 0.2).repeat(batch_size, 1, 1, 1)

        rgb_image = torch.zeros((batch_size, 3, 4, 4)).to(x.device)
        for i, conv in enumerate(self.conv_layers):
            x = self.upsample(x)
            x = conv(x, w)
            x = F.leaky_relu(x, 0.2)
            rgb_image = self.upsample(rgb_image) + F.leaky_relu(self.to_rgb_layers[i](x), 0.2)
        return x


class StyleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class StyleGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    torch.manual_seed(0)
    mapping_network = MappingNetwork().cuda()
    generator = Generator().cuda()
    z = torch.randn((7, 512)).cuda()
    w = mapping_network(z)
    image = generator(w)
    print()
    for name, param in generator.named_parameters():
        if param.requires_grad:
            print(name)

    print()
