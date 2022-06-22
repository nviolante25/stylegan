import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv2d(nn.Conv2d):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None
        ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
    
    def forward(self, input, style=None):
        """Convolutional 2D forward pass without bias
        """
        style = torch.ones((1, 256, 1, 1))
        weight = (style * self.weight)
        weight /= torch.sqrt(torch.sum(weight ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8)

        return self._conv_forward(input, weight, None)

    def apply_bias(self, input):
        return input + self.bias[:, None, None]


class StyleGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Parameter(torch.zeros((1, 512, 4, 4)), requires_grad=True)
        nn.init.kaiming_uniform_(self.c1)
        self.conv1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.convs = nn.ModuleList([
            MyConv2d(256, 128, 3, 1, 1),
            MyConv2d(128, 64, 3, 1, 1),
        ]) 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        

    def forward(self):
        x = self.c1
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)

        for conv in self.convs:
            x = self.upsample(x)
            x = conv(x)
            x = conv.apply_bias(x)
            x = F.leaky_relu(x, 0.2)
        return x

class StyleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class StyleGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    g = StyleGenerator()
    image = g()
    print()
    for name, param in g.named_parameters():
        if param.requires_grad:
            print(name)

    print()