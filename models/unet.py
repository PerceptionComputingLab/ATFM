import torch
import torch.nn as nn


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    p: dropout probability
    """

    def __init__(
        self,
        name,
        input_channels,
        num_classes,
        num_filters,
        apply_last_layer=True,
        padding=True,
        p=0,
        batch_norm=False,
    ):
        super(Unet, self).__init__()
        self.name = name
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.apply_last_layer = apply_last_layer

        self.contracting_path = nn.ModuleList()
        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True
            self.contracting_path.append(
                DownConvBlock(
                    input, output, padding, pool=pool, p=p, batch_norm=batch_norm
                )
            )

        self.upsampling_path = nn.ModuleList()
        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(
                UpConvBlock(input, output, padding, p=p, batch_norm=batch_norm)
            )

        if self.apply_last_layer:
            last_layer = []
            last_layer.append(nn.Conv2d(output, 8, kernel_size=1))
            last_layer.append(nn.ReLU(inplace=True))
            last_layer.append(nn.Conv2d(8, 8, kernel_size=1))
            last_layer.append(nn.ReLU(inplace=True))
            last_layer.append(nn.Conv2d(8, num_classes, kernel_size=1))
            self.last_layer = nn.Sequential(*last_layer)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        del blocks

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """

    def __init__(
        self, input_dim, output_dim, padding, pool=True, p=0, batch_norm=False
    ):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            )

        if batch_norm:
            layers.append(nn.BatchNorm2d(input_dim))

        layers.append(nn.Dropout2d(p=p))
        layers.append(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """

    def __init__(
        self, input_dim, output_dim, padding, bilinear=True, p=0, batch_norm=False
    ):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            upconv_layer = []
            upconv_layer.append(
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            )
            self.upconv_layer = nn.Sequential(*upconv_layer)

        self.conv_block = DownConvBlock(
            input_dim, output_dim, padding, pool=False, p=p, batch_norm=batch_norm
        )

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(
                x, mode="bilinear", scale_factor=2, align_corners=True
            )
            # up = nn.functional.interpolate(
            #    x, mode='nearest', scale_factor=2)
        else:
            up = self.upconv_layer(x)

        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


if __name__ == "__main__":
    image = torch.rand(4, 1, 640, 640)

    unet = Unet(
        "test",
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        apply_last_layer=True,
        padding=True,
        p=0,
        batch_norm=False,
    )

    output = unet(image)

    print(output.size())
