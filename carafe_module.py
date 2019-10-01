import torch.nn as nn

from carafe_layer import CarafeLayer


class PixelShuffle2d(nn.Module):
    """Pixel Shuffle 2D version."""

    def __init__(self, upscale_factor):
        super(PixelShuffle2d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.size()
        out_channels = in_channels // self.upscale_factor ** 2
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        out = x.reshape(batch_size, out_channels, self.upscale_factor,
                        self.upscale_factor, in_height, in_width)
        out = out.permute(0, 1, 4, 2, 5, 3)
        return out.reshape(batch_size, out_channels, out_height, out_width)


class CARAFEModule(nn.Module):
    """CARAFE is a content-based up-sampling module.

    Paper: https://arxiv.org/abs/1905.02188
    It can be used to replace the up-sampling layer in networks.

    Args:
        in_channels (int): Input channels.
        upscale_factor (int): The upsampling scale.
        compressed_channels (int): The channels number for compressed feature.
        upsample_kernel_size (int): The size of neighbor of the location.
    """

    def __init__(self,
                 in_channels,
                 upscale_factor,
                 compressed_channels=64,
                 upsample_kernel_size=7):
        super(CARAFEModule, self).__init__()
        self.in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.compressed_channels = compressed_channels
        self.upsample_kernel_size = upsample_kernel_size
        # empirical formula: k_encoder = k_up - 2
        self.encoder_kernel_size = self.upsample_kernel_size - 2

        # channel compressor
        self.channel_compressor = nn.Conv2d(
            self.in_channels, self.compressed_channels, 1)
        # content encoder
        encoder_out_channels = (self.upscale_factor *
                                self.upsample_kernel_size) ** 2
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            encoder_out_channels,
            self.encoder_kernel_size,
            padding=self.encoder_kernel_size // 2)
        # pixel shuffle layer
        self.pixel_shuffle = PixelShuffle2d(self.upscale_factor)
        # kernel normalizer
        self.kernel_normalizer = nn.Softmax(dim=1)
        # carafe layer
        self.carafe_layer = CarafeLayer(kernel_size=self.upsample_kernel_size,
                                        up_factor=self.upscale_factor)

    def forward(self, x):
        # (b, c, H, W) -> (b, c_compressed, H, W)
        compressed_x = self.channel_compressor(x)
        # (b, c_compressed, H, W) -> (b, (k_up * k_kernel)**2, H, W)
        content_x = self.content_encoder(compressed_x)
        # (b, (k_up * k_kernel)**2, H, W) -> ((b, k_kernel**2, k_upH, k_upW))
        upsampled_map = self.pixel_shuffle(content_x)
        # normalized kernel map
        kernel_map = self.kernel_normalizer(upsampled_map)
        # carafe layer
        out = self.carafe_layer(x, kernel_map)
        return out
