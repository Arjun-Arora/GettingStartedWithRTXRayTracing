import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(input_channels, output_channels, stride=1, padding=0, bias=False):
    return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)    

def conv3x3(input_channels, output_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

class ApplyKernel(nn.Module):
    """
    apply convolution with pixel-wise kernels as in KPCN
    """
    def __init__(self, kernel_size):
        super(ApplyKernel, self).__init__()
        self.ks = kernel_size

    def forward(self, x, k, padding=True, padding_value=0):
        """
        :param x: (b, c, hx, wx)
        :param k: pixel-wise kernel (b, ks^2, hk, wk)
        :param padding: if padding is True, hx=hk and wx=wk. Then we need to pad x with padding_value.
                        if padding is False, hx=hk+ks-1, wx=wk+ks-1. No padding is needed.
        """
        orig_dev = x.device
        assert self.ks * self.ks == k.shape[1]
        b, ks2, h, w = k.shape
        b, c, hx, wx = x.shape
        half_ks = (self.ks - 1) // 2
        if padding:
            assert h == hx and w == wx
            x = nn.functional.pad(x, (half_ks, half_ks, half_ks, half_ks), value=padding_value)
        else:
            assert hx == h + self.ks - 1 and wx == w + self.ks - 1
        expand_x = []
        for ii in range(self.ks):
            for jj in range(self.ks):
                expand_x.append(x[:, :, ii: ii+h, jj: jj+w])
        expand_x = torch.stack(expand_x, 1)  # b, ks*ks, c, h, w
        k = torch.unsqueeze(k, 2)  # b, ks*ks, 1, h, w
        y = torch.sum(expand_x * k, 1)  # b, c, h, w
        return y.to(orig_dev)

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(input_channels, output_channels, stride)
        self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(output_channels)
        # self.conv2 = conv3x3(output_channels, output_channels)
        # self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample

    def forward(self, x):
        res = x
        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample != None:
            res = self.downsample(res)
        out = out + res
        out = self.relu(out)
        return out

class ESPCN_KPCN(nn.Module):
    def __init__(self, upscale_factor=1, input_channel_size=14, kernel_size=3):
        super(ESPCN_KPCN, self).__init__()

        self.input_channel_size = input_channel_size
        self.conv1 = nn.Conv2d(input_channel_size, 8, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(8, 4, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(4, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.num_res_blocks = 1

        # source encoder
        self.conv4 = conv3x3(input_channel_size, 14)

        # spatial-feature extractor
        self.resblock = ResBlock(14, 14)

        # kernel predictor
        self.conv5 = conv1x1(14, kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)

        self.apply_kernel = ApplyKernel(kernel_size)

    def forward(self, x, g):
        x = torch.cat((x, g), dim=1)
        assert(x.shape[1] == self.input_channel_size)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x_up = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))

        x = torch.cat((x_up, g), dim=1)

        x = self.conv4(x)
        x = self.resblock(x)
        x = self.conv5(x)
        # x is kernel here
        x = self.softmax(x)
        return self.apply_kernel(x_up, x), x_up

if __name__ == "__main__":
    test_x = torch.rand(1,3,128,128)
    test_g = torch.rand(1,11,128,128)
    model = ESPCN_KPCN(kernel_size=3)
    out, _ = model(test_x, test_g)
    print(out.shape)
