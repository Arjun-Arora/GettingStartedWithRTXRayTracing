import torch
import torch.nn as nn

def conv1x1(input_channels, output_channels, stride=1, padding=0, bias=False):
    return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)    

def conv3x3(input_channels, output_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def conv5x5(input_channels, output_channels, stride=1, padding=2, bias=False):
    return nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=stride, padding=padding, bias=bias)

class ApplyKernel(nn.Module):
    """
    apply convolution with pixel-wise kernels as in KPCN
    """
    def __init__(self, kernel_size):
        super(ApplyKernel, self).__init__()
        self.ks = kernel_size

    def forward(self, x, k, padding=False, padding_value=0):
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
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = conv3x3(output_channels, output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample != None:
            res = self.downsample(res)
        out = out + res
        out = self.relu(out)
        return out

class KPCN(nn.Module):
    def __init__(self, input_channels, kernel_size=21):
        super(KPCN, self).__init__()
        # source encoder
        self.conv1 = conv3x3(input_channels, 100)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(100, 100)

        # spatial-feature extractor
        self.resblocks = nn.Sequential(*self.make_extractor(100, 100))

        # kernel predictor
        self.conv3 = conv1x1(100, kernel_size * kernel_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = conv1x1(kernel_size * kernel_size, kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)

    def make_extractor(self, input_channels, output_channels):
        resblock_list = []
        # maybe use append then * 23?
        [resblock_list.append(ResBlock(input_channels, input_channels)) for i in range(23)]
        resblock_list.append(ResBlock(input_channels, output_channels))
        return resblock_list

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        for i in range(24):
            x = self.resblocks[i](x)
        x = self.conv3(x)
        x = self.relu2(x)
        x = self.conv4(x)
        return self.softmax(x)

class KPCN_light(nn.Module):
    def __init__(self, input_channels, kernel_size=21):
        super(KPCN_light, self).__init__()
        # source encoder
        self.conv1 = conv3x3(input_channels, 100)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(100, 100)

        # spatial-feature extractor
        self.resblocks = nn.Sequential(*self.make_extractor(100, 100))

        # kernel predictor
        self.conv3 = conv1x1(100, kernel_size * kernel_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = conv1x1(kernel_size * kernel_size, kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)

    def make_extractor(self, input_channels, output_channels):
        resblock_list = []
        # maybe use append then * 23?
        [resblock_list.append(ResBlock(input_channels, input_channels)) for i in range(3)]
        resblock_list.append(ResBlock(input_channels, output_channels))
        return resblock_list

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        for i in range(8):
            x = self.resblocks[i](x)
        x = self.conv3(x)
        x = self.relu2(x)
        x = self.conv4(x)
        return self.softmax(x)

if __name__ == "__main__":
    test_in1 = torch.rand(4,14,128,128)
    test_in2 = torch.rand(4,14,128,128)
    model1 = KPCN(14)
    model2 = KPCN_light(14)
    w1 = model1(test_in1)
    w2 = model2(test_in2)
    apply_kernel = ApplyKernel(21)
    test_out1 = apply_kernel.forward(test_in1, w1, padding=True)
    test_out2 = apply_kernel.forward(test_in2, w2, padding=True)
    print(w1.shape)
    print(test_out1.shape)
    print(w2.shape)
    print(test_out2.shape)