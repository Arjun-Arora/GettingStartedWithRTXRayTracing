import torch
import torch.nn as nn
import torch.nn.functional as F


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, n_channel, n_denselayer, n_feat, n_scale, growth_rate):
        super(RDN, self).__init__()
        nChannel = n_channel
        nDenselayer = n_denselayer
        nFeat = n_feat
        scale = n_scale
        growthRate = growth_rate

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat * scale * scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        F_ = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)

        output = self.conv3(us)

        return output


def _crop_and_merge(to_crop: torch.Tensor, to_merge_to: torch.Tensor) -> torch.Tensor:
    padding = [0, 0, to_merge_to.size()[2] - to_crop.size()[2], to_merge_to.size()[3] - to_crop.size()[3]]
    cropped_to_crop = F.pad(to_crop, padding)

    return torch.cat((cropped_to_crop, to_merge_to), dim=1)


class UNet(nn.Module):
    def __init__(self, depth: int = 5):
        super(UNet, self).__init__()

        # part 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(2, stride=1, padding=1)

        # part 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(2, stride=1, padding=1)

        # part 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(2, stride=1, padding=1)

        # part 4
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, padding=1)

        # part5
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=True)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=True)
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, padding=1, bias=True)

        # part6
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, padding=1, bias=True)

        # part7
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, padding=1, bias=True)

        # part8
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, padding=1, bias=True)

        # part9
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv19 = nn.Conv2d(64, 3, kernel_size=1, bias=True)

    def forward(self, x):
        level_1_down = F.relu(self.conv2(F.relu(self.conv1(x))))
        level_2_down = F.relu(self.conv4(F.relu(self.conv3(self.pool1(level_1_down)))))
        level_3_down = F.relu(self.conv6(F.relu(self.conv5(self.pool2(level_2_down)))))
        level_4_down = F.relu(self.conv8(F.relu(self.conv7(self.pool3(level_3_down)))))

        level_5_up = self.up_conv1(F.relu(self.conv10(F.relu(self.conv9(self.pool4(level_4_down))))))
        level_6_up = self.up_conv2(F.relu(self.conv12(F.relu(self.conv11(_crop_and_merge(level_4_down, level_5_up))))))
        level_7_up = self.up_conv3(F.relu(self.conv14(F.relu(self.conv13(_crop_and_merge(level_3_down, level_6_up))))))
        level_8_up = self.up_conv4(F.relu(self.conv16(F.relu(self.conv15(_crop_and_merge(level_2_down, level_7_up))))))
        out = self.conv19(F.relu(self.conv18(F.relu(self.conv17(_crop_and_merge(level_1_down, level_8_up))))))

        return F.relu(out)


class ESPCN(nn.Module):
    def __init__(self, upscale_factor: int, input_channel_size: int, output_channel_size: int):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(input_channel_size, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, output_channel_size * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x
