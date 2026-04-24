import torch
import torch.nn as nn
import logging
from logging.handlers import RotatingFileHandler

def conv1d_5(inplanes, outplanes, stride=1):
    return nn.Conv1d(inplanes, outplanes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

# 下采样block，下采样过程中使用resnet作为backbone
class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, bn=False):
        super(Block, self).__init__()
        self.bn = bn
        self.conv1 = conv1d_5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv1d_5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # print('Downsampe Block forward x.shape={}'.format(x.shape))
        out = self.conv1(x)
        # print('Downsampe Block forward out.shape={}'.format(out.shape))
        if self.bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

# 上采样block
class Decoder_block(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=5, stride=2, ppadding=0, outputpadding=0):
        super(Decoder_block, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=outplanes,
                           padding=ppadding, output_padding=outputpadding, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv1 = conv1d_5(inplanes, outplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv1d_5(outplanes, outplanes)

    def forward(self, x1, x2):
        # print('x1.shape={}, x2.shape={}'.format(x1.shape, x2.shape))
        x1 = self.upsample(x1)
        # print('after upsample x1.shape={}, x2.shape={}, kernel_size={}, stride={}'.format(x1.shape, x2.shape, self.kernel_size, self.stride))
        out = torch.cat((x1, x2), dim=1)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        return out


class Uresnet(nn.Module):
    def __init__(self, inplanes=6, layers=[2,2,2,2]):
        super(Uresnet, self).__init__()
        self.inplanes = inplanes
        self.encoder1 = self._make_encoder(Block, 32, layers[0], 2)
        self.encoder2 = self._make_encoder(Block, 64, layers[1], 2)
        self.encoder3 = self._make_encoder(Block, 128, layers[2], 2)
        self.encoder4 = self._make_encoder(Block, 256, layers[3], 2)
        self.decoder3 = Decoder_block(256, 128, stride=2, kernel_size=2)
        self.decoder2 = Decoder_block(128, 64, stride=2, kernel_size=2, outputpadding=0)
        self.decoder1 = Decoder_block(64, 32, stride=2, kernel_size=2, outputpadding=0)
        # 如果要修改输出的通道，在这里修改
        self.conv1x1 = nn.ConvTranspose1d(32, 6, kernel_size=2, stride=2, bias=False)

    def _make_encoder(self, block, planes, blocks, stride=1):
        downsample = None
        if self.inplanes != planes or stride != 1:
            # print('update downsample: planes={}, stride={}'.format(planes, stride))
            downsample = nn.Conv1d(self.inplanes, planes, stride=stride, kernel_size=1, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(x.shape[0], 1, -1)
        # print('x.shape={}'.format(x.shape))          # x.shape=torch.Size([50, 6, 201])
        down1 = self.encoder1(x)
        # print('down1.shape={}'.format(down1.shape))  # down1.shape=torch.Size([50, 32, 101])
        down2 = self.encoder2(down1)
        # print('down2.shape={}'.format(down2.shape))  # down2.shape=torch.Size([50, 64, 51])
        down3 = self.encoder3(down2)
        # print('down3.shape={}'.format(down3.shape))  # down3.shape=torch.Size([50, 128, 26])
        
        # down4 = self.encoder4(down3)
        # print('down4.shape={}'.format(down4.shape))  # down4.shape=torch.Size([50, 256, 13])
        # up3 = self.decoder3(down4, down3) 
        # print('up3.shape={}'.format(up3.shape))      # up3.shape=torch.Size([50, 128, 26])
        # up2 = self.decoder2(up3, down2)

        up2 = self.decoder2(down3, down2)
        # print('up2.shape={}'.format(up2.shape))
        up1 = self.decoder1(up2, down1)
        # print('up1.shape={}'.format(up1.shape))

        # print('UNet up1.shape={}'.format(up1.shape))
        out = self.conv1x1(up1)
        # print('UNet out.shape={}'.format(out.shape))
        # return out.view(out.shape[0], out.shape[-1])
        return out
