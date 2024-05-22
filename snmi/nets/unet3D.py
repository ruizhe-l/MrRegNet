import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet3d(nn.Module):
    def __init__(self, n_classes, in_channels, root_channels=16, n_layers=5, kernel_size=3, use_bn=False, use_res=False):
        super().__init__()
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.n_layers = n_layers
        n_channels = root_channels
        
        self.down_list.append(ConvBlock(in_channels, n_channels, kernel_size, use_bn, use_res))
        for _ in range(n_layers-1):
            self.down_list.append(ConvBlock(n_channels, n_channels*2, kernel_size, use_bn, use_res))
            n_channels *= 2
        
        for _ in range(n_layers-1):
            self.up_list.append(UpBlock(n_channels, n_channels//2, kernel_size, use_bn, use_res))
            n_channels //= 2

        self.conv_out = nn.Conv3d(n_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x, dropout_rate=0):
        down_xs = []
        for i in range(self.n_layers):
            if i > 0:
                x = F.max_pool3d(x, 2)
            x = self.down_list[i](x, dropout_rate)
            down_xs.append(x)

        for i in range(self.n_layers-1):
            x = self.up_list[i](x, down_xs[-2-i], dropout_rate)
        
        x = self.conv_out(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=False, use_res=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding='same', bias=False)
        
        self.bn1 = nn.BatchNorm3d(out_channels) if use_bn else None
        self.bn2 = nn.BatchNorm3d(out_channels) if use_bn else None

        self.res = None
        if use_res:
            res_layers = []
            if in_channels != out_channels:
                res_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, padding='same', bias=False))
                if use_bn:
                    res_layers.append(nn.BatchNorm3d(out_channels))
            self.res = nn.Sequential(res_layers)

    def forward(self, x, dropout_rate):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x) if self.bn1 else x
        x = F.relu(x)
        x = F.dropout3d(x, dropout_rate)

        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 else x

        x = x + self.res(residual) if self.res else x
        x = F.relu(x)
        x = F.dropout3d(x, dropout_rate)

        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=False, use_res=False):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, use_bn, use_res)

    def forward(self, x, skip, dropout_rate):
        x = self.deconv(x)
        x = torch.cat([skip, x], 1)
        x = F.relu(x)
        x = F.dropout3d(x, dropout_rate)

        x = self.conv_block(x, dropout_rate)
        return x


