import torch
import torch.nn as nn
import torch.nn.functional as F

from snmi.nets.layers import SpatialTransformer, VecInt, ResizeTransform



class MrReg(nn.Module):
    def __init__(self, in_channel=1, n_layers=5, int_steps=1):
        super().__init__()
        self.enc = Encoder(in_channel*2, 32, n_layers)
        self.dec = Decoder(32, n_layers, int_steps)

    def forward(self, source, target):
        x = torch.concat([source, target], 1)
        x, x_enc = self.enc(x)
        disps, disps_res = self.dec(x, x_enc)
        return disps, disps_res
 

class Encoder(nn.Module):
    def __init__(self, in_channel, n_channel, n_layers=5):
        super().__init__()
        assert n_layers > 0
        self.n_layers = n_layers
        self.enc_list = nn.ModuleList()

        self.enc_list.append(ResidualBlock(in_channel, n_channel))
        for i in range(n_layers-1):
            dw = nn.Conv2d(n_channel, n_channel, 3, 2, padding=1, bias=True)
            res = ResidualBlock(n_channel, n_channel)
            self.enc_list.append(nn.Sequential(dw, nn.LeakyReLU(), res))

    def forward(self, x):
        xs = []
        for i in range(self.n_layers):
            x = self.enc_list[i](x)
            xs.append(x)
        return x, xs

class Decoder(nn.Module):
    def __init__(self, n_channel, n_layers, int_step):
        super().__init__()
        self.n_layers = n_layers
        self.dec_list = nn.ModuleList()

        self.dec_list.append(DecLayer(n_channel, int_step, up=False))
        for i in range(n_layers-1):
            self.dec_list.append(DecLayer(n_channel, int_step, up=True))
        self.rescaler = ResizeTransform(0.5, ndims=2)

    def forward(self, x, xs_enc):
        disps_res = []
        disps = []
        x, disp_res_pos, disp_res_neg = self.dec_list[0](x, None)
        disps_res.append([disp_res_pos, disp_res_neg])
        disps.append(disps_res[0])
        for i in range(1, self.n_layers):
            x, disp_res_pos, disp_res_neg = self.dec_list[i](x, xs_enc[-i-1])
            disp_pos = self.rescaler(disps[-1][0]) + disp_res_pos
            disp_neg = self.rescaler(disps[-1][1]) + disp_res_neg
            disps_res.append([disp_res_pos, disp_res_neg])
            disps.append([disp_pos, disp_neg])
        
        return disps, disps_res


class DecLayer(nn.Module):
    def __init__(self, n_channel, int_step, up=True):
        super().__init__()
        self.int_step = int_step
        self.deconv = None
        if up:
            self.deconv = nn.ConvTranspose2d(n_channel, n_channel, 2, 2, padding=0, bias=True)
            self.conv = nn.Conv2d(n_channel*2, n_channel, 3, padding='same', bias=True)
        else:
            self.conv = nn.Conv2d(n_channel, n_channel, 3, padding='same', bias=True)
        self.flow_pos = nn.Conv2d(n_channel, 2, 3, padding='same', bias=True)

    def forward(self, x, x_enc):
        if self.deconv:
            x = self.deconv(x)
            x = F.leaky_relu(x)
        if x_enc is not None:
            x = torch.concat([x, x_enc], 1)
        x = self.conv(x)
        flow_pos = self.flow_pos(x)
        flow_neg = -flow_pos

        if self.int_step > 0:
            vecint = VecInt(x.shape[2:], self.int_step)

            flow_pos = vecint(flow_pos)
            flow_neg = vecint(flow_neg)

        return x, flow_pos, flow_neg


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding='same', bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding='same', bias=True)
        self.addition = nn.Conv2d(in_channel+out_channel, out_channel, 3, padding='same', bias=True)
    
    def forward(self, x_in):
        x = self.conv1(x_in)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.addition(torch.concat([x_in, x], 1))
        x = F.leaky_relu(x)
        return x 
