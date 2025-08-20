import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

def catcat(inputs1, inputs2):
    return torch.cat((inputs1, inputs2), 2)
class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x1, x2):
        return catcat(x1, x2)

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention,self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((None, 1))
        self.max = nn.AdaptiveMaxPool2d((1, None))
        self.conv1x1 = default_conv(dim, dim//2, kernel_size=1, bias=True)
        self.conv3x3 = default_conv(dim//2, dim, kernel_size=3, bias=True)
        self.con3x3 = default_conv(dim, dim, kernel_size=3, bias=True)
        self.GELU = nn.GELU()
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
    def forward(self, x):
        batch_size, channel, height, width = x.size()
        x_h = self.avg(x)
        x_w = self.max(x)
        x_h = torch.squeeze(x_h, 3)
        x_w = torch.squeeze(x_w, 2)
        x_h1 = x_h.unsqueeze(3)
        x_w1 = x_w.unsqueeze(2)
        x_h_w = catcat(x_h, x_w)
        x_h_w = x_h_w.unsqueeze(3)
        x_h_w = self.conv1x1(x_h_w)
        x_h_w = self.GELU(x_h_w)
        x_h_w = torch.squeeze(x_h_w, 3)
        x1, x2 = torch.split(x_h_w, [height, width], 2)
        x1 = x1.unsqueeze(3)
        x2 = x2.unsqueeze(2)
        x1 = self.conv3x3(x1)
        x2 = self.conv3x3(x2)
        mix1 = self.mix1(x_h1, x1)
        mix2 = self.mix2(x_w1, x2)
        x1 = self.con3x3(mix1)
        x2 = self.con3x3(mix2)
        matrix = torch.matmul(x1, x2)
        matrix = torch.sigmoid(matrix)
        final = torch.mul(x, matrix)
        final = x + final
        return final

def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1 * torch.exp(1j * (fft_.angle())))
    return fft_.real


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class LKABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, linear=False, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(n_feat * 4, n_feat * 2, 3, 1, 1, bias=False))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class se_block(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(se_block, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, inplanes//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = x
        x = self.se(x)
        return input*x

class se_block(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(se_block, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, inplanes//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = x
        x = self.se(x)
        return input*x

class residual_block(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super(residual_block, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=True),
            se_block(out_channels, reduction=16)
        )

    def forward(self, x):
        input = x
        x = self.residual(x)
        return input + x

class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        hidden_features = int(dim * 3)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        return self.project_out(x)

#change
class UpSample_same(nn.Module):
    def __init__(self, in_channels):
        super(UpSample_same, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, int(in_channels/2), 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, int(in_channels/2), 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):

        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CFA(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(CFA, self).__init__()
        modules_body = [nn.Conv2d(n_feat, n_feat, kernel_size, padding=1), act, nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)]
        self.body = nn.Sequential(*modules_body)
        ## Pixel Attention
        self.SA = spatial_attn_layer()
        ## Channel Attention
        self.CA = CALayer(n_feat)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

class BICA(nn.Module):
    def __init__(self, chns):
        super(BICA, self).__init__()

        self.norm = LayerNorm(chns, 'BiasFree')
        self.act = nn.GELU()

        self.DA = CFA(chns)

        self.Conv1 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv4 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv5 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)
        self.Conv6 = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=1)

        self.conv_dialted = nn.Conv2d(chns, chns, kernel_size=3, stride=1, padding=3, dilation=3)

        self.down = DownSample(chns)

        self.up = UpSample(chns*2)
        self.Conv1_1 = nn.Conv2d(chns*2, chns*2, kernel_size=3, stride=1, padding=1)
        self.Conv1_2 = nn.Conv2d(chns*2, chns*2, kernel_size=3, stride=1, padding=1)
        self.upsame = UpSample_same(chns*2)

    def forward(self, x):
        x = self.norm(x)

        x2 = self.DA(x)

        # Resolution-Guided Intrinsic Attention Module (ReGIA)
        x1 = x2
        x1 = self.down(x1)
        x1 = self.act(self.Conv1_1(x1))
        x1 = self.Conv1_2(x1)
        x1 = self.up(x1)
        x1 = torch.sigmoid(x1)
        x11 = x1 * x2

        # Hierarchical Context-Aware Feature Extraction (HCAFE)
        x3 = self.conv_dialted(x2)
        x4 = self.Conv4(x2)
        x22 = self.Conv2(x3 + x4)
        x22 = self.Conv5(self.act(x22))

        out = torch.cat([x11, x22], dim=1)
        out = self.upsame(out) + x2

        out = self.Conv6(self.act(out))

        return out


# Frequency-Domain Pixel Attention
class FDPA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.conv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x2_fft = torch.fft.fft2(x2)
        
        # Get phase
        phase = torch.angle(x2_fft)
        
        # Reconstruct complex tensor with unit magnitude and phase
        phase_only_fft = torch.polar(torch.ones_like(phase), phase)
        
        # Apply inverse FFT to get phase-only attention
        out = torch.fft.ifft2(phase_only_fft)
        out = torch.real(out)  # Discard imaginary part (can also use torch.abs)

        return out * self.alpha + x * self.beta


class HybridDomainAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias=False)
        self.fpa = FDPA(dim)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.fpa(x)
        s_attn = self.conv(self.pool(self.norm1(out)))
        out = s_attn * out
        out = x + out
        return out + self.ffn(self.norm2(out))




class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        k = self.kernel_size()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        return k if k % 2 else k + 1

    def forward(self, x):
        x1 = inv_mag(x)
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)




class CSC_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)
        self.dw_13 = nn.Conv2d(dim, dim, (1, ker), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, (ker, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, ker, padding=pad, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
        return self.out_conv(self.act(out))


class UIR_PolyKernel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=36, bias=False):
        super().__init__()
        self.input_embed = nn.Conv2d(in_channels, dim, 1)
        self.encoder_level1 = HybridDomainAttention(dim)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = LKABlock(dim * 2)
        self.attention_lka1_to_lka2 = Attention(dim * 2)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = LKABlock(dim * 4)
        self.attention_lka2_to_csc = Attention(dim * 4)
        self.bottleneck = CSC_Block(dim * 4)

        self.eca_level3 = ECA(dim * 4)
        self.eca_level2 = ECA(dim * 2)
        self.eca_level1 = ECA(dim)
        self.eca_atteention_lka1_to_lka2 = ECA(dim * 2)
        self.eca_atteention_lka2_to_csc = ECA(dim * 4)

        self.reduce_before_aha= nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.attention_csc_to_lka2 = Attention(dim * 4)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = LKABlock(dim * 4)
        self.up3_2 = Upsample(dim * 4)
        self.reduce_before_aha2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.attention_lka2_to_lka1 = Attention(dim * 2)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = LKABlock(dim * 2)
        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.decoder_level1 = HybridDomainAttention(dim)

        self.final_conv = nn.Conv2d(dim, out_channels, 1)
        self.norm = nn.Sigmoid()

        self.start_conv = default_conv(in_channels=3, out_channels=256, kernel_size=3, bias=True)
        self.Residual_block = residual_block(in_channels=256, out_channels=256, kernel_size=3)
        self.final_conv_drb = default_conv(in_channels=256, out_channels=3, kernel_size=3, bias=True)

    def forward(self, x):
        inp = self.input_embed(x)
        out_enc_level1 = self.encoder_level1(inp)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        aha1_output = self.attention_lka1_to_lka2(out_enc_level2)
        inp_enc_level3 = self.down2_3(aha1_output)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        aha2_output = self.attention_lka2_to_csc(out_enc_level3)
        latent = self.bottleneck(aha2_output)

        out_enc_level3 = self.eca_level3(out_enc_level3)
        out_enc_level2 = self.eca_level2(out_enc_level2)
        out_enc_level1 = self.eca_level1(out_enc_level1)
        out_eca_aha1= self.eca_atteention_lka1_to_lka2(aha1_output)
        out_eca_aha2 = self.eca_atteention_lka2_to_csc(aha2_output)

        first_decode = cat(latent, out_eca_aha2)
        aha_input=self.reduce_before_aha(first_decode)
        aha_input = self.attention_csc_to_lka2(aha_input)
        inp_dec_level3 = cat(aha_input, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        aha2_input = cat(inp_dec_level2, out_eca_aha1)
        aha2_input = self.reduce_before_aha2(aha2_input)
        aha2_input = self.attention_lka2_to_lka1(aha2_input)
        inp_dec_level2 = cat(aha2_input, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, out_enc_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)


      
        return self.norm(self.final_conv(out_dec_level1) + x)


if __name__ == '__main__':
    from thop import profile, clever_format

    t = torch.randn(1, 3, 256, 256).cuda()
    model = UIR_PolyKernel().cuda()
    macs, params = profile(model, inputs=(t,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
