import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from miscc.config import cfg
from attention import MultiSpatialAttention, MultiChannelAttention
from darts.search_units import FixedCondGenUnit, SearchableCondGenUnit, normalize_arch

class NetGDARTS(nn.Module):
    """
    DARTS supernet for high-resolution conditional generator search.

    Search sites:
      s1: after block5, before word/channel attention
      s2: after word/channel attention fusion, before block6
      s3: after block7, before conv_img
    """

    def __init__(self, ngf=64, nz=100, lstm=None, unit_alpha_init=1.0e-3, arch_tau=1.0):
        super(NetGDARTS, self).__init__()
        self.ngf = ngf
        self.ef_dim = cfg.TEXT.EMBEDDING_DIM
        self.lstm = lstm
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)

        # TDM is intentionally disabled during architecture search.
        self.block0 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block1 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block2 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block3 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block4 = G_Block(ngf * 8, ngf * 4, lstm)
        self.block5 = G_Block(ngf * 4, ngf * 2, lstm)

        self.att = MultiSpatialAttention(ngf * 2, self.ef_dim, 1)
        self.channel_att = MultiChannelAttention(ngf * 2, self.ef_dim, 128, 1)

        self.block6 = G_Block(ngf * 2 * 3, ngf * 1, lstm)
        self.block7 = G_Block(ngf * 1, ngf * 1, lstm)

        self.search_s1 = SearchableCondGenUnit(ngf * 2, tau=arch_tau, alpha_init=unit_alpha_init)
        self.search_s2 = SearchableCondGenUnit(ngf * 2 * 3, tau=arch_tau, alpha_init=unit_alpha_init)
        self.search_s3 = SearchableCondGenUnit(ngf * 1, tau=arch_tau, alpha_init=unit_alpha_init)

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def arch_parameters(self):
        return [
            self.search_s1.alpha_visual,
            self.search_s1.alpha_text,
            self.search_s2.alpha_visual,
            self.search_s2.alpha_text,
            self.search_s3.alpha_visual,
            self.search_s3.alpha_text,
        ]

    def weight_parameters(self):
        arch_ids = {id(p) for p in self.arch_parameters()}
        for p in self.parameters():
            if id(p) not in arch_ids:
                yield p

    def genotype(self):
        return {
            "s1": self.search_s1.genotype(),
            "s2": self.search_s2.genotype(),
            "s3": self.search_s3.genotype(),
        }

    def alpha_summary(self):
        return {
            "s1": self.search_s1.alpha_summary(),
            "s2": self.search_s2.alpha_summary(),
            "s3": self.search_s3.alpha_summary(),
            "genotype": self.genotype(),
        }

    def forward(self, x, c, word_embs=None, mask=None):
        if (self.lstm is not None) and hasattr(self.lstm, "init_hidden"):
            self.lstm.init_hidden(x)

        out = self.fc(x)
        out = out.view(x.size(0), 8 * self.ngf, 4, 4)

        out = self.block0(out, c)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = F.avg_pool2d(out, 3, 1, 1)
        out = self.block1(out, c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block2(out, c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block3(out, c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block4(out, c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block5(out, c)
        out = self.search_s1(out, c, word_embs, mask)

        if (word_embs is not None) and (mask is not None):
            self.att.applyMask(mask)

        out_code = out
        if word_embs is not None:
            c_code, att = self.att(out_code, word_embs)
            c_code_channel, _ = self.channel_att(
                c_code, word_embs, out_code.size(2), out_code.size(3)
            )
            c_code = c_code.view(word_embs.size(0), -1, out_code.size(2), out_code.size(3))
            out = torch.cat([out_code, c_code, c_code_channel], dim=1)
        else:
            out = torch.cat([out_code, out_code, out_code], dim=1)
            att = None

        out = self.search_s2(out, c, word_embs, mask)
        out = self.block6(out, c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block7(out, c)
        out = self.search_s3(out, c, word_embs, mask)

        out = self.conv_img(out)
        return out, att
class NetGFixedDARTS(nn.Module):
    """
    Fixed architecture generator exported from NetGDARTS.
    """

    def __init__(self, arch, ngf=64, nz=100, lstm=None, unit_alpha_init=1.0e-3):
        super(NetGFixedDARTS, self).__init__()
        self.arch = normalize_arch(arch)
        self.ngf = ngf
        self.ef_dim = cfg.TEXT.EMBEDDING_DIM
        self.lstm = lstm
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)

        self.block0 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block1 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block2 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block3 = G_Block(ngf * 8, ngf * 8, lstm)
        self.block4 = G_Block(ngf * 8, ngf * 4, lstm)
        self.block5 = G_Block(ngf * 4, ngf * 2, lstm)

        self.att = MultiSpatialAttention(ngf * 2, self.ef_dim, 1)
        self.channel_att = MultiChannelAttention(ngf * 2, self.ef_dim, 128, 1)

        self.block6 = G_Block(ngf * 2 * 3, ngf * 1, lstm)
        self.block7 = G_Block(ngf * 1, ngf * 1, lstm)

        self.search_s1 = FixedCondGenUnit(
            ngf * 2,
            self.arch["s1"]["visual"],
            self.arch["s1"]["text"],
            alpha_init=unit_alpha_init,
        )
        self.search_s2 = FixedCondGenUnit(
            ngf * 2 * 3,
            self.arch["s2"]["visual"],
            self.arch["s2"]["text"],
            alpha_init=unit_alpha_init,
        )
        self.search_s3 = FixedCondGenUnit(
            ngf * 1,
            self.arch["s3"]["visual"],
            self.arch["s3"]["text"],
            alpha_init=unit_alpha_init,
        )


        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )


    def genotype(self):
        return dict(self.arch)

    def forward(self, x, c, word_embs=None, mask=None):
        if (self.lstm is not None) and hasattr(self.lstm, "init_hidden"):
            self.lstm.init_hidden(x)

        coarse_c = c
        fine_c = c
        low_c = coarse_c
        mid_c = fine_c
        high_c = fine_c

        out = self.fc(x)
        out = out.view(x.size(0), 8 * self.ngf, 4, 4)

        out = self.block0(out, low_c)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = F.avg_pool2d(out, 3, 1, 1)
        out = self.block1(out, low_c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block2(out, low_c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block3(out, low_c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block4(out, low_c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block5(out, mid_c)
        out = self.search_s1(out, mid_c, word_embs, mask)

        if (word_embs is not None) and (mask is not None):
            self.att.applyMask(mask)

        out_code = out
        if word_embs is not None:
            c_code, att = self.att(out_code, word_embs)
            c_code_channel, _ = self.channel_att(
                c_code, word_embs, out_code.size(2), out_code.size(3)
            )
            c_code = c_code.view(word_embs.size(0), -1, out_code.size(2), out_code.size(3))
            out = torch.cat([out_code, c_code, c_code_channel], dim=1)
        else:
            out = torch.cat([out_code, out_code, out_code], dim=1)
            att = None

        out = self.search_s2(out, mid_c, word_embs, mask)
        out = self.block6(out, mid_c)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        out = self.block7(out, high_c)
        out = self.search_s3(out, high_c, word_embs, mask)

        out = self.conv_img(out)
        return out, att
class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, lstm):
        super(G_Block, self).__init__()
        self.lstm = lstm

        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)

        self.affine4 = affine(in_ch)
        self.affine5 = affine(out_ch)

        self.fea_l = nn.Linear(in_ch, 256)
        self.fea_ll = nn.Linear(out_ch, 256)
        self.gamma = nn.Parameter(torch.zeros(1))


        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

        self.act = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, yy=None):
        y, _ = self.lstm(yy)

        h = self.affine0(x, y)
        h = self.act(h)
        h = self.affine1(h, y)
        h = self.act(h)


        h = self.c1(h)

        h = self.affine2(h, y)
        h = self.act(h)
        h = self.affine3(h, y)
        h = self.act(h)


        return self.c2(h)
class affine(nn.Module):
    def __init__(self, num_features, w_scale=0.1, b_scale=0.1):
        super(affine, self).__init__()
        self.w_scale = w_scale
        self.b_scale = b_scale

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        w_raw = self.fc_gamma(y)
        b_raw = self.fc_beta(y)

        weight = 1.0 + self.w_scale * torch.tanh(w_raw - 1.0)
        bias = self.b_scale * torch.tanh(b_raw)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias
class D_GET_LOGITS_att(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS_att, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 3, 1, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

        self.block = resD(ndf * 16 + 256, ndf * 16)

        self.joint_conv_att = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.softmax = nn.Softmax(2)


    def forward(self, out, y_):
        B, C, H, W = out.size()

        y = y_.view(B, 256, 1, 1)
        y = y.repeat(1, 1, H, W)

        h_c_code = torch.cat((out, y), 1)
        p = self.joint_conv_att(h_c_code)

        p = p.view(B, 1, H * W)
        p = self.softmax(p)
        p = p.view(B, 1, H, W)
        self.p = p

        p = p.repeat(1, 256, 1, 1)
        y_weighted = y * p

        h_c_code = torch.cat((out, y_weighted), 1)
        h_c_code = self.block(h_c_code)

        B2, C2, H2, W2 = h_c_code.size()

        y2 = y_.view(B2, 256, 1, 1)
        y2 = y2.repeat(1, 1, H2, W2)

        h_c_code = torch.cat((h_c_code, y2), 1)
        out = self.joint_conv(h_c_code)
        return out
class ProjectedTextConditionedDiscriminator(nn.Module):
    def __init__(self, img_dim=256, text_dim=256, hidden_dim=256):
        super().__init__()
        self.img_norm = nn.LayerNorm(int(img_dim))
        self.txt_norm = nn.LayerNorm(int(text_dim))
        self.img_proj = nn.Sequential(
            nn.Linear(int(img_dim), int(hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(int(text_dim), int(hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.uncond = nn.Linear(int(hidden_dim), 1)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, img_code, sent_cond):
        img_h = self.img_proj(self.img_norm(img_code.float()))
        txt_h = self.txt_proj(self.txt_norm(sent_cond.float()))
        img = F.normalize(img_h, dim=1)
        txt = F.normalize(txt_h, dim=1)
        cond = self.logit_scale.float().clamp(0.1, 10.0) * (img * txt).sum(dim=1)
        return self.uncond(img_h.float()).view(-1) + cond
class NetD(nn.Module):
    def __init__(self, ndf, use_ptcdfm=False,
                     ptcdfm_img_dim=256, ptcdfm_text_dim=256, ptcdfm_hidden_dim=256):
        super(NetD, self).__init__()
        self.use_ptcdfm = bool(use_ptcdfm)

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)
        self.block0 = resD(ndf * 1, ndf * 2)
        self.block1 = resD(ndf * 2, ndf * 4)
        self.block2 = resD(ndf * 4, ndf * 8)
        self.block3 = resD(ndf * 8, ndf * 16)
        self.block4 = resD(ndf * 16, ndf * 16)

        self.COND_DNET = D_GET_LOGITS_att(
            ndf,
        )

        self.PTC_DNET = None
        if self.use_ptcdfm:
            self.PTC_DNET = ProjectedTextConditionedDiscriminator(
                img_dim=ptcdfm_img_dim,
                text_dim=ptcdfm_text_dim,
                hidden_dim=ptcdfm_hidden_dim,
            )

    def forward(self, x, return_features=False):
        feats = []
        out = self.conv_img(x)
        out = self.block0(out)
        feats.append(out)
        out = self.block1(out)
        feats.append(out)
        out = self.block2(out)
        feats.append(out)
        out = self.block3(out)
        feats.append(out)
        out = self.block4(out)
        feats.append(out)
        if return_features:
            return out, feats
        return out
class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)
