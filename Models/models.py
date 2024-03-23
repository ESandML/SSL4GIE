import torch
import torch.nn as nn
import torchvision

from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import VisionTransformer
from timm.models.hub import download_cached_file

from .moco_v3 import vits
from .mae import models_mae

from .DPT_decoder import DPT_decoder


class ResNet_Dec_Block(nn.Module):
    def __init__(self, channels, fusion=False):
        super(ResNet_Dec_Block, self).__init__()
        if fusion:
            self.identity = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1), nn.BatchNorm2d(channels)
            )
            conv1 = nn.Conv2d(channels * 2, channels // 4, 1)
        else:
            self.identity = nn.Identity()
            conv1 = nn.Conv2d(channels, channels // 4, 1)
        self.process = nn.Sequential(
            conv1,
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.identity(x)
        x = self.process(x)
        x += identity
        return self.relu(x)


class ResNet_Dec_Level(nn.Module):
    def __init__(self, channels, n_blocks):
        super(ResNet_Dec_Level, self).__init__()
        self.chan_reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), nn.BatchNorm2d(channels)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        blocks = [ResNet_Dec_Block(channels, fusion=True)]
        for _ in range(1, n_blocks):
            blocks.append(ResNet_Dec_Block(channels, fusion=False))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x_low, x_high):
        x = torch.cat((self.up(self.chan_reduce(x_low)), x_high), 1)
        return self.blocks(x)


class ResNet_from_Any(torchvision.models.resnet.ResNet):
    def __init__(
        self, weight_path, head, num_classes, frozen, dense, ImageNet_weights=False
    ):
        super(ResNet_from_Any, self).__init__(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3]
        )
        if ImageNet_weights:
            state_dict = torchvision.models.utils.load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                progress=True,
            )
            self.load_state_dict(state_dict)

        self.fc = nn.Identity()
        if weight_path is not None:
            weights = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(weights)

        self.head = head
        if head:
            self.lin_head = nn.Linear(2048, num_classes)
        self.frozen = frozen
        self.dense = dense

        if self.dense:
            self.decoder_levels = nn.ModuleList(
                [
                    ResNet_Dec_Level(1024, 3),
                    ResNet_Dec_Level(512, 3),
                    ResNet_Dec_Level(256, 3),
                ]
            )
            self.output_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

    def forward_features(self, x):
        if self.dense:
            fmaps = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.dense:
            fmaps.append(x)
        x = self.layer2(x)
        if self.dense:
            fmaps.append(x)
        x = self.layer3(x)
        if self.dense:
            fmaps.append(x)
        x = self.layer4(x)
        if self.dense:
            fmaps.append(x)
        return x if not self.dense else fmaps

    def decode(self, x):
        out = self.decoder_levels[0](x[-1], x[-2])
        for i in range(1, len(self.decoder_levels)):
            out = self.decoder_levels[i](out, x[-i - 2])

        out = self.output_conv(out)

        return out

    def _forward_impl(self, x):
        if self.frozen:
            with torch.no_grad():
                x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        if not self.dense:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if self.head:
                x = self.lin_head(x)
        else:
            x = self.decode(x)

        return x


class WindowedAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=16,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

    def forward(self, x):
        B, N, C = x.shape
        s = int(N**0.5)
        idxs = torch.arange(N).reshape(s, s)
        perm = []
        for i in range(0, s, self.window_size):
            for j in range(0, s, self.window_size):
                perm.append(
                    idxs[i : i + self.window_size, j : j + self.window_size].reshape(
                        self.window_size**2
                    )
                )
        windows = len(perm)
        perm = torch.cat(perm)
        inv_perm = torch.argsort(perm)

        x = x[:, perm]

        qkv = (
            self.qkv(x)
            .reshape(B, windows, N // windows, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, N, C)
        x = x[:, inv_perm]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTDet_FPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fpn1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(768, 256, 1),
            nn.LayerNorm((256, 32, 32)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LayerNorm((256, 32, 32)),
        )
        self.fpn2 = nn.Sequential(
            nn.Conv2d(768, 256, 1),
            nn.LayerNorm((256, 64, 64)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LayerNorm((256, 64, 64)),
        )
        self.fpn3 = nn.Sequential(
            nn.ConvTranspose2d(768, 768, 2, 2),
            nn.Conv2d(768, 256, 1),
            nn.LayerNorm((256, 128, 128)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LayerNorm((256, 128, 128)),
        )
        self.fpn4 = nn.Sequential(
            nn.ConvTranspose2d(768, 768, 2, 2),
            nn.LayerNorm((768, 128, 128)),
            nn.GELU(),
            nn.ConvTranspose2d(768, 768, 2, 2),
            nn.Conv2d(768, 256, 1),
            nn.LayerNorm((256, 256, 256)),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LayerNorm((256, 256, 256)),
        )

    def forward(self, x):
        B = x.shape[0]
        C = x.shape[2]
        H = int(x.shape[1] ** 0.5)
        W = H
        x = torch.reshape(x.transpose(1, 2), (B, C, H, W))
        l1 = self.fpn1(x)
        l2 = self.fpn2(x)
        l3 = self.fpn3(x)
        l4 = self.fpn4(x)
        pool = nn.functional.max_pool2d(l1, kernel_size=1, stride=2, padding=0)
        return OrderedDict([("0", l4), ("1", l3), ("2", l2), ("3", l1), ("pool", pool)])


class VisionTransformer_from_Any(VisionTransformer):
    def __init__(
        self,
        head,
        num_classes,
        frozen,
        dense,
        det,
        fixed_size,
        embed_dim,
        depth,
        num_heads,
        out_token,
        ImageNet_weights=False,
    ):
        super().__init__(
            patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )

        if det:
            for i in [0, 1, 3, 4, 6, 7, 9, 10]:
                self.blocks[i].attn = WindowedAttention(
                    dim=embed_dim, num_heads=num_heads, qkv_bias=True
                )
        if ImageNet_weights:
            loc = download_cached_file(
                "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
            )
            self.load_pretrained(loc)
        self.head = nn.Identity()

        self.head_bool = head
        if head:
            self.lin_head = nn.Linear(embed_dim, num_classes)
        self.frozen = frozen
        self.dense = dense
        self.det = det

        if dense:
            self.decoder = DPT_decoder(num_classes=num_classes, dense=dense)
        if det:
            self.fixed_size = fixed_size
            self.fpn = ViTDet_FPN()
            self.out_channels = 256
            self.patch_embed.img_size = (fixed_size, fixed_size)
            del self.cls_token
        self.out_token = out_token

    def _pos_embed_interp(self, x):
        pos_embed2d = self.pos_embed[:, 1:, :].transpose(1, 2).reshape(1, 768, 14, 14)
        pos_embed2d = torch.nn.functional.interpolate(
            pos_embed2d,
            size=(self.fixed_size // 16, self.fixed_size // 16),
            mode="bilinear",
            align_corners=True,
        )
        pos_embed = pos_embed2d.reshape(
            1, 768, self.fixed_size // 16 * self.fixed_size // 16
        ).transpose(1, 2)

        x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.det:
            x = self._pos_embed_interp(x)
        else:
            x = self._pos_embed(x)

        z = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.dense and i in [2, 5, 8, 11]:
                z.append(x)

        return z if self.dense else self.norm(x)

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        if self.dense:
            x = self.decoder(x)
        elif not self.det:
            if self.out_token == "cls":
                x = x[:, 0]
            elif self.out_token == "spatial":
                x = x[:, 1:].mean(1)
            if self.head_bool:
                x = self.lin_head(x)
        else:
            x = self.fpn(x)
        return x


class ViT_from_MAE(models_mae.MaskedAutoencoderViT):
    def __init__(
        self,
        weight_path,
        head,
        num_classes,
        frozen,
        dense,
        det,
        fixed_size,
        embed_dim,
        depth,
        num_heads,
        out_token,
    ):
        super(ViT_from_MAE, self).__init__(
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        if det:
            for i in [0, 1, 3, 4, 6, 7, 9, 10]:
                self.blocks[i].attn = WindowedAttention(
                    dim=embed_dim, num_heads=num_heads, qkv_bias=True
                )
        if weight_path is not None:
            weights = torch.load(weight_path, map_location="cpu")["model"]
            self.load_my_state_dict(weights)
        del self.mask_token
        del self.decoder_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred

        self.head = head
        if head:
            self.lin_head = nn.Linear(embed_dim, num_classes)
        self.frozen = frozen
        self.dense = dense
        self.det = det
        if dense:
            self.decoder = DPT_decoder(num_classes=num_classes, dense=dense)
        if det:
            self.fixed_size = fixed_size
            self.fpn = ViTDet_FPN()
            self.out_channels = 256
            self.patch_embed.img_size = (fixed_size, fixed_size)
            del self.cls_token
        self.out_token = out_token

    def load_my_state_dict(self, state_dict):
        i = 0
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
            i += 1
        print(f"Successfully loaded params for {i} items")

    def forward_encoder(self, x):
        x = self.patch_embed(x)

        if self.det:
            pos_embed2d = (
                self.pos_embed[:, 1:, :].transpose(1, 2).reshape(1, 768, 14, 14)
            )
            pos_embed2d = torch.nn.functional.interpolate(
                pos_embed2d,
                size=(self.fixed_size // 16, self.fixed_size // 16),
                mode="bilinear",
                align_corners=True,
            )
            pos_embed = pos_embed2d.reshape(
                1, 768, self.fixed_size // 16 * self.fixed_size // 16
            ).transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.pos_embed[:, 1:, :]
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        z = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.dense and i in [2, 5, 8, 11]:
                z.append(x)

        return z if self.dense else self.norm(x)

    def forward(self, imgs):
        if self.frozen:
            with torch.no_grad():
                x = self.forward_encoder(imgs)
        else:
            x = self.forward_encoder(imgs)
        if self.dense:
            x = self.decoder(x)
        elif not self.det:
            if self.out_token == "cls":
                x = x[:, 0]
            elif self.out_token == "spatial":
                x = x[:, 1:].mean(1)
            if self.head:
                x = self.lin_head(x)
        else:
            x = self.fpn(x)
        return x


class ViT_from_MoCoV3(vits.VisionTransformerMoCo):
    def __init__(
        self,
        weight_path,
        head,
        num_classes,
        frozen,
        dense,
        det,
        fixed_size,
        embed_dim,
        out_token,
    ):
        super(ViT_from_MoCoV3, self).__init__(
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=4096,
        )
        if det:
            for i in [0, 1, 3, 4, 6, 7, 9, 10]:
                self.blocks[i].attn = WindowedAttention(
                    dim=embed_dim, num_heads=12, qkv_bias=True
                )
        self.head = nn.Identity()
        if weight_path is not None:
            weights = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(weights)
        self.patch_embed.proj.weight.requires_grad = True
        self.patch_embed.proj.bias.requires_grad = True

        self.head_bool = head
        if head:
            self.lin_head = nn.Linear(embed_dim, num_classes)
        self.frozen = frozen
        self.dense = dense
        self.det = det

        if dense:
            self.decoder = DPT_decoder(num_classes=num_classes, dense=dense)
        if det:
            self.fixed_size = fixed_size
            self.fpn = ViTDet_FPN()
            self.out_channels = 256
            self.patch_embed.img_size = (fixed_size, fixed_size)
            del self.cls_token
        self.out_token = out_token

    def _pos_embed_interp(self, x):
        pos_embed2d = self.pos_embed[:, 1:, :].transpose(1, 2).reshape(1, 768, 14, 14)
        pos_embed2d = torch.nn.functional.interpolate(
            pos_embed2d,
            size=(self.fixed_size // 16, self.fixed_size // 16),
            mode="bilinear",
            align_corners=True,
        )
        pos_embed = pos_embed2d.reshape(
            1, 768, self.fixed_size // 16 * self.fixed_size // 16
        ).transpose(1, 2)

        x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.det:
            x = self._pos_embed_interp(x)
        else:
            x = self._pos_embed(x)

        z = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.dense and i in [2, 5, 8, 11]:
                z.append(x)

        return z if self.dense else self.norm(x)

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        if self.dense:
            x = self.decoder(x)
        elif not self.det:
            if self.out_token == "cls":
                x = x[:, 0]
            elif self.out_token == "spatial":
                x = x[:, 1:].mean(1)
            if self.head_bool:
                x = self.lin_head(x)
        else:
            x = self.fpn(x)

        return x

