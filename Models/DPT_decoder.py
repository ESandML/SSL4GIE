import torch
import torch.nn as nn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT_decoder(nn.Module):
    def __init__(
        self,
        num_classes=1,
        dense="seg",
        vit_features=768,
        features=[96, 192, 384, 768],
        fusion_features=256,
        use_readout="ignore",
        size=[224, 224],
        patch_size=[16, 16],
    ):
        super().__init__()
        readout_oper = get_readout_oper(vit_features, features, use_readout)
        self.act_postprocess11 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
        )
        self.act_postprocess12 = nn.Sequential(
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess21 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
        )
        self.act_postprocess22 = nn.Sequential(
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess31 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
        )
        self.act_postprocess32 = nn.Sequential(
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.act_postprocess41 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
        )
        self.act_postprocess42 = nn.Sequential(
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.layer1_rn = nn.Conv2d(
            features[0],
            fusion_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
        )
        self.layer2_rn = nn.Conv2d(
            features[1],
            fusion_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
        )
        self.layer3_rn = nn.Conv2d(
            features[2],
            fusion_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
        )
        self.layer4_rn = nn.Conv2d(
            features[3],
            fusion_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
        )

        self.unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        size[0] // patch_size[0],
                        size[1] // patch_size[1],
                    ]
                ),
            )
        )

        use_bn = dense == "seg"

        self.refinenet1 = _make_fusion_block(fusion_features, use_bn)
        self.refinenet2 = _make_fusion_block(fusion_features, use_bn)
        self.refinenet3 = _make_fusion_block(fusion_features, use_bn)
        self.refinenet4 = _make_fusion_block(fusion_features, use_bn)

        if dense == "depth":
            self.output_conv = nn.Sequential(
                nn.Conv2d(
                    fusion_features,
                    fusion_features // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(fusion_features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )
        else:
            self.output_conv = nn.Sequential(
                nn.Conv2d(
                    fusion_features,
                    fusion_features,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(fusion_features),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(fusion_features, num_classes, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )

        self.dense = dense

    def forward_skip(self, activations):

        layer_1 = self.act_postprocess11(activations[0])
        layer_2 = self.act_postprocess21(activations[1])
        layer_3 = self.act_postprocess31(activations[2])
        layer_4 = self.act_postprocess41(activations[3])

        if layer_1.ndim == 3:
            layer_1 = self.unflatten(layer_1).contiguous()
        if layer_2.ndim == 3:
            layer_2 = self.unflatten(layer_2).contiguous()
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3).contiguous()
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4).contiguous()

        layer_1 = self.act_postprocess12(layer_1)
        layer_2 = self.act_postprocess22(layer_2)
        layer_3 = self.act_postprocess32(layer_3)
        layer_4 = self.act_postprocess42(layer_4)

        layer_1 = self.layer1_rn(layer_1)
        layer_2 = self.layer2_rn(layer_2)
        layer_3 = self.layer3_rn(layer_3)
        layer_4 = self.layer4_rn(layer_4)

        return [layer_1, layer_2, layer_3, layer_4]

    def forward(self, activations):
        activations = self.forward_skip(activations)

        path_4 = self.refinenet4(activations[3])
        path_3 = self.refinenet3(path_4, activations[2])
        path_2 = self.refinenet2(path_3, activations[1])
        path_1 = self.refinenet1(path_2, activations[0])

        out = self.output_conv(path_1)

        return out

