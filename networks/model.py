"""Dual-stream network for joint csPCa classification and segmentation."""
import os
import torch
from torch import nn
import torch.nn.functional as F


class SEblock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = x * y.expand(x.size())
        return y


class SEblock2D(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEblock2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, s, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, s)
        y = self.fc(y).view(b, c, s, 1, 1)
        y = x * y.expand(x.size())
        return y


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels * BasicBlock.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.InstanceNorm3d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        super(Upsample, self).__init__()
        self.ConvTrans = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.norm = nn.InstanceNorm3d(
            out_channels,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=False,
        )

    def forward(self, x):
        return self.norm(self.ConvTrans(x))


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class ConvDropoutNormReLU(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        is_dropout=False,
    ):
        super(ConvDropoutNormReLU, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
        )
        self.norm = nn.InstanceNorm3d(
            out_channels,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=False,
        )
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if is_dropout:
            self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        else:
            self.dropout = Identity()
        self.all = nn.Sequential(self.conv, self.dropout, self.norm, self.nonlin)

    def forward(self, x):
        return self.all(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3, 3),
        conv_stride=(1, 1, 1),
        is_dropout=False,
    ):
        super(DecoderBlock, self).__init__()
        self.conv = ConvDropoutNormReLU(in_channels, out_channels, kernel_size, conv_stride, is_dropout)

    def forward(self, x):
        return self.conv(x)


class ProstateMMTNet(nn.Module):
    """Prostate Multi-Modality Multi-Task Network for csPCa diagnosis.

    Combines B-mode and strain elastography inputs for joint classification
    and segmentation. Multi-scale prototype-guided learning is implemented
    by the training script.
    """

    def __init__(self, num_channels1=1, num_channels2=1, num_classes=2):
        super(ProstateMMTNet, self).__init__()
        layers1 = [3, 4, 6, 3]
        layers2 = [3, 4, 6, 3]
        encoder_block = BasicBlock
        self.in_channels = 64
        self.conv1_1 = nn.Conv3d(
            num_channels1,
            64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1_1 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.layer1_1 = self._make_layer(encoder_block, 64, layers1[0], stride=1)
        self.layer1_2 = self._make_layer(encoder_block, 128, layers1[1], stride=2)
        self.layer1_3 = self._make_layer(encoder_block, 256, layers1[2], stride=2)
        self.layer1_4 = self._make_layer(encoder_block, 512, layers1[3], stride=2)
        self.in_channels = 64
        self.conv2_1 = nn.Conv3d(
            num_channels2,
            64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn2_1 = nn.InstanceNorm3d(64)
        self.layer2_1 = self._make_layer(encoder_block, 64, layers2[0], stride=1)
        self.layer2_2 = self._make_layer(encoder_block, 128, layers2[1], stride=2)
        self.layer2_3 = self._make_layer(encoder_block, 256, layers2[2], stride=2)
        self.layer2_4 = self._make_layer(encoder_block, 512, layers2[3], stride=2)
        self.FF0 = FeatureFusion(64, 72, 72, 100, 16, 16, 16)
        self.FF1 = FeatureFusion(64, 36, 36, 50, 8, 8, 16)
        self.FF2 = FeatureFusion(128, 18, 18, 25, 4, 4, 8)
        self.FF3 = FeatureFusion(256, 9, 9, 13, 2, 2, 4)
        self.FF4 = FeatureFusion(512, 5, 5, 7, 2, 2, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.smooth1 = nn.Conv3d(256, 256, 3, 1)
        self.smooth2 = nn.Conv3d(128, 128, 3, 1)
        self.fc_fusion = nn.Linear(896, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.up1 = Upsample(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up2 = Upsample(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up3 = Upsample(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up4 = Upsample(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.decoder4 = DecoderBlock(128, 64)
        self.out_conv = nn.Conv3d(
            64,
            num_classes,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=False,
        )

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, 'number of blocks should be greater than zero'
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def get_model_py_path(self):
        return os.path.abspath(__file__)

    def forward(self, x1, x2, return_feats=False):
        bs, _, _, _, _ = x1.size()
        layer1_0_out = self.relu(self.bn1_1(self.conv1_1(x1)))
        layer2_0_out = self.relu(self.bn2_1(self.conv2_1(x2)))
        fusion_out0, fusion_ca0 = self.FF0(layer1_0_out, layer2_0_out)
        layer1_1_in = self.max_pool(layer1_0_out + fusion_out0)
        layer2_1_in = self.max_pool(layer2_0_out + fusion_ca0)
        layer1_1_out = self.layer1_1(layer1_1_in)
        layer2_1_out = self.layer2_1(layer2_1_in)
        fusion_out1, fusion_ca1 = self.FF1(layer1_1_out, layer2_1_out)
        layer1_2_in = layer1_1_out + fusion_out1
        layer2_2_in = layer2_1_out + fusion_ca1
        layer1_2_out = self.layer1_2(layer1_2_in)
        layer2_2_out = self.layer2_2(layer2_2_in)
        fusion_out2, fusion_ca2 = self.FF2(layer1_2_out, layer2_2_out)
        layer1_3_in = layer1_2_out + fusion_out2
        layer2_3_in = layer2_2_out + fusion_ca2
        layer1_3_out = self.layer1_3(layer1_3_in)
        layer2_3_out = self.layer2_3(layer2_3_in)
        fusion_out3, fusion_ca3 = self.FF3(layer1_3_out, layer2_3_out)
        layer1_4_in = layer1_3_out + fusion_out3
        layer2_4_in = layer2_3_out + fusion_ca3
        layer1_4_out = self.layer1_4(layer1_4_in)
        layer2_4_out = self.layer2_4(layer2_4_in)
        fusion_out4, _ = self.FF4(layer1_4_out, layer2_4_out)
        out_fusion_up = self.up1(fusion_out4)
        if out_fusion_up.size() != fusion_out3.size():
            out_fusion_up = F.interpolate(
                out_fusion_up,
                size=[fusion_out3.size(2), fusion_out3.size(3), fusion_out3.size(4)],
                mode='trilinear',
            )
        out4_out5 = torch.cat((fusion_out3, out_fusion_up), dim=1)
        y1 = self.decoder1(out4_out5)
        y1_up = self.up2(y1)
        if y1_up.size() != fusion_out2.size():
            y1_up = F.interpolate(
                y1_up,
                size=[fusion_out2.size(2), fusion_out2.size(3), fusion_out2.size(4)],
                mode='trilinear',
            )
        x3_y1 = torch.cat((fusion_out2, y1_up), dim=1)
        y2 = self.decoder2(x3_y1)
        y2_up = self.up3(y2)
        x2_y2 = torch.cat((fusion_out1, y2_up), dim=1)
        y3 = self.decoder3(x2_y2)
        y3_up = self.up4(y3)
        x1_y3 = torch.cat((fusion_out0, y3_up), dim=1)
        y4 = self.decoder4(x1_y3)
        out_seg = self.out_conv(y4)
        feats = [y3, y4]
        y1 = self.smooth1(y1)
        y2 = self.smooth2(y2)
        out_fusion = self.avg_pool(fusion_out4).view(bs, -1)
        y1 = self.avg_pool(y1).view(bs, -1)
        y2 = self.avg_pool(y2).view(bs, -1)
        out_fusion = torch.cat((out_fusion, y1, y2), dim=1)
        out_cls_fusion = self.fc_fusion(out_fusion)
        if return_feats:
            return [out_cls_fusion, out_seg, feats]
        else:
            return [out_cls_fusion, out_seg]


class CrossAtt(nn.Module):

    def __init__(self, in_channels):
        super(CrossAtt, self).__init__()
        self.conv1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1)
        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, f1, f2):
        avg_feature = f1 * f2
        attention_map1 = self.bn1(self.conv1(torch.cat((f1, avg_feature), dim=1)))
        attention_map2 = self.bn2(self.conv2(torch.cat((f2, avg_feature), dim=1)))
        attention_map_sm1 = torch.exp(attention_map1) / (torch.exp(attention_map1) + torch.exp(attention_map2))
        attention_map_sm2 = torch.exp(attention_map2) / (torch.exp(attention_map1) + torch.exp(attention_map2))
        attention_feature1 = f1 * attention_map_sm1
        attention_feature2 = f2 * attention_map_sm2
        fusion_attention_feature = attention_feature1 + attention_feature2
        return fusion_attention_feature


class SelfAtt(nn.Module):

    def __init__(self, in_channels, x_s, y_s, z_s, r1, r2, r3):
        super(SelfAtt, self).__init__()
        self.sex = SEblock2D(x_s, r1)
        self.sey = SEblock2D(y_s, r2)
        self.sez = SEblock2D(z_s, r3)
        self.se = SEblock(in_channels)

    def forward(self, f1):
        f1_x = self.sex(f1)
        f1_y = self.sey(f1.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
        f1_z = self.sez(f1.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)
        out = self.se(f1 + f1_x + f1_y + f1_z)
        return out


class FeatureFusion(nn.Module):

    def __init__(self, in_channels, x_s, y_s, z_s, r_x, r_y, r_z):
        super(FeatureFusion, self).__init__()
        self.dam = SelfAtt(in_channels, x_s, y_s, z_s, r_x, r_y, r_z)
        self.asam = CrossAtt(in_channels)
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(),
        )

    def forward(self, f1, f2):
        f1 = self.dam(f1)
        fusion = self.asam(f1, f2)
        f1 = torch.cat((f1, fusion), dim=1)
        f1 = self.conv_block1(f1)
        f2 = torch.cat((f2, fusion), dim=1)
        f2 = self.conv_block2(f2)
        return (f1, f2)
