import torch
import torch.nn as nn
import torch.nn.functional as F


class HRFPN(nn.Module):

    def __init__(self, cfg):
        super(HRFPN, self).__init__()

        config = cfg.MODEL.NECK
        self.pooling_type = config.POOLING
        self.num_outs = config.NUM_OUTS  ##等于5，为什么呢？
        self.in_channels = config.IN_CHANNELS  ##4个分支、4个channel，IN_CHANNELS:[18, 36, 72, 144]
        self.out_channels = config.OUT_CHANNELS  ##等于256
        self.num_ins = len(self.in_channels)  ##等于4

        assert isinstance(self.in_channels, (list, tuple))

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(self.in_channels),
                      out_channels=self.out_channels,
                      kernel_size=1),
        )
        self.fpn_conv = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_conv.append(nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1
            ))
        if self.pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins  ##4个分支、4个channel，IN_CHANNELS:[18, 36, 72, 144]
        outs = [inputs[0]]  ##分支一的feature map的分辨率
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))  ##上采样到分支一的feature map的分辨率
        out = torch.cat(outs, dim=1)  ##在Channel维度进行融合(通道叠加)
        out = self.reduction_conv(out)  ##用1x1卷积进行通道变换(270->256)
        outs = [out]
        for i in range(1, self.num_outs):  ##self.num_outs等于5
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))  ##用池化层生成特征金字塔的多层
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_conv[i](outs[i]))  ##用卷积层提取特征金字塔的各个层的特征
        return tuple(outputs)
