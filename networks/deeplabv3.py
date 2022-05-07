import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder, build_boundarydecoder
from networks.backbone import build_backbone
import numpy as np
from torch.distributions.uniform import Uniform



class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, classmates=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if classmates:
            self.boundarydecoder1 = build_boundarydecoder(num_classes, backbone, BatchNorm)
            self.boundarydecoder2 = build_boundarydecoder(num_classes, backbone, BatchNorm)
            self.uni_dist = Uniform(-0.3, 0.3)


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, classmates=False):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        feature = x
        x1 = self.decoder(x, low_level_feat)
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)

        if classmates:
            # perturbation
            x_dropout = self.feature_dropout(x)
            x_noise = self.feature_based_noise(x)

            x2 = self.boundarydecoder1(x_dropout, low_level_feat)
            x2 = F.interpolate(x2, size=input.size()[2:], mode='bilinear', align_corners=True)

            x3 = self.boundarydecoder2(x_noise, low_level_feat)
            x3 = F.interpolate(x3, size=input.size()[2:], mode='bilinear', align_corners=True)

            return x1, x2, x3
        else:
            return x1


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:

                            yield p

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise



if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())