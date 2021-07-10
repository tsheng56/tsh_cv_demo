import torch
import torch.nn as nn
from torch.nn.functional import pad

from detect.models.torch_common import Conv, Detect


class Model(nn.Module):
    def __init__(self, nc, anchors, ch):
        super(Model, self).__init__()
        print("find model_torch: Test !!")

        self.b = Conv(ch, 32, k=3, s=4, p=1, g=1, act=True)
        self.b1 = Conv(32, 64, k=3, s=2, p=1, g=1, act=True)
        self.b2 = Conv(64, 128, k=3, s=2, p=1, g=1, act=True)
        self.b3 = Conv(128, 256, k=3, s=2, p=1, g=1, act=True)

        self.detect = Detect(nc=nc, anchors=anchors, ch=[64, 128, 256])
    
    def forward(self, x):
        
        b1 = self.b1(self.b(x))
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        return self.detect([b1, b2, b3])