import os
import argparse
import torch
import torch.nn as nn

from convert.torch2caffe.pytorch_to_caffe import trans_net

from torchvision.ops import RoIAlign, RoIPool

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple_name", type=int, default=1)
    parser.add_argument("--pre", type=str, default='test')
    parser.add_argument("--draw", type=int, default=1)
    opt = parser.parse_args()
    
    return opt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pool1 = RoIAlign(spatial_scale=0.25, output_size=7, sampling_ratio=-1, aligned=False)
        self.pool2 = RoIPool(spatial_scale=0.5, output_size=7)
        self.up1 = torch.nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.up2 = torch.nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False)

        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)


    
    def forward(self, x):
        box = torch.FloatTensor([[0,0,0,32,32]])
        roi1 = self.pool1(x, box)
        kk = torch.ones([3,3,3])
        roi2 = self.pool2(x, box)
        bb = torch.ones_like(kk)

        y = torch.ones([1,3,256,256])
        
        y2 = self.up2(y)
        y1 = self.up1(y2)

        # z = torch.ones([1,3,257,257])
        # z1 = self.pool3(z)
        # z2 = self.pool4(z1)

        # z2 = z2.flatten(1)

        # z2 = torch.nn.ReLU()(z2)

        return None



if __name__ == '__main__':
    opt = get_args()

    model = Model()
    input_size = [1,3, 800, 1333]

    trans_net(model, input_size, ckpt='yolodet.pth', simple_name=opt.simple_name, output_blob_pre=opt.pre, draw=opt.draw)