# YOLOv5 YOLO-specific modules
import sys
import math
import logging
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from detect.models.detect_module import Detect
from detect.models.torch_common import autoShape, NMS, Conv
from detect.utils.autoanchor import check_anchor_order
from detect.utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, copy_attr

from detect.models.model_torch.torch_test import Model as torch_test

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Model(nn.Module):
    def __init__(self, version, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()

        # logger.info(f'\n\t>> Create Torch Model: torch_{cfg}\n')
        try:
            exec(f"from detect.models.model_torch.{cfg} import Model")
            logger.info(f"from detect.models.model_torch.{cfg} import Model\n")
        except:
            exec(f"from detect.models.model_torch.torch_{cfg} import Model")
            logger.info(f"from detect.models.model_torch.torch_{cfg} import Model\n")

        self.model = eval('Model')(nc, anchors, ch)

        logger.info(f'model_torch anchors={anchors}')
        logger.info(f"model_troch nc={nc}")

        self.names = [str(i) for i in range(nc)]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model.detect  # Detect()
        m.version = version
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            # check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        t = time_synchronized()
        x = self.model(x)
        dt = (time_synchronized() - t) * 100

        if profile:
            logger.info('%.1fms total' % dt)

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model.detect  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        try:
            present = type(self.model.nms) is NMS  # last layer is NMS
            if mode and not present:
                logger.info('Adding NMS... ')
                m = NMS()  # module
                m.f = -1  # from
                m.i = self.model.nms.i + 1  # index
                self.model.add_module(name='%s' % m.i, module=m)  # add
                self.eval()
            elif not mode and present:
                logger.info('Removing NMS... ')
                self.model.nms = None  # remove
        except Exception as e:
            raise Exception('torch model has no NMS')
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

