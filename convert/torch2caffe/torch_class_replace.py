import torch
from torchvision.ops import RoIAlign, RoIPool
from convert.torch2caffe.utils import trans_log, Rp
from convert.torch2caffe.caffe_layer import Layer

# ----- for torch class and custom class--------
# class not contain torch func, torch opt, tensor opt, torchvision opt

def _roi_pool_forward(self, input, rois):
    print('roi pool')
    print('\troi_pool bottom blob: {}, dim:{}'.format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))
    
    x = Raw.roi_pool_forward(self, input, rois)

    # rois input layer
    roi_name = trans_log.add_layer(name='data')
    roi_top_blobs = trans_log.add_blobs([rois], name='data')
    input_layer = Layer(name=roi_name, type='Input', top=roi_top_blobs)
    input_layer.input_param(rois.shape)
    trans_log.cnet.add_layer(input_layer, after=trans_log.cnet.last_input_name)
    trans_log.cnet.last_input_name = roi_name

    # rois pool layer
    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]

    name = trans_log.add_layer(name='roi_pool')

    top_blobs = trans_log.add_blobs([x], name='roi_blob')

    layer = Layer(name=name,
                  type='ROIPooling',
                  bottom=bottom_blobs,
                  top=top_blobs)
    if isinstance(self.output_size, int):
        self.output_size = [self.output_size, self.output_size]
    layer.roi_pooling_param(self.output_size[0], self.output_size[1], self.spatial_scale)

    trans_log.cnet.add_layer(layer)
    return x


def _roi_align_forward(self, input, rois):
    print('roi align')
    print('\troi_align bottom blob: {}, dim:{}'.format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))

    x = Raw.roi_align_forward(self, input, rois)

    # rois input layer
    roi_name = trans_log.add_layer(name='data')
    roi_top_blobs = trans_log.add_blobs([rois], name='data')
    input_layer = Layer(name=roi_name, type='Input', top=roi_top_blobs)
    input_layer.input_param(rois.shape)
    trans_log.cnet.add_layer(input_layer, after=trans_log.cnet.last_input_name)
    trans_log.cnet.last_input_name = roi_name

    # rois pool layer
    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]

    name = trans_log.add_layer(name='roi_align')

    top_blobs = trans_log.add_blobs([x], name='roi_blob')

    layer = Layer(name=name,
                  type='ROIAlign',
                  bottom=bottom_blobs,
                  top=top_blobs)

    if isinstance(self.output_size, int):
        self.output_size = [self.output_size, self.output_size]
    layer.roi_align_param(self.output_size[0], self.output_size[1], self.spatial_scale)

    trans_log.cnet.add_layer(layer)
    return x


class Raw(object):
    def __init__(self):
        self.type = 'torch.Tensor'


def replace():
    pass

    # replace in torchvision_operation_replace
    # setattr(Raw, 'roi_pool_forward', RoIPool.forward)
    # RoIPool.forward = _roi_pool_forward

    # replace in torchvision_operation_replace
    # setattr(Raw, 'roi_align_forward', RoIAlign.forward)
    # RoIAlign.forward = _roi_align_forward