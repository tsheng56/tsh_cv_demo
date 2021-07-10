import torch
import torchvision.ops as ops
from convert.torch2caffe.utils import trans_log, Rp
from convert.torch2caffe.caffe_layer import Layer

# ----- for torchvision operation --------

def _roi_pool(raw, input, rois, spatial_scale, output_size_h, output_size_w):

    print('\troi_pool bottom blob: {}, dim:{}'.format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))

    # rois input layer
    roi_name = trans_log.add_layer(name='data')
    roi_top_blobs = trans_log.add_blobs([rois], name='data')
    input_layer = Layer(name=roi_name, type='Input', top=roi_top_blobs)
    input_layer.input_param(rois.shape)
    trans_log.cnet.add_layer(input_layer, after=trans_log.cnet.last_input_name)
    trans_log.cnet.last_input_name = roi_name

    # rois pool layer
    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]
    x, _ = raw(input, rois, spatial_scale, output_size_h, output_size_w)
    name = trans_log.add_layer(name='roi_pool')

    top_blobs = trans_log.add_blobs([x], name='roi_blob')

    layer = Layer(name=name,
                  type='ROIPooling',
                  bottom=bottom_blobs,
                  top=top_blobs)

    layer.roi_pooling_param(output_size_h, output_size_w, spatial_scale)

    trans_log.cnet.add_layer(layer)
    return x, _


def _roi_align(raw,
               input,
               rois,
               spatial_scale,
               output_size_h,
               output_size_w,
               sampling_ratio=-1,
               aligned=False):

    print('\troi_align bottom blob: {}, dim:{}'.format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))

    # rois input layer
    roi_name = trans_log.add_layer(name='data')
    roi_top_blobs = trans_log.add_blobs([rois], name='data')
    input_layer = Layer(name=roi_name, type='Input', top=roi_top_blobs)
    input_layer.input_param(rois.shape)
    trans_log.cnet.add_layer(input_layer, after=trans_log.cnet.last_input_name)
    trans_log.cnet.last_input_name = roi_name

    # rois pool layer
    bottom_blobs = [trans_log.blobs(input), trans_log.blobs(rois)]
    x = raw(input, rois, spatial_scale, output_size_h, output_size_w, sampling_ratio, aligned)
    name = trans_log.add_layer(name='roi_align')

    top_blobs = trans_log.add_blobs([x], name='roi_blob')

    layer = Layer(name=name,
                  type='ROIAlign',
                  bottom=bottom_blobs,
                  top=top_blobs)

    layer.roi_align_param(output_size_h, output_size_w, spatial_scale)

    trans_log.cnet.add_layer(layer)
    return x


def replace():

    torch.ops.torchvision.roi_pool = Rp(torch.ops.torchvision.roi_pool,
                                        _roi_pool)
    torch.ops.torchvision.roi_align = Rp(torch.ops.torchvision.roi_align,
                                         _roi_align)
