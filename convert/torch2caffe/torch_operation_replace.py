import torch
import numpy as np

from convert.torch2caffe.caffe import caffe_pb2 as pb
from convert.torch2caffe.utils import trans_log, Rp
from convert.torch2caffe.caffe_layer import Layer

# ----- for torch.functional --------

def _split(raw, tensor, split_size, dim=0):
    # split in pytorch is slice in caffe
    x = raw(tensor, split_size, dim)
    layer_name = trans_log.add_layer('split')
    top_blobs = trans_log.add_blobs(x, name='split_blob')
    layer = Layer(name=layer_name,
                  type='Slice',
                  bottom=[trans_log.blobs(tensor)],
                  top=top_blobs)
    slice_num = int(np.floor(tensor.size()[dim] / split_size))
    slice_param = pb.SliceParameter(
        axis=dim, slice_point=[split_size * i for i in range(1, slice_num)])
    layer.param.slice_param.CopyFrom(slice_param)
    trans_log.cnet.add_layer(layer)
    return x


def _max(raw, *args):
    x = raw(*args)
    if len(args) == 1:
        # TODO max in one tensor
        assert NotImplementedError
    else:
        bottom_blobs = []
        for arg in args:
            bottom_blobs.append(trans_log.blobs(arg))
        layer_name = trans_log.add_layer(name='max')
        top_blobs = trans_log.add_blobs([x], name='max_blob')
        layer = Layer(name=layer_name,
                      type='Eltwise',
                      bottom=bottom_blobs,
                      top=top_blobs)
        layer.param.eltwise_param.operation = 2
        trans_log.cnet.add_layer(layer)
    return x


def _cat(raw, inputs, dim=0):
    x = raw(inputs, dim)
    bottom_blobs = []
    for input_idx, input in enumerate(inputs):
        bottom_blobs.append(trans_log.blobs(input))
        print('\tcat bottom blob {}: {}, dim:{}'.format(input_idx+1, trans_log.blobs(input), [int(dim) for dim in input.shape]))
        
    layer_name = trans_log.add_layer(name='cat')
    top_blobs = trans_log.add_blobs([x], name='cat_blob')
    layer = Layer(name=layer_name,
                  type='Concat',
                  bottom=bottom_blobs,
                  top=top_blobs)
    layer.param.concat_param.axis = dim
    trans_log.cnet.add_layer(layer)
    return x


def _div(raw, inputs, inputs2):
    x = raw(inputs, inputs2)
    trans_log.add_blobs([x], name='div_blob')
    return x

def _ones(raw, shape):
    x = raw(shape)

    layer_name = trans_log.add_layer(name='data')
    top_blobs = trans_log.add_blobs([x], name='data')

    layer = Layer(name=layer_name,
                  type='Input',
                  top=top_blobs)

    layer.input_param(shape)
    trans_log.cnet.add_layer(layer, after=trans_log.cnet.last_input_name)
    trans_log.cnet.last_input_name = layer_name

    return x

def _ones_like(raw, input):
    x = raw(input)

    layer_name = trans_log.add_layer(name='data')
    top_blobs = trans_log.add_blobs([x], name='data')

    layer = Layer(name=layer_name,
                  type='Input',
                  top=top_blobs)

    layer.input_param(input.shape)
    trans_log.cnet.add_layer(layer, after=trans_log.cnet.last_input_name)
    trans_log.cnet.last_input_name = layer_name

    return x


def replace():

    torch.split = Rp(torch.split, _split)
    torch.max = Rp(torch.max, _max)
    torch.cat = Rp(torch.cat, _cat)
    torch.div = Rp(torch.div, _div)

    torch.ones = Rp(torch.ones, _ones)
    torch.zeros = Rp(torch.zeros, _ones)
    torch.ones_like = Rp(torch.ones_like, _ones_like)
    torch.zeros_like = Rp(torch.zeros_like, _ones_like)

    

