import os
import torch
from convert.torch2caffe.utils import trans_log
from convert.torch2caffe.torch_function_replace import replace as replace_torch_function
from convert.torch2caffe.torch_operation_replace import replace as replace_torch_operation
from convert.torch2caffe.torch_tensor_operation_replace import replace as replace_torch_tensor_operation
from convert.torch2caffe.torchvision_operation_replace import replace as replace_torchvision_operation
from convert.torch2caffe.torch_class_replace import replace as replace_torch_class
from convert.torch2caffe.draw_net import draw_caffe_net

"""
How to support a new layer type:
 layer_name = trans_log.add_layer(layer_type_name)
 top_blobs = trans_log.add_blobs(<output of that layer>)
 layer = Layer(xxx)
 <set layer parameters>
 [<layer.add_data(*datas)>]
 trans_log.cnet.add_layer(layer)
 
Please MUTE the inplace operations to avoid not find in graph

注意：只有torch.nn.functional中的函数才能转换为caffe中的层
"""

def trans_net(net, input_size, ckpt='yolodet.pth', merge_bn=True, simple_name=True, output_blob_pre='', draw=False):
    print('Starting Transform, This will take a while')
    net.eval()
    # input_var = torch.ones(input_size)

    trans_log.simple_name = simple_name
    trans_log.output_blob_pre = output_blob_pre
    trans_log.ckpt = ckpt
    trans_log.merge_bn = merge_bn

    # trans_log.init([input_var]) # init input_var and cnet
    trans_log.init() # only init cnet

    trans_log.cnet.net.name = ckpt
    # trans_log.cnet.net.input.extend([trans_log.blobs(input_var)])
    # trans_log.cnet.net.input_dim.extend(input_var.size())

    trans_log.NET_INITTED = True

    for name, layer in net.named_modules():
        trans_log.layer_names[layer] = name
    # print("torch ops name:", trans_log.layer_names)
    
    replace_torch_function()
    replace_torch_operation()
    replace_torch_tensor_operation()
    replace_torchvision_operation()
    replace_torch_class()

    input_var = torch.ones(input_size)
    out = net.forward(input_var)
    print('Transform Completed\n')

    prototxt, caffemode = trans_log.cnet.save()

    if draw:
        output_image_file = os.path.splitext(prototxt)[0] + '.jpg'
        draw_caffe_net(input_net_proto_file=prototxt, output_image_file=output_image_file, rankdir='LR', phase=None, display_lrm=None)
