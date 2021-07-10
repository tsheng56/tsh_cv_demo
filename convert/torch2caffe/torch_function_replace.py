import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from convert.torch2caffe.caffe import caffe_pb2 as pb
from convert.torch2caffe.utils import trans_log, Rp
from convert.torch2caffe.caffe_layer import Layer

# ----- for torch.nn.functional --------

def _conv2d(raw,
            input,
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1):

    print('\tconv bottom blob: {}, dim:{}'.format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    name = trans_log.add_layer(name='conv')

    top_blobs = trans_log.add_blobs([x], name='conv_blob')

    layer = Layer(name=name,
                  type='Convolution',
                  bottom=[trans_log.blobs(input)],
                  top=top_blobs)

    layer.conv_param(x.size()[1],
                     weight.size()[2:],
                     stride=_pair(stride),
                     pad=_pair(padding),
                     dilation=_pair(dilation),
                     bias_term=bias is not None,
                     groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    
    trans_log.cnet.add_layer(layer)
    return x


def _conv_transpose2d(raw,
                      input,
                      weight,
                      bias=None,
                      stride=1,
                      padding=0,
                      output_padding=0,
                      groups=1,
                      dilation=1):
    x = raw(input, weight, bias, stride, padding, output_padding, groups,
            dilation)
    name = trans_log.add_layer(name='conv_transpose')
    trans_log.add_blobs([x], name='conv_transpose_blob')
    layer = Layer(name=name,
                  type='Deconvolution',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])
    layer.conv_param(x.size()[1],
                     weight.size()[2:],
                     stride=_pair(stride),
                     pad=_pair(padding),
                     dilation=_pair(dilation),
                     bias_term=bias is not None,
                     groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())

    trans_log.cnet.add_layer(layer)
    return x


def _linear(raw, input, weight, bias=None):
    x = raw(input, weight, bias)
    layer_name = trans_log.add_layer(name='fc')
    top_blobs = trans_log.add_blobs([x], name='fc_blob')
    layer = Layer(name=layer_name,
                  type='InnerProduct',
                  bottom=[trans_log.blobs(input)],
                  top=top_blobs)
    layer.fc_param(x.size()[1], has_bias=bias is not None)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.add_data(weight.cpu().data.numpy())

    trans_log.cnet.add_layer(layer)
    return x


def _pool(type, raw, input, x, kernel_size, stride, padding, ceil_mode):
    # TODO dilation,ceil_mode,return indices

    print('\tpool bottom blob: {}, dim:{}'.format(
        trans_log.blobs(input), [int(dim) for dim in input.shape]))

    layer_name = trans_log.add_layer(name='{}_pool'.format(type))
    top_blobs = trans_log.add_blobs([x], name='{}_pool_blob'.format(type))
    layer = Layer(name=layer_name,
                  type='Pooling',
                  bottom=[trans_log.blobs(input)],
                  top=top_blobs)
    # TODO w,h different kernel, stride and padding
    # processing ceil mode
    layer.pool_param(kernel_size=kernel_size,
                     stride=kernel_size if stride is None else stride,
                     pad=padding,
                     type=type.upper(),
                     ceil_mode=ceil_mode)

    trans_log.cnet.add_layer(layer)
    if ceil_mode == False and stride is not None:
        oheight = (input.size()[2] - _pair(kernel_size)[0] +
                   2 * _pair(padding)[0]) % (_pair(stride)[0])
        owidth = (input.size()[3] - _pair(kernel_size)[1] +
                  2 * _pair(padding)[1]) % (_pair(stride)[1])
        if oheight != 0 or owidth != 0:
            caffe_out = raw(input,
                            kernel_size,
                            stride,
                            padding,
                            ceil_mode=True)
            print(
                "WARNING: the output shape miss match at {}: "
                "input {} output---Pytorch:{}---Caffe:{}\n"
                "This is caused by the different implementation that ceil mode in caffe and the floor mode in pytorch.\n"
                "You can add the clip layer in caffe prototxt manually if shape mismatch error is caused in caffe. "
                .format(layer_name, input.size(), x.size(), caffe_out.size()))

        top_name = top_blobs[0]
        tmp1_name = top_name + '_tmp1'
        drop1_name = top_name + '_drop1'
        tmp2_name = top_name + '_tmp2'
        drop2_name = top_name + '_drop2'

        if oheight != 0 and owidth != 0:
            trans_log.cnet.net.layer[-1].top[0] = tmp1_name
            slice1_layer = Layer(name=layer_name + '_slice1',
                                type='Slice',
                                bottom=[tmp1_name],
                                top=[tmp2_name, drop1_name])
            slice1_layer.slice_param(-1, [x.size()[-1]])
            trans_log.cnet.add_layer(slice1_layer)

            slice2_layer = Layer(name=layer_name + '_slice2',
                                type='Slice',
                                bottom=[tmp2_name],
                                top=top_blobs + [drop2_name])
            slice2_layer.slice_param(-2, [x.size()[-2]])
            trans_log.cnet.add_layer(slice2_layer)

            trans_log.cnet.blobs.remove(drop1_name)
            trans_log.cnet.blobs.remove(drop2_name)

        elif oheight != 0:
            trans_log.cnet.net.layer[-1].top[0] = tmp1_name
            slice1_layer = Layer(name=layer_name + '_slice1',
                                type='Slice',
                                bottom=[tmp1_name],
                                top=top_blobs + [drop1_name])
            slice1_layer.slice_param(-2, [x.size()[-2]])
            trans_log.cnet.add_layer(slice1_layer)

            trans_log.cnet.blobs.remove(drop1_name)

        elif owidth != 0:
            trans_log.cnet.net.layer[-1].top[0] = tmp1_name
            slice1_layer = Layer(name=layer_name + '_slice1',
                                type='Slice',
                                bottom=[tmp1_name],
                                top=top_blobs + [drop1_name])
            slice1_layer.slice_param(-1, [x.size()[-1]])
            trans_log.cnet.add_layer(slice1_layer)

            trans_log.cnet.blobs.remove(drop1_name)


def _max_pool2d(raw,
                input,
                kernel_size,
                stride=None,
                padding=0,
                dilation=1,
                ceil_mode=False,
                return_indices=False):
    x = raw(input, kernel_size, stride, padding, dilation, ceil_mode,
            return_indices)
    _pool('max', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _avg_pool2d(raw,
                input,
                kernel_size,
                stride=None,
                padding=0,
                ceil_mode=False,
                count_include_pad=True,
                divisor_override=None):
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    _pool('ave', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _adaptive_avg_pool2d(raw, input, output_size):
    x = raw(input, output_size)
    if isinstance(output_size, int):
        out_dim = output_size
    else:
        out_dim = output_size[0]
    tmp = max(input.shape[2], input.shape[3])
    stride = tmp // out_dim
    kernel_size = tmp - (out_dim - 1) * stride
    _pool('ave', raw, input, x, kernel_size, stride, 0, False)
    return x


def _dropout(raw, input, p=0.5, training=False, inplace=False):
    x = raw(input, p, training, inplace)
    bottom_blobs = [trans_log.blobs(input)]
    layer_name = trans_log.add_layer(name='dropout')
    top_blobs = trans_log.add_blobs([x], name=bottom_blobs[0], with_num=False)
    layer = Layer(name=layer_name,
                  type='Dropout',
                  bottom=bottom_blobs,
                  top=top_blobs)
    layer.param.dropout_param.dropout_ratio = p
    layer.param.include.extend([pb.NetStateRule(phase=0)
                                ])  # 1 for test, 0 for train

    trans_log.cnet.add_layer(layer)
    return x


def _threshold(raw, input, threshold, value, inplace=False):
    # for threshold or relu
    if threshold == 0 and value == 0:
        x = raw(input, threshold, value, inplace)
        bottom_blobs = [trans_log.blobs(input)]
        name = trans_log.add_layer(name='relu')
        trans_log.add_blobs([x], name='relu_blob')
        layer = Layer(name=name,
                      type='ReLU',
                      bottom=bottom_blobs,
                      top=[trans_log.blobs(x)])
        trans_log.cnet.add_layer(layer)
        return x
    if value != 0:
        raise NotImplemented("value !=0 not implemented in caffe")
    x = raw(input, input, threshold, value, inplace)
    bottom_blobs = [trans_log.blobs(input)]
    layer_name = trans_log.add_layer(name='threshold')
    top_blobs = trans_log.add_blobs([x], name='threshold_blob')
    layer = Layer(name=layer_name,
                  type='Threshold',
                  bottom=bottom_blobs,
                  top=top_blobs)
    layer.param.threshold_param.threshold = threshold

    trans_log.cnet.add_layer(layer)
    return x


def _relu(raw, input, inplace=False):
    # for threshold or prelu
    x = raw(input, False)
    name = trans_log.add_layer(name='relu')

    bottom_blob = trans_log.blobs(input)

    # relu top blob name same as bottom blob name
    # trans_log.add_blobs([x], name='relu_blob')
    top_blobs = trans_log.add_blobs([x], name=bottom_blob, with_num=False)

    layer = Layer(name=name, type='ReLU', bottom=[bottom_blob], top=top_blobs)

    trans_log.cnet.add_layer(layer)
    return x


def _prelu(raw, input, weight):
    # for threshold or prelu
    x = raw(input, weight)
    bottom_blobs = [trans_log.blobs(input)]
    name = trans_log.add_layer(name='prelu')
    trans_log.add_blobs([x], name='prelu_blob')
    layer = Layer(name=name,
                  type='PReLU',
                  bottom=bottom_blobs,
                  top=[trans_log.blobs(x)])
    if weight.size()[0] == 1:
        layer.param.prelu_param.channel_shared = True
        layer.add_data(weight.cpu().data.numpy()[0])
    else:
        layer.add_data(weight.cpu().data.numpy())

    trans_log.cnet.add_layer(layer)
    return x


def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    x = raw(input, negative_slope)
    name = trans_log.add_layer(name='leaky_relu')
    trans_log.add_blobs([x], name='leaky_relu_blob')
    layer = Layer(name=name,
                  type='ReLU',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])
    layer.param.relu_param.negative_slope = negative_slope

    trans_log.cnet.add_layer(layer)
    return x


def _tanh(raw, input):
    # for tanh activation
    x = raw(input)
    name = trans_log.add_layer(name='tanh')
    trans_log.add_blobs([x], name='tanh_blob')
    layer = Layer(name=name,
                  type='TanH',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])

    trans_log.cnet.add_layer(layer)
    return x


def _softmax(raw, input, dim=None, _stacklevel=3):
    # for F.softmax
    x = raw(input, dim=dim)
    if dim is None:
        dim = F._get_softmax_dim('softmax', input.dim(), _stacklevel)
    bottom_blobs = [trans_log.blobs(input)]
    name = trans_log.add_layer(name='softmax')
    trans_log.add_blobs([x], name='softmax_blob')
    layer = Layer(name=name,
                  type='Softmax',
                  bottom=bottom_blobs,
                  top=[trans_log.blobs(x)])
    layer.param.softmax_param.axis = dim

    trans_log.cnet.add_layer(layer)
    return x


def _batch_norm(raw,
                input,
                running_mean,
                running_var,
                weight=None,
                bias=None,
                training=False,
                momentum=0.1,
                eps=1e-5):
    # because the runing_mean and runing_var will be changed after the _batch_norm operation, we first save the parameters
    
    x = raw(input, running_mean, running_var, weight, bias, training, momentum,
            eps)

    if trans_log.merge_bn:
        last_layer_name = trans_log.layer_name_order[-1]
        last_layer_type = trans_log.layer_type_order[-1]

        if last_layer_type == 'conv':
            print('\tmerge bn, conv layer:', last_layer_name)

            layer_index = trans_log.cnet.layer_index(last_layer_name)
            conv_layer = trans_log.cnet.net.layer[layer_index]
            convolution_param = conv_layer.convolution_param
            num_output = convolution_param.num_output

            kernel_h = convolution_param.kernel_h
            kernel_w = convolution_param.kernel_w
            if kernel_h == 0 and kernel_w == 0:
                kernel_size = convolution_param.kernel_size
                kernel_h = kernel_w = kernel_size[0]
            
            # conv and bn params
            blobs_num = len(conv_layer.blobs)
            if blobs_num > 1:
                conv_weights = np.array(conv_layer.blobs[0].data).reshape(num_output, -1, kernel_h, kernel_w)
                conv_bias = np.array(conv_layer.blobs[1].data).reshape(num_output)
            else:
                conv_weights = np.array(conv_layer.blobs[0].data).reshape(num_output, -1, kernel_h, kernel_w)
                conv_bias = np.zeros([num_output,])
            
            running_mean_clone = running_mean.clone().cpu().numpy()
            running_var_clone = running_var.clone().cpu().numpy()
            bn_weights = weight.cpu().data.numpy()
            bn_bias = bias.cpu().data.numpy()

            # new conv params
            w_conv = conv_weights.reshape(num_output, -1)
            w_bn = np.diag(bn_weights / np.sqrt(eps + running_var_clone))
            new_conv_weights = np.dot(w_bn, w_conv)

            b_conv = conv_bias
            b_bn = bn_bias - bn_weights * running_mean_clone / (np.sqrt(running_var_clone + eps))
            new_conv_bias = np.dot(w_bn, b_conv) + b_bn

            # change conv params
            del conv_layer.blobs[:]
            for data in [new_conv_weights, new_conv_bias]:
                new_blob = conv_layer.blobs.add()
                for dim in data.shape:
                    new_blob.shape.dim.append(dim)
                new_blob.data.extend(data.flatten().astype(float))
            
            conv_layer.convolution_param.bias_term = False
            
            # change conv top blob
            for blob in conv_layer.top:
                trans_log.cnet.blobs.remove(blob)

            # top_blobs = trans_log.add_blobs([x], name='batch_norm_blob')
            top_blobs = trans_log.add_blobs([x], name=trans_log.blobs(input))
            del conv_layer.top[:]
            conv_layer.top.extend(top_blobs)

            return x

    
    bottom_blobs = [trans_log.blobs(input)]
    layer_name1 = trans_log.add_layer(name='batch_norm')
    # top_blobs = trans_log.add_blobs([x], name='batch_norm_blob')
    top_blobs = trans_log.add_blobs([x], name=bottom_blobs[0])
    layer1 = Layer(name=layer_name1,
                   type='BatchNorm',
                   bottom=bottom_blobs,
                   top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
        running_mean_clone = running_mean.clone()
        running_var_clone = running_var.clone()
        layer1.add_data(running_mean_clone.cpu().numpy(),
                        running_var_clone.cpu().numpy(), np.array([1.0]))

    trans_log.cnet.add_layer(layer1)

    if weight is not None and bias is not None:
        layer_name2 = trans_log.add_layer(name='bn_scale')
        layer2 = Layer(name=layer_name2,
                       type='Scale',
                       bottom=top_blobs,
                       top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())

        trans_log.cnet.add_layer(layer2)
    return x


def _instance_norm(raw,
                   input,
                   running_mean=None,
                   running_var=None,
                   weight=None,
                   bias=None,
                   use_input_stats=True,
                   momentum=0.1,
                   eps=1e-5):
    # TODO: the batch size!=1 view operations
    print(
        "WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1"
    )
    if running_var is not None or weight is not None:
        # TODO: the affine=True or track_running_stats=True case
        raise NotImplementedError(
            "not implement the affine=True or track_running_stats=True case InstanceNorm"
        )
    x = torch.batch_norm(input, weight, bias, running_mean, running_var,
                         use_input_stats, momentum, eps,
                         torch.backends.cudnn.enabled)
    bottom_blobs = [trans_log.blobs(input)]
    layer_name1 = trans_log.add_layer(name='instance_norm')
    top_blobs = trans_log.add_blobs([x], name='instance_norm_blob')
    layer1 = Layer(name=layer_name1,
                   type='BatchNorm',
                   bottom=bottom_blobs,
                   top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
        running_mean = torch.zeros(input.size()[1])
        running_var = torch.ones(input.size()[1])
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
    running_mean_clone = running_mean.clone()
    running_var_clone = running_var.clone()
    layer1.add_data(running_mean_clone.cpu().numpy(),
                    running_var_clone.cpu().numpy(), np.array([1.0]))

    trans_log.cnet.add_layer(layer1)

    if weight is not None and bias is not None:
        layer_name2 = trans_log.add_layer(name='bn_scale')
        layer2 = Layer(name=layer_name2,
                       type='Scale',
                       bottom=top_blobs,
                       top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())

        trans_log.cnet.add_layer(layer2)
    return x


#upsample layer
def _interpolate(raw,
                 input,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
    # 定义的参数包括 scale,即输出与输入的尺寸比例,如 2;scale_h、scale_w,
    # 同 scale,分别为 h、w 方向上的尺寸比例;pad_out_h、pad_out_w,仅在 scale 为 2 时
    # 有用,对输出进行额外 padding 在 h、w 方向上的数值;upsample_h、upsample_w,输
    # 出图像尺寸的数值。在 Upsample 的相关代码中,推荐仅仅使用 upsample_h、
    # upsample_w 准确定义 Upsample 层的输出尺寸,其他所有的参数都不推荐继续使用。

    x = raw(input, size, scale_factor, mode, align_corners)
    layer_name = trans_log.add_layer(name='upsample')
    top_blobs = trans_log.add_blobs([x], name='upsample_blob'.format(type))

    if mode == 'nearest':
        if align_corners != None:
            raise NotImplementedError(
                "not implement F.interpolate totoaly, align_corners is not None"
            )

        # layer = Layer(name=layer_name,
        #               type='Upsample',
        #               bottom=[trans_log.blobs(input)],
        #               top=top_blobs)

        # if scale_factor is None:
        #     scale_factor = max(
        #         x.size(2) / input.size(2),
        #         x.size(3) / input.size(3))

        # layer.upsample_param(size=(input.size(2), input.size(3)),
        #                      scale_factor=int(scale_factor))
        
        layer = Layer(name=layer_name,
                      type='Deconvolution',
                      bottom=[trans_log.blobs(input)],
                      top=top_blobs)

        scale = 1
        if scale_factor is not None:
            scale = scale_factor
        if size is not None and scale_factor is None:
            raise ValueError(
                "Converting Interpolate with random size is still not supported!"
            )
        
        kernel_size = int(scale)
        stride = int(scale)
        pad = 0
        channels = x.size(1)
        weight = np.ones([channels, 1, kernel_size, kernel_size], dtype='float32')
        
        layer.conv_param(channels,
                         kernel_size,
                         stride=stride,
                         pad=pad,
                         bias_term=False,
                         groups=channels)
        layer.add_data(weight)

    elif mode == 'bilinear':
        layer = Layer(name=layer_name,
                      type='Deconvolution',
                      bottom=[trans_log.blobs(input)],
                      top=top_blobs)

        def bilinear_weight(shape):
            shape = [int(shape_i) for shape_i in shape]
            weight = np.zeros(np.prod(shape), dtype='float32')
            f = np.ceil(shape[3] / 2.0)
            c = (2 * f - 1 - f % 2) / (2.0 * f)
            for i in range(np.prod(shape)):
                x = i % shape[3]
                y = i % (shape[3] * shape[2]) // shape[3]
                weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            return weight.reshape(shape)

        scale = 1
        if scale_factor is not None:
            scale = scale_factor
        if size is not None and scale_factor is None:
            raise ValueError(
                "Converting Interpolate with random size is still not supported!"
            )

        kernel_size = int(2 * scale - scale % 2)
        stride = int(scale)
        pad = int(np.ceil((scale - 1) / 2))
        channels = x.size(1)
        weight = bilinear_weight([channels, 1, kernel_size, kernel_size])
        layer.conv_param(channels,
                         kernel_size,
                         stride=stride,
                         pad=pad,
                         bias_term=False,
                         groups=channels)
        layer.add_data(weight)
    else:
        raise NotImplementedError("The interpolate upsample only support nearest/bilinear in caffe")

    trans_log.cnet.add_layer(layer)
    return x


#sigmid layer
def _sigmoid(raw, input):
    # Applies the element-wise function:
    #
    # Sigmoid(x)= 1/(1+exp(−x)）
    #
    # ​
    x = raw(input)
    name = trans_log.add_layer(name='sigmoid')
    trans_log.add_blobs([x], name='sigmoid_blob')
    layer = Layer(name=name,
                  type='Sigmoid',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])

    trans_log.cnet.add_layer(layer)
    return x


#tanh layer
def _tanh(raw, input):
    # Applies the element-wise function:
    #
    # torch.nn.Tanh
    #
    # ​
    x = raw(input)
    name = trans_log.add_layer(name='tanh')
    trans_log.add_blobs([x], name='tanh_blob')
    layer = Layer(name=name,
                  type='TanH',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])

    trans_log.cnet.add_layer(layer)
    return x


def _hardtanh(raw, input, min_val, max_val, inplace):
    # Applies the element-wise function:
    #
    # torch.nn.ReLu6
    #
    # ​
    print('relu6: ', trans_log.blobs(input))
    x = raw(input, min_val, max_val)
    name = trans_log.add_layer(name='relu6')
    trans_log.add_blobs([x], name='relu6_blob')
    layer = Layer(name=name,
                  type='ReLU6',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])

    trans_log.cnet.add_layer(layer)
    return x


#L2Norm layer
def _l2Norm(raw, input, weight, eps):
    # Applies the element-wise function:
    #
    # L2Norm in vgg_ssd
    #
    # ​
    x = raw(input, weight, eps)
    name = trans_log.add_layer(name='normalize')
    trans_log.add_blobs([x], name='normalize_blob')
    layer = Layer(name=name,
                  type='Normalize',
                  bottom=[trans_log.blobs(input)],
                  top=[trans_log.blobs(x)])
    layer.norm_param(eps)

    layer.add_data(weight.cpu().data.numpy())

    trans_log.cnet.add_layer(layer)
    return x


def replace():

    F.conv2d = Rp(F.conv2d, _conv2d)
    F.linear = Rp(F.linear, _linear)
    F.relu = Rp(F.relu, _relu)
    F.leaky_relu = Rp(F.leaky_relu, _leaky_relu)
    F.max_pool2d = Rp(F.max_pool2d, _max_pool2d)
    F.avg_pool2d = Rp(F.avg_pool2d, _avg_pool2d)
    F.adaptive_avg_pool2d = Rp(F.adaptive_avg_pool2d, _adaptive_avg_pool2d)
    F.dropout = Rp(F.dropout, _dropout)
    F.threshold = Rp(F.threshold, _threshold)
    F.prelu = Rp(F.prelu, _prelu)
    F.batch_norm = Rp(F.batch_norm, _batch_norm)
    F.instance_norm = Rp(F.instance_norm, _instance_norm)
    F.softmax = Rp(F.softmax, _softmax)
    F.conv_transpose2d = Rp(F.conv_transpose2d, _conv_transpose2d)
    F.interpolate = Rp(F.interpolate, _interpolate)
    F.sigmoid = Rp(F.sigmoid, _sigmoid)
    F.tanh = Rp(F.tanh, _tanh)
    F.tanh = Rp(F.tanh, _tanh)
    F.hardtanh = Rp(F.hardtanh, _hardtanh)
    # F.l2norm = Rp(F.l2norm,_l2Norm)