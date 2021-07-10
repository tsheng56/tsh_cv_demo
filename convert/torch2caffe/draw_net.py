#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

from convert.torch2caffe.caffe import draw
from convert.torch2caffe.caffe import caffe_pb2

def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('output_image_file',
                        help='Output image file')
    parser.add_argument('--rankdir',
                        help=('One of TB (top-bottom, i.e., vertical), '
                              'RL (right-left, i.e., horizontal), or another '
                              'valid dot option; see '
                              'http://www.graphviz.org/doc/info/'
                              'attrs.html#k:rankdir'),
                        default='LR')
    parser.add_argument('--phase',
                        help=('Which network phase to draw: can be TRAIN, '
                              'TEST, or ALL.  If ALL, then all layers are drawn '
                              'regardless of phase.'),
                        default="ALL")
    parser.add_argument('--display_lrm', action='store_true',
                        help=('Use this flag to visualize the learning rate '
                              'multiplier, when non-zero, for the learning '
                              'layers (Convolution, Deconvolution, InnerProduct).'))

    args = parser.parse_args()
    return args


def draw_caffe_net(input_net_proto_file, output_image_file, rankdir='LR', phase=None, display_lrm=None):

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(input_net_proto_file).read(), net)
    print('>> Drawing net: %s' % os.path.abspath(output_image_file))

    draw.draw_net_to_file(net, output_image_file, rankdir, phase, display_lrm)


if __name__ == '__main__':
    draw_caffe_net()
