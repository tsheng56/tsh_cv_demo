import os
import torch
import argparse
import numpy as np

from detect.utils.torch_utils import intersect_dicts
# from detect.models.yaml_yolo import Model as Model_yaml
# from detect.models.torch_yolo import Model as Model_torch
# from detect.models.darknet_yolo import Model as Model_darknet
from detect.models.yaml_yolo import Model

from configs.detect.data import *

from convert.torch2caffe.pytorch_to_caffe import trans_net

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", '-c', type=str, default='configs/detect/coco_test_cfg.py')
    parser.add_argument("--weights", '-w', type=str, default='detect.pth')
    parser.add_argument("--merge_bn", type=int, default=1)
    parser.add_argument("--simple_name", type=int, default=0)
    parser.add_argument("--pre", type=str, default='detect')
    parser.add_argument("--draw", type=int, default=0)
    opt = parser.parse_args()
    
    assert os.path.exists(opt.cfg) or os.paht.exists('configs/detect/' + opt.cfg), f'{opt.cfg} not exists'

    return opt

if __name__ == '__main__':
    opt = get_args()

    cfg_name, _ = os.path.splitext(os.path.basename(opt.cfg))
    exec(f"from configs.detect.{cfg_name} import cfg_train, cfg_hyp, cfg_hyp_custom")
    print(f"\nfrom configs/detect/{cfg_name} import cfg_train, cfg_hyp, cfg_hyp_custom\n")

    device = 'cpu'
    opt.cfg_train = eval('cfg_train')
    hyp = {**eval('cfg_hyp'), **eval('cfg_hyp_custom')}
    opt.cfg_data = eval(opt.cfg_train['data'])

    version = opt.cfg_train['loss_type']
    model_name = opt.cfg_train['model']
    anchors = opt.cfg_train['anchors']
    nc = opt.cfg_data['nc']
    imgsz = [opt.cfg_train['height'], opt.cfg_train['width']]
    input_size = (1, 3, imgsz[0], imgsz[1])
    print("\t>> input_size", input_size)

    # Set model file
    if model_name.endswith('.yaml'):
        from detect.models.yaml_yolo import Model
        model_file = 'detect/models/model_yaml/' + model_name
        assert os.path.exists(model_file), f"{model_file} not exists"
    elif model_name.endswith('.cfg'):
        from detect.models.darknet_yolo import Model
        model_file = 'detect/models/model_darknet/' + model_name
        assert os.path.exists(model_file), f"{model_file} not exists"
    else:
        from detect.models.torch_yolo import Model
        model_file = model_name
        assert os.path.exists('detect/models/model_torch/torch_' + model_file + '.py'), f"{model_file} not exists"

    # Create Model
    pretrained = opt.weights != 'detect.pth' and (opt.weights.endswith('.pt') or opt.weights.endswith('.pth'))
    if pretrained:
        # with torch_distributed_zero_first(rank):
        #     attempt_download(weights)  # download if not found locally

        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        model = Model(version, model_file or ckpt['model'].yaml, ch=3, nc=nc, anchors=anchors).to(device)  # create
        exclude = ['anchor'] if (model_name or hyp.get('anchors')) and not opt.cfg_train['resume'] else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), opt.weights))  # report
    else:
        model = Model(version, model_file, ch=3, nc=nc, anchors=anchors).to(device)  # create

    # Set export
    try:
        model.model[-1].export = True
    except:
        model.model.detect.export = True

    # Merge Bn
    # if opt.merge_bn:
    #     model.fuse()

    model.eval()

    # Pytorch to Caffe
    trans_net(model, input_size, ckpt=opt.weights, merge_bn=opt.merge_bn, simple_name=opt.simple_name, output_blob_pre=opt.pre, draw=opt.draw)

