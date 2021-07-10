import os
import cv2
import glob
import random
import argparse
import numpy as np

import torch

from detect.utils.torch_utils import intersect_dicts
# from detect.models.yaml_yolo import Model as Model_yaml
# from detect.models.torch_yolo import Model as Model_torch
# from detect.models.darknet_yolo import Model as Model_darknet
from detect.models.yaml_yolo import Model

from configs.detect.data import *

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_mode", type=str, help='1:video, 2.image')
    parser.add_argument("--cfg", '-c', type=str, default='')
    
    parser.add_argument("--videos", type=str, default=None)
    parser.add_argument("--list", type=str, default=None)
    parser.add_argument("--speed", type=int, default=1)
    
    parser.add_argument("--images", type=str, default=None)
    
    parser.add_argument("--shuffle", type=int, default=0)
    
    parser.add_argument("--weights", '-w', type=str, default='')
    
    parser.add_argument("--gaussianblur", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf_thresh", type=int, default=-1)
    parser.add_argument("--cls_conf_thresh", type=list, default=[])
    parser.add_argument("--nms_thresh", type=list, default=[])
    parser.add_argument("--nms_all", type=int, default=0)
    
    parser.add_argument("--show", type=int, default=1)
    parser.add_argument("--save", type=int, default=0)
    opt = parser.parse_args()
    
    assert os.path.exists(opt.cfg) or os.paht.exists('configs/' + opt.cfg), f'{opt.cfg} not exists'
    opt.cfg, _ = os.path.splitext(os.path.basename(opt.cfg))
    opt.cfg_train = eval(opt.cfg + '_train')
    
    # data
    opt.cfg_data = eval(opt.cfg_train['data'])
    opt.cls_conf_thresh = opt.cfg_data['conf_thres']
    opt.nms_thresh = opt.cfg_data['iou_thres']
    
    assert (opt.images is not None) or (opt.videos is not None), "input images or videos"
    assert (opt.show > 0 or opt.save > 0), "choose show and save"
    assert os.path.isfile(opt.weights), f"weights:{opt.weights} is not file"
    
    if opt.images is None:
        opt.demo_mode = '1'
        
        if os.path.isfile(opt.videos):
            if opt.videos.endswith('.txt') or opt.videos.endswith('.list'):
                opt.videos = [video.strip().split(';')[0] for video in open(opt.vides, 'r').readlines()]
            else:
                opt.videos = [opt.videos]

        elif os.path.isdir(opt.videos):
            opt.videos = glob.glov(os.path.join(opt.videos, '*.*'))

        else:
            opt.videos = glob.glob(os.path.koin(opt.videos, '*.*'))

            if len(opt.videos) == 0:
                 opt.videos = [f'{opt.videos} is not file or dir']

        if opt.shuffle:
            random.shuffle(opt.list)

    else:
        opt.demo_mode = '2'
        
        if os.path.isfile(opt.images):
            if opt.images.endswith('.txt') or opt.images.endswith('.list'):
                opt.images = [image.strip().split(';')[0] for image in open(opt.images, 'r').readlines()]
    	    else:
                opt.images = [opt.images]

        elif os.path.isdir(opt.images):
            opt.images = glob.glob(os.path.join(opt.images, '*.*'))

        else:
            opt.images = glob.glob(os.path.join(opt.images, '*.*'))

            if len(opt.images) == 0:
                opt.images = [f'{opt.images} is not file or dir']

        if opt.shuffle:
            random.shuffle(opt.images)

    return opt

def nothing(emp):
    pass
                                                                                    
def video_demo(model, imgsz, opt, classes, class_thresh, device):
                                    
    for video_index in opt.list:
        video = opt.videos[video_index]
        print("\tread index:{} video{}".format(video_index, video))
        if video.split('.')[-1].lower() not in vid_formats:
            continue
        
        try:
            capture = cv2.VideoCapture(video)
            cap_width = int(capture.get(3))
            cap_height = int(capture.get(4))
            framerate = int(capture.get(5))
            framenum = int(capture.get(7))
            
            timeF = opt.speed
            flag = 0
            namedWindow = os.path.basename(video) + ' ' + opt.weights.replace('/', '-')
            cv2.namedWindow(namedWindow, 0)
            cv2.resizeWindow(namedWindow, 700, 500)
            cv2.moveWindow(namedWindow, 600, 300)
            
            frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            loop_flag = 0
            pos = 0
            barname = 'time/{}'.format(timeF)
            cv2.createTrackbar(barname, namedWindow, 0, int(frame/timeF), nothing)
            cv2.setTrackbarPos(barname, namedWindow, int(pos))
        except:
            cv2.destroyAllWindows()
            continue

        if opt.save:
            video_name = os.path.join('output', os.path.basename(video))
            if video_name[-3:] == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif video_name[-3:] == 'mp4':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            else:
                print(video_name, 'is not avi or mp4')
                raise NameError
            print('\t save video: {}'.format(video_name))
            videoWrite = cv2.VideoWriter(video_name, fourcc, 25, (cap_width, cap_height))                            
        
        while True:
            ret, image = capture.read()
            if ret == False:
                break
            if loop_flag % timeF != 0:
                loop_flag += 1
                continue

            img_deploy = image.copy()
            if opt.gaussianblur:
                image = cv2.GaussianBlur(image, (3,3), 0)

            input_imgs, scale, _, pad = letterbox(image, auto=False, new_shape=imgsz)
            pad_h_top, bottom, pad_w_left, right = pad
            input_imgs = input_imgs.transpose(2,0,1)
            input_imgs = np.ascontiguousarray(input_imgs)
            input_imgs = torch.from_numpy(input_imgs).to(device).float()
            input_imgs /= 255.0
            if input_imgs.ndimension() == 3:
                input_imgs = input_imgs.unsqueeze(0)

            with torch.no_grad():
                detections = model(input_imgs)[0]

            detections = tsh_non_max_suppression(scale=scale,prediction=detections,conf_thres=opt.conf_thresh, nms_thres=opt.nms_thresh,all_nms=opt.nms_all,pad_left=pad_w_left,pad_top=pad_h_top)
            
            if detections[0] is not None:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                    cat_id = int(cls_pred)
                    if cls_conf < class_thresh[cat_id]:
                        continue
                    plot_one_box(img_deploy, (x1,y1,x2,y2), int(cls_pred),score=cls_conf, label=classes[cat_id])

            cv2.setTrackbarPos(barname, namedWindow, int(loop_flag/timeF))
            
            if pos * timeF != loop_flag and loop_flag != timeF:
                loop_flg = pos*timeF
                capture.set(cv2.CAP_PROP_POS_FRAMESM, loop_flag)
                cv2.setTrackbarPos(barname, namedWindow, pos)

            loop_flag = loop_flag + 1

            if ret == True:
                if opt.show:
                    cv2.imshow(namedWindow, img_deploy)
                if flag == 0:
                    key = cv2.waitKey(0)
                else:
                    key = cv2.waitKey(1)
                if key == 98:
                    flag = 0
                if key == 99:
                    flag = 1
                if key == ord('q') or key == ord('n'):
                    break
            
        if opt.save:
            videoWrite.release()

        capture.release()
        cv2.destroyAllWindows()
        if key == ord('q'):
            break

def image_demo(model, imgsz, opt, classes, class_thresh, device):
                                    
    for image_index, image in enumerate(opt.images):
        print("\tread index:{} video{}".format(image_index, image))

        if image.split('.')[-1].lower() not in img_formats:
            continue

        image_org = cv2.imread(image)
        namedWindow = opt.weights.replace('/', '-')
        cv2.namedWindow(namedWindow, 0)
        cv2.resizeWindow(namedWindow, 700, 500)
        cv2.moveWindow(namedWindow, 600, 300)
            

        if opt.save:
            image_name = os.path.join('output', os.path.basename(image))
            print('\t save image: {}'.format(image_name))                     


        img_deploy = image_org.copy()

        if opt.gaussianblur:
            image = cv2.GaussianBlur(image_org, (3,3), 0)

        input_imgs, scale, _, pad = letterbox(image, auto=False, new_shape=imgsz)
        pad_h_top, bottom, pad_w_left, right = pad
        input_imgs = input_imgs.transpose(2,0,1)
        input_imgs = np.ascontiguousarray(input_imgs)
        input_imgs = torch.from_numpy(input_imgs).to(device).float()
        input_imgs /= 255.0
        if input_imgs.ndimension() == 3:
            input_imgs = input_imgs.unsqueeze(0)

        with torch.no_grad():
            detections = model(input_imgs)[0]

        detections = tsh_non_max_suppression(scale=scale,prediction=detections,conf_thres=opt.conf_thresh, nms_thres=opt.nms_thresh,all_nms=opt.nms_all,pad_left=pad_w_left,pad_top=pad_h_top)
        
        if detections[0] is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                cat_id = int(cls_pred)
                if cls_conf < class_thresh[cat_id]:
                    continue
                plot_ont_box(img_deploy, (x1,y1,x2,y2), int(cls_pred),score=cls_conf, label=classes[cat_id])

        if opt.save:
            cv2.imwrite(image_name, img_deploy)
       
        if opt.show:
            cv2.imshow(namedWindow, img_deploy)
        if flag == 0:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)
        if key == 98:
            flag = 0
        if key == 99:
            flag = 1
        if key == ord('q') or key == ord('n'):
            break
        
        
def run(opt):
    if opt.save:
        os.makedirs("output",exist_ok=True)
    
    cfg_name, _ = os.path.splitext(os.path.basename(opt.cfg))
    exec(f"from configs.detect.{cfg_name} import cfg_train, cfg_hyp, cfg_hyp_custom")
    print(f"\nfrom configs/detect/{cfg_name} import cfg_train, cfg_hyp, cfg_hyp_custom\n")
    
    opt.cfg_train = eval('cfg_train')
    hyp = {**eval('cfg_hyp'), **eval('cfg_hyp_custom')}
    version = opt.cfg_train['loss_type']
    model_name = opt.cfg_train['model']
    anchors = opt.cfg_train['anchors']
    nc = opt.cfg_data['nc']
    classes = opt.cfg_data['names']
    imgsz = [opt.cfg_train['height'], opt.cfg_train['width']]
    print(">>>>> \n imgsz", imgsz)
    
    assert len(opt.cls_conf_thresh) == nc, "confg_thresh is wrong"
    class_thresh = np.array(opt.cls_conf_thresh)[:nc].tolist()
    if opt.conf_thresh < 0:
        opt.conf_thresh = min(class_thresh)
    
    # device
    device = torch.device("cuda")
    torch.cuda.set_device(opt.device)

    # Set model file
    if model_name.endswith('.yaml'):
        from detect.models.yaml_yolo import Model
        model_file = 'yolodet/models/model_yaml/' + model_name
        assert os.path.exists(model_file), f"{model_file} not exists"
    elif model_name.endswith('.cfg'):
        from detect.models.darknet_yolo import Model
        model_file = 'yolodet/models/model_darknet/' + model_name
        assert os.path.exists(model_file), f"{model_file} not exists"
    else:
        from detect.models.torch_yolo import Model
        model_file = model_name
        assert os.path.exists('yolodet/models/model_torch/torch_' + model_file + '.py'), f"{model_file} not exists"
   
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    model = Model(version, model_file or ckpt['model'].yaml, ch=3, nc=nc, anchors=anchors).to(device)  # create
    exclude = ['anchor'] if (model_name or hyp.get('anchors')) and not opt.cfg_train['resume'] else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), opt.weights))  # report
    
    if 'epoch' in ckpt:
        print(f"opech is {ckpt['epoch']}")
        
    model.fuse()
    model.eval()
    
    Demo = {
        '1': video_demo,
        '2': image_demo,    
    }[opt.demo_mode](model, imgsz, opt, classes, class_thresh, device)
    
if __name__ == '__main__':
    opt = get_args()
    run(opt)
                