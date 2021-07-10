
cfg_train = {
    'data': 'coco_test',
    'loss_type': 'v5',
    'model': 'yolov5_simple.yaml', #'yolov5_simple.yaml',
    'weights': '',
    'anchors': [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]],
    
    'debug': 0,
    'width': 576,
    'height': 384,
    'multi_scale': False,
    'rect': False,
    'quad': False,

    'adam': False,
    'nbs': 64,  # nominal batch size
    'batch_size': 8,
    'total_batch_size': 0,
    'epochs':  2,
    'workers': 1,
    'sync_bn': True,
    'device': '',

    'resume': False,
    'test': True,
    'evolve': False,
    'save': True,
    'autoanchor': False,
    'image_weights': False,
    'cache_images': False,
    'autobalance': False,
    'bucket':'',
    
    'project': 'runs_det/train',
    'name': 'exp',
    'exist_ok': False,

    'save_period': -1,
    'bbox_interval': -1,
    'upload_dataset': False,
    'entity': None,

    # cfg_v4
    'stride': [8, 16, 32],
    'use_all_anchors': False,
    'object_scale': 1.0,
    'noobject_scale': 1.0,
    'class_scale': 1.0,
    'coord_scale': 1.0,
    'cls_normalizer': 1.0,
    'iou_normalizer': 1.0,
    'ignore_thresh': 0.5,
    'iou_thresh': 1.0,
    'box_loss_type': 'ciou', # mse | smooth | giou | diou | ciou
    'focalloss': False,
}

cfg_hyp_custom = {
    'averaged_between_devices': False,
    'loss_mean': False,
    'label_smoothing': 0.0,
    'linear_lr': False,
    'lr_cos': True,
    'lr_step': [100, 200],
}

# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials
cfg_hyp = {
    'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    'lrf': 0.2,  # final OneCycleLR learning rate (lr0 * lrf)
    'momentum': 0.937,  # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # optimizer weight decay 5e-4
    'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
    'warmup_momentum': 0.8,  # warmup initial momentum
    'warmup_bias_lr': 0.1,  # warmup initial bias lr
    'box': 0.05,  # box loss gain
    'cls': 0.5,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 1.0,  # obj loss gain (scale with pixels)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.20,  # IoU training threshold
    'anchor_t': 4.0,  # anchor-multiple threshold
    # 'anchors': 3,  # anchors per output layer (0 to ignore)
    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
    'degrees': 0.0,  # image rotation (+/- deg)
    'translate': 0.1,  # image translation (+/- fraction)
    'scale': 0.5,  # image scale (+/- gain)
    'shear': 0.0,  # image shear (+/- deg)
    'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,  # image flip up-down (probability)
    'fliplr': 0.5,  # image flip left-right (probability)
    'mosaic': 0.0,  # image mosaic (probability)
    'mixup': 0.0,  # image mixup (probability)
    'cutcolor': 0.1, # image cutcolor (probability)
}