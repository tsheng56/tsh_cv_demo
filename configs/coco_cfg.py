
cfg_train = {
    'data': 'coco',
    'loss_type': 'v5',
    'model': 'yolov5s.yaml',
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
    'batch_size': 4,
    'total_batch_size': 0,
    'epochs':  5,
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
    
    'project': 'runs/train',
    'name': 'exp',
    'exist_ok': False,

    'save_period': -1,
    'bbox_interval': -1,
    'upload_dataset': False,
    'entity': None,

    # cfg_v4
    'strdie': [6, 16, 32],
    'use_all_anchors': False,
    'object_scale': 1.0,
    'noobject_scale': 1.0,
    'class_scale': 1.0,
    'coord_scale': 1.0,
    'cls_normalizer': 1.0,
    'iou_normalizer': 1.0,
    'ignore_thresh': 0.5,
    'iou_thresh': 1.0,
    'label_smooth_eps': 0,
    'box_loss_func': 'ciou', # mse | smooth | giou | diou | ciou
    'focalloss': False,
}
