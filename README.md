## 2021/06/01
- create yolo v3 v4 v5 loss

## 2021/06/06
- create yolodet/convert, use to trans torch model to caffe model
    - pytorch2caffe.py
    - [--draw=1] need Graphviz
- nn.SiLU() -> nn.ReLU(), caffe not support SiLU
- change yolo v4 loss
- change yolodet/api/train.py & test.py
- change detect.py

## 2021/06/30
- rename yolodet to detect
- mv convert out from detect
- add convert/torch_class_replace.py
- create classify, adapted from: https://github.com/rwightman/pytorch-image-models.git

## 2021/07/10
- change convert/torch2caffe, merge bn, upsample