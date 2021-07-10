
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='/home/tsh/workspace/github/YOLOdetection/list_files/val2017_list.txt')
args = parser.parse_args()

lines = open(args.file, 'r').readlines()

cv2.namedWindow('img', 0)
for line in tqdm(lines):
    img_file, label_file = line.strip().split(';')

    img = cv2.imread(img_file)
    img_h, img_w = img.shape[:2]

    datas = np.loadtxt(label_file).reshape((-1, 5))

    for data in datas:

        cat, x, y, w, h = data
        x1 = int((x - w / 2.0) * img_w)
        y1 = int((y - h / 2.0) * img_h)
        x2 = int((x + w / 2.0) * img_w)
        y2 = int((y + h / 2.0) * img_h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
    
    cv2.imshow('img', img)

    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):
        break
        
