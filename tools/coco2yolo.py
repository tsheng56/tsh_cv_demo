

import os
import sys
import cv2
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

sys.path.append('../YOLOdetection/')
from configs.data import coco, custom

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='coco')
    parser.add_argument('--file', type=str, default='/home/tsh/workspace/coco/annotations/instances_val2017.json')
    parser.add_argument('--root', type=str, default='/home/tsh/workspace/coco/val2017/')
    parser.add_argument('--save', type=str, default='/home/tsh/workspace/coco/val_txt/')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    return args

def run(args):

    assert os.path.exists(args.file), f"file: {args.file}, is not exists" 

    try:    
        date_cfg = eval(args.data)
    except:
        print(f"data_cfg: {args.data}, is no define in configs.data")

    class_name = date_cfg['names']
    print("class_name")
    print(class_name)

    with open(args.file, 'r') as f:
        data = json.load(f)
    
    categories = data['categories']
    categories_map = {info['id']: info['name'] for info in categories}
    images = set([info['id'] for info in data['images']])
    annotations = data['annotations']
    images_anns = set()

    sort_data = defaultdict(list)
 
    print("sort json data >>>>")
    for ann in tqdm(annotations):
        id = ann['image_id']
        cat = ann['category_id']
        bbox = ann['bbox']

        categorie = categories_map[cat]
        cat_index = class_name.index(categorie)
        bbox.append(cat_index)

        images_anns.add(id)
        name = '{:012d}.jpg'.format(id)
        sort_data[name].append(bbox)
    
    print(len(images), len(images_anns))
    

    os.makedirs(args.root, exist_ok=True)
    print("create anns txtfile >>>>")
    for img_file in tqdm(sort_data.keys()):
        anns = sort_data[img_file]

        image = cv2.imread(os.path.join(args.root, img_file))
        h, w = image.shape[:2]

        txt_file = os.path.join(args.save, img_file + '.txt')
        with open(txt_file, 'w') as f:
            for ann in anns:
                sx, sy ,sw, sh = (ann[0]+ann[2] / 2.0) / w, (ann[1]+ann[3] / 2.0) / h, 1.0 * ann[2] / w, 1.0 * ann[3] / h
                sx = min(max(0.0, sx), 1.0)
                sy = min(max(0.0, sy), 1.0)
                sw = min(max(0.0, sw), 1.0)
                sh = min(max(0.0, sh), 1.0)

                f.write(f"{int(ann[-1])} {sx} {sy} {sw} {sh}\n")
    

    images_null = images - images_anns
    print("create last txtfile >>>>")
    for img_id in tqdm(images_null):
        name = '{:012d}.jpg'.format(img_id)
        txt_file = os.path.join(args.save, name + '.txt')
        open(txt_file, 'w')


if __name__ == '__main__':

    args = parse_args()
    run(args)


