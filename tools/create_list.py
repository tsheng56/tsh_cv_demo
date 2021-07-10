import os
import argparse
from tqdm import tqdm

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_src', type=str, default='/home/tsh/workspace/coco/val2017')
    parser.add_argument('--img_seg', type=str, default='val2017/')
    parser.add_argument('--txt_seg', type=str, default='val_txt/')
    parser.add_argument('--list_file', type=str, default='val2017_list.txt')
    parser.add_argument('--save', type=str, default='list_files')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    img_src = args.img_src
    img_seg = args.img_seg
    txt_seg = args.txt_seg
    
    args.save = os.path.join(os.path.abspath('.'), args.save)
    os.makedirs(args.save, exist_ok=True)
    print('dirname:',os.path.abspath('.'), 'save:',args.save)

    list_file = os.path.join(args.save, args.list_file)

    cmd = f"find {img_src} -name '*.*' > {list_file}"
    print('cmd:',cmd)
    os.system(cmd)
    with open(list_file, 'r') as f:
        img_files = f.readlines()

    img_files = [x.strip() for x in img_files if x.strip().split('.')[-1].lower() in img_formats]

    list_file = open(list_file, 'w')

    write_num = 0
    for img_file in tqdm(img_files):
        txt_file = img_file.replace(img_seg, txt_seg)+'.txt'
        if os.path.exists(txt_file):
            list_file.write(f"{img_file};{txt_file}\n")
            write_num += 1

    list_file.close()

    print('write list:', write_num)

