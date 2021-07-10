
import os
import time
import yaml
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from configs import *
from configs.data import *
from configs.hyp import cfg_hyp, cfg_hyp_custom

from detect.api.train import train
from detect.utils.general import set_logging, check_requirements, increment_path, get_latest_run, colorstr, print_mutation
# from detect.utils.wandb_logging.wandb_utils import check_wandb_resume
from detect.utils.torch_utils import select_device
from detect.utils.plots import plot_evolution
from detect.utils.metrics import fitness

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '-c', type=str, default='coco_test_cfg.py', help='initial weights path')
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
opt = parser.parse_args()

# Set cfg_train
assert os.path.exists(opt.cfg) or os.path.exists('configs/' + opt.cfg), f'{opt.cfg} not exists'
cfg_name, _ = os.path.splitext(os.path.basename(opt.cfg))
opt.cfg_train = eval(cfg_name + '_train')

# Set data and hyp
opt.cfg_data = eval(opt.cfg_train['data'])
opt.cfg_hyp = cfg_hyp
opt.cfg_hyp_custom = cfg_hyp_custom
opt.img_size = [opt.cfg_train['height'], opt.cfg_train['width']]

# Set DDP variables
opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

set_logging(opt.global_rank)
if opt.global_rank in [-1, 0]:
    check_requirements()

# Resume
# wandb_run = check_wandb_resume(opt)
wandb_run = None
if opt.cfg_train['resume'] and not wandb_run:  # resume an interrupted run
    ckpt = opt.cfg_train['resume'] if isinstance(opt.cfg_train['resume'], str) else get_latest_run()  # specified or most recent path
    assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
    apriori = opt.global_rank, opt.local_rank
    with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace
    opt.cfg_train['model'], opt.cfg_train['weights'], opt.cfg_train['resume'], opt.cfg_train['batch_size'], opt.global_rank, opt.local_rank = \
        '', ckpt, True, opt.cfg_train['total_batch_size'], *apriori  # reinstate
    logger.info('Resuming training from %s' % ckpt)
else:
    assert len(opt.cfg_train['model']) or len(opt.cfg_train['weights']), 'either --cfg or --weights must be specified'
    opt.cfg_train['name'] = 'evolve' if opt.cfg_train['evolve'] else opt.cfg_train['name']
    opt.save_dir = str(increment_path(Path(opt.cfg_train['project']) / cfg_name / (opt.cfg_train['loss_type'] + '_' + opt.cfg_train['model'].replace('.', '_') + '_' + opt.cfg_train['name']), exist_ok=opt.cfg_train['exist_ok'] | opt.cfg_train['evolve']))

# DDP mode
opt.cfg_train['total_batch_size'] = opt.cfg_train['batch_size']
device = select_device(opt.cfg_train['device'], batch_size=opt.cfg_train['batch_size'])
if opt.local_rank != -1:
    assert torch.cuda.device_count() > opt.local_rank
    torch.cuda.set_device(opt.local_rank)
    device = torch.device('cuda', opt.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    assert opt.cfg_train['batch_size'] % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    opt.cfg_train['batch_size'] = opt.cfg_train['total_batch_size'] // opt.world_size

# Hyperparameters
hyp = {**opt.cfg_hyp, **opt.cfg_hyp_custom}

# Train
if not opt.cfg_train['evolve']:
    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.cfg_train['project']}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device, tb_writer, logger)

# Evolve hyperparameters (optional)
else:
    # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'cutcolor': (0, 0.0, 1.0),}  # image cutcolor (probability)

    assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
    opt.cfg_train['notest'], opt.cfg_train['nosave'] = True, True  # only test/save final epoch
    # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
    if opt.cfg_train['bucket']:
        os.system('gsutil cp gs://%s/evolve.txt .' % opt.cfg_train['bucket'])  # download evolve.txt if exists

    for _ in range(300):  # generations to evolve
        if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
            # Select parent(s)
            parent = 'single'  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt('evolve.txt', ndmin=2)
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
            w = fitness(x) - fitness(x).min()  # weights
            if parent == 'single' or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == 'weighted':
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            mp, s = 0.8, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([x[0] for x in meta.values()])  # gains 0-1
            ng = len(meta)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                hyp[k] = float(x[i + 7] * v[i])  # mutate

        # Constrain to limits
        for k, v in meta.items():
            hyp[k] = max(hyp[k], v[1])  # lower limit
            hyp[k] = min(hyp[k], v[2])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        # Train mutation
        results = train(hyp.copy(), opt, device, logger)

        # Write mutation results
        print_mutation(hyp.copy(), results, yaml_file, opt.cfg_train['bucket'])

    # Plot results
    plot_evolution(yaml_file)
    print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
            f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
