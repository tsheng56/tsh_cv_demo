
import os
import math
import yaml
import time
import random
import logging
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from threading import Thread

import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

import detect.api.test as test  # import test.py to get mAP after each epoch
from detect.models.experimental import attempt_load
from detect.utils.autoanchor import check_anchors
# from detect.datasets.datasets import create_dataloader
from detect.datasets.yolo_datasets_list import create_dataloader_list
from detect.utils.general import labels_to_class_weights, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, check_img_size, set_logging, one_cycle, colorstr, lr_step
# from detect.utils.google_utils import attempt_download
from detect.loss.yolo_base import ComputeLoss
from detect.utils.plots import plot_images, plot_labels, plot_results, plot_lr_scheduler
from detect.utils.torch_utils import ModelEMA, intersect_dicts, is_parallel
# from detect.utils.wandb_logging.wandb_utils import WandbLogger

def create_model(opt, hyp, weights, nc, device, logger):
    # Set model file
    if opt.cfg_train['model'].endswith('.yaml'):
        from detect.models.yaml_yolo import Model
        model_file = 'detect/models/model_yaml/' + opt.cfg_train['model']
        assert os.path.exists(model_file), f"{model_file} not exists"
    elif opt.cfg_train['model'].endswith('.cfg'):
        from detect.models.darknet_yolo import Model
        model_file = 'detect/models/model_darknet/' + opt.cfg_train['model']
        assert os.path.exists(model_file), f"{model_file} not exists"
    else:
        from detect.models.torch_yolo import Model
        model_file = opt.cfg_train['model']
        assert os.path.exists('detect/models/model_torch/' + model_file + '.py'), f"{model_file} not exists"

    # Model
    pretrained = weights.endswith('.pt') or weights.endswith('.pth')
    if pretrained:
        # with torch_distributed_zero_first(rank):
        #     attempt_download(weights)  # download if not found locally

        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg_train['loss_type'], model_file or ckpt['model'].yaml, ch=3, nc=nc, anchors=opt.cfg_train['anchors']).to(device)  # create
        exclude = ['anchor'] if (opt.cfg_train['model']or hyp.get('anchors')) and not opt.cfg_train['resume'] else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        ckpt = None
        model = Model(opt.cfg_train['loss_type'], model_file, ch=3, nc=nc, anchors=opt.cfg_train['anchors']).to(device)  # create
    # with torch_distributed_zero_first(rank):
    #     check_dataset(opt.cfg_data)  # check
    

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    
    return model, ckpt, pretrained

def create_optimizer_scheduler(opt, model, hyp, total_batch_size, epochs, logger):
    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.cfg_train['adam']:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if hyp['linear_lr']:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    if hyp['lr_cos']:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lr_step(hyp['lr_step'])  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, 600)

    return optimizer, scheduler, lf


def train(hyp, opt, device, tb_writer=None, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
        set_logging(opt.global_rank)

    logger.info(opt)
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Configure
    epochs           = opt.cfg_train['epochs']
    batch_size       = opt.cfg_train['batch_size']
    total_batch_size = opt.cfg_train['total_batch_size']
    weights          = opt.cfg_train['weights']
    rank             = opt.global_rank
    init_seeds(2 + rank)
    cuda             = device.type != 'cpu'
    plots            = not opt.cfg_train['evolve']  # create plots
    nl               = len(opt.cfg_train['anchors'])
   
    # Directories
    save_dir         = Path(opt.save_dir)
    wdir             = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last             = wdir / 'last.pt'
    best             = wdir / 'best.pt'
    results_file     = save_dir / 'results.txt'
    with open(results_file, 'w') as f:
        f.write(('\n' + '%10s' * 17 + '\n') % ('Epoch', 'Batch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_w', 'img_h', 'mp', 'mr', 'map50', 'map', 'box', 'obj', 'cls'))

    # Data set
    is_coco          = opt.cfg_train['data'] == 'coco'
    train_path       = opt.cfg_data['train']
    val_path         = opt.cfg_data['val']
    nc = int(opt.cfg_data['nc'])  # number of classes
    names = opt.cfg_data['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.cfg_train['data'])  # check

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    
    # Logging- Doing this before checking the dataset. Might update opt.cfg_data
    # loggers = {'wandb': None}  # loggers dict
    # if rank in [-1, 0]:
    #     opt.hyp = hyp  # add hyperparameters
    #     run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
    #     wandb_logger = WandbLogger(opt, save_dir.stem, run_id, opt.cfg_data)
    #     loggers['wandb'] = wandb_logger.wandb
    #     opt.cfg_data = wandb_logger.opt.cfg_data
    #     if wandb_logger.wandb:
    #         weights, epochs, hyp = opt.cfg_train['weights'], opt.cfg_train['epochs'], opt.hyp  # WandbLogger might update weights, epochs if resuming


    # Create Model
    model, ckpt, pretrained  = create_model(opt, hyp, weights, nc, device, logger)

    # Scaled hyp
    nbs = opt.cfg_train['nbs']  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers

    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    logger.info(f"Scaled hyp box = {hyp['box']}")
    logger.info(f"Scaled hyp cls = {hyp['cls']}")

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    hyp['obj'] *= (max(imgsz) / 640) ** 2 * 3. / nl  # scale to image size and layers
    logger.info(f"Scaled hyp obj = {hyp['obj']}")


    # Create_Optimizer and Scheduler
    optimizer, scheduler, lf = create_optimizer_scheduler(opt, model, hyp, total_batch_size, epochs, logger)
    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.cfg_train['resume']:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)

        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.cfg_train['sync_bn'] and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader_list(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cfg_train['cache_images'], rect=opt.cfg_train['rect'], rank=rank,
                                            world_size=opt.world_size, workers=opt.cfg_train['workers'],
                                            image_weights=opt.cfg_train['image_weights'], quad=opt.cfg_train['quad'], prefix=colorstr('train: '))
    nb = len(dataloader)  # number of batches

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.cfg_train['data'], nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if opt.cfg_train['test']:
            testloader = create_dataloader_list(val_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cfg_train['cache_images'] and opt.cfg_train['test'], rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.cfg_train['workers'],
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.cfg_train['resume']:
            # labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # # model._initialize_biases(cf.to(device))
            # if plots:
            #     plot_labels(labels, names, save_dir, loggers)
            #     if tb_writer:
            #         tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if opt.cfg_train['autoanchor']:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                
            # model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    if opt.cfg_train['image_weights']:
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training

    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model, opt.cfg_train).compute_loss  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.cfg_train['image_weights']:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        
        if opt.cfg_train['loss_type'] != 'v4':
            pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 10) % ('Epoch', 'Batch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_w', 'img_h'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
        else:
            pbar = enumerate(dataloader)

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, opt.cfg_train['nbs'] / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.cfg_train['multi_scale']:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items, logstrs = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1 and opt.cfg_train['averaged_between_devices']:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.cfg_train['quad']:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 3 + '%10.4g' * 7) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), mem, *mloss, targets.shape[0], imgs.shape[-1], imgs.shape[-2])
                
                if logstrs:
                    logger.info(('\n' + '%10s' * 10) % ('Epoch', 'Batch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_w', 'img_h'))
                    logger.info(s)
                    [logger.info(logstr) for logstr in logstrs]
                else:
                    pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

                #     if tb_writer:
                #         tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                #         tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                # elif plots and ni == 10 and wandb_logger.wandb:
                #     wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                #                                   save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if opt.cfg_train['test'] or final_epoch:  # Calculate mAP
                # wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(opt.cfg_data,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=False,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                #  wandb_logger=wandb_logger,
                                                 wandb_logger=None,
                                                 compute_loss=None,
                                                 is_coco=is_coco)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.cfg_train['name']) and opt.cfg_train['bucket']:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.cfg_train['bucket'], opt.cfg_train['name']))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                # if wandb_logger.wandb:
                #     wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            # wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (opt.cfg_train['save']) or (final_epoch and not opt.cfg_train['evolve']):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        # 'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None
                        }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                # if wandb_logger.wandb:
                #     if ((epoch + 1) % opt.cfg_train['save_period'] == 0 and not final_epoch) and opt.cfg_train['save_period'] != -1:
                #         wandb_logger.log_model(
                #             last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            # if wandb_logger.wandb:
            #     files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
            #     wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
            #                                   if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.cfg_train['data'].endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.cfg_train['data'],
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.cfg_train['bucket']:
            os.system(f"gsutil cp {final} gs://{opt.cfg_train['bucket']}/weights")  # upload
        # if wandb_logger.wandb and not opt.cfg_train['evolve']:  # Log the stripped model
        #     wandb_logger.wandb.log_artifact(str(final), type='model',
        #                                     name='run_' + wandb_logger.wandb_run.id + '_model',
        #                                     aliases=['last', 'best', 'stripped'])
        # wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


