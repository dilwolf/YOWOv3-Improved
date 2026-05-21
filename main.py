import os
import csv
import copy
import tqdm
import yaml
import torch
import wandb
import datetime
import numpy as np
from utils import util
from torch.utils import data
from utils.dataset import Dataset
from argparse import ArgumentParser
from model.TSN.YOWOv3 import build_yowov3
from utils.loss import build_loss, LinearWarmup

def train_model(config, aug_config, args):
    # Build train dataset & loader
    sampler = None
    dataset = Dataset(config=config, aug_config=aug_config, phase='train')
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=True) if args.distributed else None
    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=config['num_workers'], pin_memory=True, collate_fn=Dataset.collate_fn)

    # Build model
    device = torch.device("cuda", args.local_rank)
    model = build_yowov3(config)
    model.to(device)

    # Save effective config after model loading, including CLI overrides.
    # Exclude backbone/loss lookup tables — active choices are captured by backbone2D/backbone3D/loss keys.
    if args.local_rank == 0:
        class _InlineListDumper(yaml.Dumper):
            pass
        _InlineListDumper.add_representer(
            list, lambda d, v: d.represent_sequence('tag:yaml.org,2002:seq', v, flow_style=True)
        )
        saved_config = {**config, 'batch_size': args.batch_size}
        with open(os.path.join(config['save_folder'], 'config.yaml'), 'w') as f:
            yaml.dump(saved_config, f, Dumper=_InlineListDumper, default_flow_style=False, sort_keys=False)

    # EMA only on rank 0
    ema = util.ModelEMA(model) if args.local_rank == 0 else None
    
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)  # Ensure model is on device after SyncBatchNorm conversion
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Loss and optimizer
    criterion = build_loss(model, config)
    g = [[], [], []]
    bn_types = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if n == 'bias': g[2].append(p)
            elif n == 'weight' and isinstance(m, bn_types): g[1].append(p)
            else: g[0].append(p)
    optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
    optimizer.add_param_group({'params': g[2], 'weight_decay': 0.0})

    warmup = LinearWarmup(config)
    acc_grad = config['acc_grad']
    adjust_schedule = set(config['adjustlr_schedule'])
    lr_decay = config['lr_decay']
    max_epoch = config['max_epoch']
    save_folder = config['save_folder']

    scaler = torch.amp.GradScaler()
    best_metric = 0.0

    if args.wandb and args.local_rank == 0:
        wandb.init(
            project=args.project,
            name=args.run_name,
            mode="offline", # save logs locally
            dir="weights/model_checkpoint/",
            config={
                'backbone2D': config['backbone2D'], 'backbone3D': config['backbone3D'],
                'fusion_module': config['fusion_module'], 'dataset': config['data_root'],
                'img_size': config['img_size'], 'clip_length': config['clip_length'],
                'lr': config['lr'], 'weight_decay': config['weight_decay'],
                'batch_size': args.batch_size, 'acc_grad': config['acc_grad'],
                'max_epoch': config['max_epoch'],
                'scale_cls_loss': config.get('scale_cls_loss'),
                'scale_box_loss': config.get('scale_box_loss'),
                'scale_dfl_loss': config.get('scale_dfl_loss'),
                'class_names': config['idx2name'],
            }
        )

    with open('weights/model_checkpoint/results.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'box', 'cls', 'dfl', 'mAP', 'mAP@50', 'Recall'])
            logger.writeheader()

        for epoch in range(1, max_epoch + 1):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)

            # Initialize average meters for loss components
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()

            p_bar = train_loader
            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('Epoch', 'GPU_mem', 'Box Loss', 'Cls Loss', 'DFL Loss'))
                p_bar = tqdm.tqdm(p_bar, total=len(train_loader))     

            updates = 0
            optimizer.zero_grad(set_to_none=True)

            for it, (clips, bboxes, labels, _) in enumerate(p_bar, start=1):
                clips = clips.cuda(args.local_rank, non_blocking=True)
                bboxes = [b.cuda(args.local_rank, non_blocking=True) for b in bboxes]
                labels = [l.cuda(args.local_rank, non_blocking=True) for l in labels]

                # Build targets
                targets = []
                for i, (bb, lb) in enumerate(zip(bboxes, labels)):
                    t = torch.zeros(bb.size(0), 5 + lb.size(1), device=bb.device)
                    t[:, 0] = i
                    t[:, 1:5] = bb
                    t[:, 5:] = lb
                    targets.append(t)
                targets = torch.cat(targets, 0)

                # Forward pass with AMP
                with torch.amp.autocast(device_type='cuda'):
                    outs = model(clips)
                    loss_cls, loss_box, loss_dfl = criterion(outs, targets)
                    loss = loss_cls + loss_box + loss_dfl
                    loss = loss / acc_grad  # Gradient accumulation

                # Backward pass
                scaler.scale(loss).backward()

                # Optimizer step every `acc_grad` steps
                if it % acc_grad == 0:
                    updates += 1
                    if epoch == 1:
                        warmup(optimizer, updates)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(model)

                # Update loss meters
                avg_box_loss.update(loss_box.item(), clips.size(0))
                avg_cls_loss.update(loss_cls.item(), clips.size(0))
                avg_dfl_loss.update(loss_dfl.item(), clips.size(0))

                torch.cuda.synchronize()

                # Progress bar on rank 0
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}GB'
                    log_line = (
                        f'{epoch}/{max_epoch}',
                        memory,
                        f'{avg_box_loss.avg:.4f}',
                        f'{avg_cls_loss.avg:.4f}',
                        f'{avg_dfl_loss.avg:.4f}'
                    )
                    p_bar.set_description(('%10s' * 5) % log_line)

            # LR scheduler
            if epoch in adjust_schedule:
                for g in optimizer.param_groups:
                    g['lr'] *= lr_decay

            # Evaluation and saving (only on rank 0)
            if args.local_rank == 0:
                m_rec, map50, map50_90, ap50_all, ap_all = eval_model(config=config, args=args, model=ema.ema)
                logger.writerow({'epoch': str(epoch).zfill(3),
                                'box': str(f'{avg_box_loss.avg:.3f}'),
                                'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                'mAP': str(f'{map50_90:.3f}'),
                                'mAP@50': str(f'{map50:.3f}'),
                                'Recall': str(f'{m_rec:.3f}')})
                log.flush()
                if args.wandb:
                    per_class = {}
                    for c, name in config['idx2name'].items():
                        if not np.isnan(ap50_all[c]): per_class[f'val/AP50/{name}']    = float(ap50_all[c])
                        if not np.isnan(ap_all[c]):   per_class[f'val/AP50_95/{name}'] = float(ap_all[c])
                    wandb.log({
                        'epoch': epoch,
                        'train/loss_box': avg_box_loss.avg,
                        'train/loss_cls': avg_cls_loss.avg,
                        'train/loss_dfl': avg_dfl_loss.avg,
                        'train/loss': avg_box_loss.avg + avg_cls_loss.avg + avg_dfl_loss.avg,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'val/recall': m_rec,
                        'val/mAP50': map50,
                        'val/mAP50_95': map50_90,
                        **per_class,
                    })
                save = {
                    'epoch': epoch,
                    'class_names': config['idx2name'],
                    'model': copy.deepcopy(ema.ema.state_dict())
                }
                if map50_90 > best_metric:
                    if args.wandb:
                        wandb.run.summary.update({'best_mAP50_95': map50_90, 'best_mAP50': map50, 'best_epoch': epoch})
                    best_metric = map50_90
                    torch.save(save, os.path.join(save_folder, 'best.pth'))
                    print(f"New best {best_metric:.4f}, saved.")
                torch.save(save, os.path.join(save_folder, 'last.pth'))
                del save
                
            # Synchronize all ranks before next epoch
            if args.distributed:
                torch.distributed.barrier()
    
    if args.wandb and args.local_rank == 0:
        wandb.finish()

    # if args.strip_optimizer and args.local_rank == 0:
    #     util.strip_optimizer(os.path.join(save_folder, 'best.pth'))  # strip optimizers
    #     util.strip_optimizer(os.path.join(save_folder, 'last.pth'))  # strip optimizers

@torch.no_grad()
def eval_model(config, args, model=None):
    class_names = config['idx2name']
    img_size = config['img_size']
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size     
    height, width = img_size[0], img_size[1]
    
    # Build val dataset & loader
    dataset = Dataset(config=config, phase='valid')
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=config['num_workers'], pin_memory=True, collate_fn=Dataset.collate_fn)
    
    # Load model
    # if model is None:
    #     config["pretrain_path"] = str(args.weight)
    #     model = build_yowov3(config)

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()
    n_iou = iou_v.numel()
    m_rec = 0.
    map50 = 0.
    map50_90 = 0.
    metrics = []
    model.cuda()
    model.eval()
    p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 5) % ('', '', 'recall', 'mAP@0.5', 'mAP@0.95'))

    for batch_clip, batch_bboxes, batch_labels, shapes in p_bar:
        batch_clip = batch_clip.cuda()

        targets = []
        for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
            target = torch.Tensor(bboxes.shape[0], 6)
            target[:, 0] = i
            target[:, 1] = labels
            target[:, 2:] = bboxes
            targets.append(target)

        targets = torch.cat(targets, dim=0).cuda()
        outputs = model(batch_clip) # Inference
        outputs = util.non_max_suppression(outputs, 0.005, 0.5) # NMS

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, torch.zeros(0).cuda(), torch.zeros(0).cuda(), labels[:, 0]))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], (height, width), shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                util.scale(tbox, (height, width), shapes[i][0], shapes[i][1])
                correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    n_classes = len(class_names)
    ap50_all = np.full(n_classes, float('nan'))
    ap_all   = np.full(n_classes, float('nan'))
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    if len(metrics) and metrics[0].any():
        tp, fp, m_rec, map50, map50_90, ap50_cls, ap_cls, cls_indices = util.compute_ap(*metrics)
        ap50_all[cls_indices] = ap50_cls
        ap_all[cls_indices]   = ap_cls
    print(('%10s' + '%10s' + '%10.3g' * 3) % ('', '',  m_rec, map50, map50_90), flush=True)
    def print_metrics(tp, fp, ap50, ap, class_names):
        print("=" * 75, flush=True)
        print(f"{'Class':<20}{'True Positives':>13}{'False Positives':>17}{'AP@0.5':>12}{'AP@0.5:0.95':>13}", flush=True)
        print("-" * 75, flush=True)
        for c in range(len(class_names)):
            ap50_str = f"{ap50[c]:.3f}" if not np.isnan(ap50[c]) else "  N/A"
            ap_str   = f"{ap[c]:.3f}"   if not np.isnan(ap[c])   else "  N/A"
            print(f"{class_names[c]:<20}{int(tp[c]):>13}{int(fp[c]):>17}{ap50_str:>12}{ap_str:>13}", flush=True)
        print("=" * 75, flush=True)
    print_metrics(tp, fp, ap50_all, ap_all, class_names)
    model.float()

    return float(m_rec), float(map50), float(map50_90), ap50_all, ap_all

def main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--weight', default="weights/model_checkpoint/best.pt", type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--strip_optimizer', action='store_true')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='yowov3', help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
    args = parser.parse_args()

    # ---------------------------
    # Distributed settings
    # ---------------------------
    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=20)
        )

    # ---------------------------
    # Build config (all ranks)
    # ---------------------------
    config, aug_config = util.build_config()

    # Rank 0 only during file modification and caching
    if args.local_rank == 0:
        util.update_mode_list(config)
        util.build_label_cache(config, phase='train')
        util.build_label_cache(config, phase='valid')


    # IMPORTANT: synchronize after rank-0 operation if multi-GPU training
    if args.distributed:
        torch.distributed.barrier()

    util.setup_seed()
    util.setup_multi_processes()

    # ---------------------------
    # Run mode
    # ---------------------------
    try:
        if args.train:
            train_model(config, aug_config, args)

        if args.eval:
            # Only rank 0 should run evaluation in distributed mode
            if not args.distributed or args.local_rank == 0:
                if args.wandb:
                    wandb.init(project=args.project, name=args.run_name, job_type='eval')
                m_rec, map50, map50_90, ap50_all, ap_all = eval_model(config=config, args=args)
                if args.wandb:
                    per_class = {}
                    for c, name in config['idx2name'].items():
                        if not np.isnan(ap50_all[c]): per_class[f'val/AP50/{name}']    = float(ap50_all[c])
                        if not np.isnan(ap_all[c]):   per_class[f'val/AP50_95/{name}'] = float(ap_all[c])
                    wandb.log({'val/recall': m_rec, 'val/mAP50': map50, 'val/mAP50_95': map50_90, **per_class})
                    wandb.finish()

            if args.distributed:
                torch.distributed.barrier()

    finally:
        if args.distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
