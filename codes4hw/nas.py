# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime,os
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine_lambda import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

# import models
# import models_v4, models_shrink
# from comp import evo_deit, evit, dynamic_vit
import models_small_timm3_lambda
import utils
import tome
import DiffRate 
from  models_small_timm3_lambda import Block, Block_act_compensation
import re

def extract_numbers(text):
    """_summary_
    从字符串中提取blocks.后面的数字
    Args:
        text (_type_): 输入字符串,如blocks.3.xxx
    
    Returns:
        list: 返回提取的数字列表,如[3]
    """
    if not isinstance(text,str):
        raise []
    
    pattern = r'blocks\.(\d+)'
    matches = re.findall(pattern, text)
    return [int(num) for num in matches] if matches else []
 
def swap_activation(model, target_id, current=""):
    name_to_child = dict(model.named_children())
    for name, child in model.named_children():
        fqn = f"{current}.{name}" if current else name
        if "blocks" in fqn and len(extract_numbers(fqn)) > 0:
            fqn_id = extract_numbers(fqn)[0]
        else:
            fqn_id = -1
            
        if isinstance(child, Block) and (fqn_id in target_id):
            print(f"Swapping activation for block {fqn_id}")
            dim = child.dim
            num_heads = child.num_heads
            qkv_bias = child.qkv_bias
            update_model = Block_act_compensation(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
            # load
            update_model.load_ckpt_from_orign(child)
            setattr(model, name, update_model)
            del child
        else:
            swap_activation(child, target_id, current=fqn)
                
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False) # deit base需要设置为True
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--prune_layer',type=str, default='[1,2,3,4]', help='the pruning layer')
    parser.add_argument('--done_layer',type=str, default='[]', help='the pruned layer')
    parser.add_argument('--frozen_stages',type=int, default=-1, help='the frozen layer')
    parser.add_argument('--pretrained',action='store_true', default=False, help='vit backbone need this')
    parser.add_argument('--act_prune',default='[]', type=str, help='the pruning layer')
    return parser


def main(args):
    # utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        prune_layer=eval(args.prune_layer),
        done_layer=eval(args.done_layer),
        frozen_stages=args.frozen_stages,
    )

    ### Uncomment to use ToMe, ONLY for evaluation.
    # tome.patch.timm(model,start_idx=1) # deit-base, remove [0,1,3,4,6]
    # model.r = 24 # or 28

    # tome.patch.timm(model,start_idx=0) # deit-base, remove [0,1,3,4,6,9]
    # model.r = 14

    # tome.patch.timm(model,start_idx=1) # deit-small, remove [0,2,4]
    # model.r = 22

    # tome.patch.timm(model,start_idx=1) # deit-small, remove [0,2,4,6]
    # model.r = 19

    # tome.patch.timm(model,start_idx=1) # deit-tiny, remove [0,2,4]
    # model.r = 19

    # tome.patch.timm(model,start_idx=1) # deit-tiny, remove [0,2,4,6]
    # model.r = 14

    # DiffRate.patch.deit(model, prune_granularity=4, merge_granularity=4)
    # model_name_dict = {
    #     'vit_deit_tiny_patch16_224':'ViT-T-DeiT',
    #     'vit_deit_small_patch16_224':'ViT-S-DeiT',
    #     'deit_small_patch16_224':'ViT-S-DeiT',
    #     'vit_deit_base_patch16_224': 'ViT-B-DeiT',
    #     'vit_base_patch16_mae': 'ViT-B-MAE',
    #     'vit_large_patch16_mae': 'ViT-L-MAE',
    #     'vit_huge_patch14_mae': 'ViT-H-MAE',
    #     'caformer_s36':'CAFormer-S36',
    # }
    # if True:
    #     with open('compression_rate.json', 'r') as f:
    #         compression_rate = json.load(f) 
    #         model_name = model_name_dict['deit_small_patch16_224']
    #         # if not str(args.target_flops) in compression_rate[model_name]:
    #         #     raise ValueError(f"compression_rate.json does not contaion {model_name} with {args.target_flops}G flops")
    #         prune_kept_num = eval(compression_rate[model_name][str(2.9)]['prune_kept_num'])
    #         merge_kept_num = eval(compression_rate[model_name][str(2.9)]['merge_kept_num'])
    #         model.set_kept_num(prune_kept_num, merge_kept_num)
                    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        if 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        elif 'net' in checkpoint.keys():
            checkpoint_model = checkpoint['net']
        # checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'head_cls.weight', 'head_cls.bias']:
            if (k in checkpoint_model and k in state_dict) and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
    print("ready to prune activation"+str(eval(args.act_prune)))
    swap_activation(model, target_id=eval(args.act_prune))
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, epoch=args.epochs-1)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    # print(model)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             
        if epoch == 5:
            test_stats = evaluate(data_loader_val, model, device, epoch)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            return test_stats['acc1']
        # if epoch in [280,290,295]:
        #     if args.output_dir:
        #         checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
        #         for checkpoint_path in checkpoint_paths:
        #             utils.save_on_master({
        #                 'model': model_without_ddp.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'epoch': epoch,
        #                 'model_ema': get_state_dict(model_ema),
        #                 'scaler': loss_scaler.state_dict(),
        #                 'args': args,
        #             }, checkpoint_path)

        # if max_accuracy < test_stats["acc1"]:
        #     max_accuracy = test_stats["acc1"]
        #     if args.output_dir:
        #         checkpoint_paths = [output_dir / 'best_checkpoint.pth']
        #         for checkpoint_path in checkpoint_paths:
        #             utils.save_on_master({
        #                 'model': model_without_ddp.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'epoch': epoch,
        #                 'model_ema': get_state_dict(model_ema),
        #                 'scaler': loss_scaler.state_dict(),
        #                 'args': args,
        #             }, checkpoint_path)
            
        # print(f'Max accuracy: {max_accuracy:.2f}%')

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}
        
        
        
        
        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
def get_acc(args, candidate_module):
    act_prune = []
    attn_prune = []
    for i in candidate_module:
        if i < 12:
            act_prune.append(i)
        else:
            attn_prune.append(i - 12)
    args.act_prune = str(act_prune)
    args.prune_layer = str(attn_prune)
    acc1 = main(args)
    return acc1

def iterative_swap(args, candidates, k, initial_subset=None, max_iters=50):
    """
    使用局部搜索（Iterative Swap）在 candidates 中选择 k 个，
    以使 evaluate_func(subset) 最大。
    
    candidates: list or range of all module indices (e.g. range(12))
    k: number of modules to choose (e.g. 8)
    evaluate_func: function to evaluate subset -> float (acc)
    performance_table: used for simulate evaluation
    initial_subset: optional starting solution
    max_iters: limit on total iteration to avoid infinite loops
    """
    # Step 1: 初始化解
    if initial_subset is None:
        # Random initial solution
        init = random.sample(candidates, k)
    else:
        init = initial_subset[:]
        
    S = set(init)
    print(f"Initial subset: {S}")
    performance_table = {}
    best_acc = get_acc(args, S)
    performance_table[str(S)] = best_acc
    print(f"Initial subset: {S}")
    print(f"Initial acc: {best_acc:.3f}")
    improved = True
    iteration = 0
    
    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        
        # Step 2: 遍历邻域 (swap i in S with j in C \ S)
        # 为了避免重复评价大量组合，可以采用 'first improvement' 策略
        for i in list(S):
            for j in candidates:
                if j not in S:
                    # neighbor solution
                    S_prime = set(S)
                    S_prime.remove(i)
                    S_prime.add(j)
                    
                    # Evaluate
                    if str(S_prime) in performance_table:
                        acc_prime = performance_table[str(S)]
                    else:
                        acc_prime = get_acc(args, S_prime)
                        performance_table[str(S)] = best_acc
                    print("Current subset:", S)
                    print(f"Try swap {i} -> {j}: {acc_prime:.3f}")
                    if acc_prime > best_acc:
                        # 找到更好解 -> 接受并更新
                        S = S_prime
                        best_acc = acc_prime
                        improved = True
                        print("-" * 20)
                        print(f"Improved! New best acc: {best_acc:.3f}")
                        print('subset prime:', S)
                        print("-" * 20)
                        break   # 跳出内层
            if improved:
                break   # 跳出外层，重新从新解的邻域开始
    print(f"Best subset: {S}")
    print(f"Best acc: {best_acc:.3f}")
    return list(S), best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir,datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # main(args)
    utils.init_distributed_mode(args)
    args.no_act_layer = '[]'
    args.no_attn_layer = '[]'
    # 0 - 11 表示 act
    # 12 - 23 表示 attn
    
    if utils.is_main_process():
        with open(args.output_dir+'/search.txt','a') as f:
            f.write("begin to search\n")
    candidata_set =[13, 23, 21, 9, 22, 10, 20, 11, 19, 18, 17, 8]
    init_set = [13, 23, 21, 9, 22, 10, 20, 11]
    iterative_swap(args, candidata_set, 8, initial_subset= init_set,max_iters=50)
