# 모듈 import
import os
import argparse
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset    
from mmcv.runner import load_checkpoint
from GPUtil import showUtilization as gpu_usage

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

def empty_cache():
    """
    GPU cache를 비우는 함수
    """
    print("Initial GPU Usage") 
    gpu_usage() 
    print("GPU Usage after emptying the cache") 
    torch.cuda.empty_cache() 
    gpu_usage()

def main(args):
    empty_cache()
    # config file 들고오기
    config_dir = args.config_dir
    config_file = args.config_file
    cfg = Config.fromfile(f'./configs/{config_dir}/{config_file}.py')

    # set wandb name
    if args.wandb:
        cfg.log_config = dict(interval=50,
                              hooks=[dict(type='TextLoggerHook'),
                                     dict(type='WandbLoggerHook',
                                          init_kwargs=dict(project='Object_Detection',
                                                           name=config_file))
                                    ])
    else:
        cfg.log_config = dict(interval=50,
                              hooks=[dict(type='TextLoggerHook')])

    cfg.data.samples_per_gpu = args.batch_size

    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    cfg.work_dir = os.path.join('./work_dirs', config_file)

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=args.max_keep_ckpts, interval=args.ckpts_interval)

    #build dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)

    if args.model_build_type == 0:
        model.init_weights()
    elif args.model_build_type == 1:
        checkpoint_path = os.path.join(cfg.work_dir, f'{args.ckpt_name}.pth')
        load_checkpoint(model, checkpoint_path, map_location='cpu')

    train_detector(model, datasets, cfg, distributed=False, validate=args.validate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, nargs='?', default=1995, help='random seed (default: 1995)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=2, help='batch size (default: 2)')
    parser.add_argument('--model_build_type', type=int, nargs='?', default=0, help='model_build_type (default: 0)\
                                                                                    0(init_weights),\
                                                                                    1(load_checkpoint)')

    # validation
    parser.add_argument('--validate', dest='validate', action='store_true', help='validation')
    parser.add_argument('--no-validate', dest='validate', action='store_false', help='no validation')
    parser.set_defaults(validate=True)

    # wandb
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='wandb')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', help='not use wandb')
    parser.set_defaults(wandb=True)

    #checkpoint
    parser.add_argument('--max_keep_ckpts', type=int, nargs='?', default=5, help='max_keep_ckpts (default: 5)')
    parser.add_argument('--ckpts_interval', type=int, nargs='?', default=1, help='ckpts_interval (default: 1)')
    parser.add_argument('--ckpt_name', type=str, nargs='?', default='latest')

    # directory, file path
    parser.add_argument('--data_dir', type=str, nargs='?', default='/opt/ml/detection/dataset')
    parser.add_argument('--train_file', type=str, nargs='?', default='train_split_0.json')
    parser.add_argument('--valid_file', type=str, nargs='?', default='valid_split_0.json')
    parser.add_argument('--test_file', type=str, nargs='?', default='test.json')
    
    parser.add_argument('--config_dir', type=str, nargs='?', default='swin')
    parser.add_argument('--config_file', type=str, nargs='?', default='cascade_rcnn_swin-t-p4-w7_fpn_ms_mosaic_1x_coco_val')

    args = parser.parse_args()

    # running
    main(args)
    