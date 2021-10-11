# 모듈 import
import os
import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)       
from mmcv.runner import load_checkpoint

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

def main(args):
    # config file 들고오기
    config_dir = args.config_dir
    config_file = args.config_file
    cfg = Config.fromfile(os.path.join('./configs', config_dir, config_file))

    root=args.data_dir

    # dataset config 수정
    if args.train_dataset_path == 0:
        cfg.data.train.classes = classes
        cfg.data.train.img_prefix = root
        cfg.data.train.ann_file = root + args.train_file
    elif args.train_dataset_path == 1:
        cfg.data.train.classes = classes
        cfg.train_dataset.dataset.img_prefix = root
        cfg.train_dataset.dataset.ann_file = root + args.train_file
    elif args.train_dataset_path == 2:
        cfg.data.train.dataset.classes = classes
        cfg.data.train.dataset.img_prefix = root
        cfg.data.train.dataset.ann_file = root + args.train_file
    

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + args.valid_file # test json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + args.test_file # test json 정보

    if args.img_scale_bool:
        img_scale = args.img_scale
        cfg.data.train.pipeline[2]['img_scale'] = img_scale
        cfg.data.test.pipeline[1]['img_scale'] = img_scale

    cfg.data.samples_per_gpu = 2

    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    cfg.work_dir = os.path.join('./work_dirs', config_file)

    if args.num_classes_path == 0:
        cfg.model.roi_head.bbox_head.num_classes = 10
    elif args.num_classes_path == 1:
        cfg.model.bbox_head.num_classes = 10
    elif args.num_classes_path == 2:
        for head in cfg.model.roi_head.bbox_head:
            head.num_classes = 10

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

    parser.add_argument('--seed', type=int, default=1995, help='random seed (default: 1995)')
    parser.add_argument('--model_build_type', type=int, default='0', help='model_build_type (default: 0)\
                                                                            0(init_weights),\
                                                                            1(load_checkpoint)')
    parser.add_argument('--validate', type=bool, default='False', help='validate (default: False)')

    # img_scale
    parser.add_argument('--img_scale_bool', type=int, default='1', help='img_scale을 변경할 것인지에대한 여부 (default: 1)')
    parser.add_argument('--img_scale', type=tuple, default='(512, 512)', help='img_scale을 변경할 것인지에대한 여부 (default: (512, 512))')
    
    #checkpoint
    parser.add_argument('--max_keep_ckpts', type=int, default='5', help='max_keep_ckpts (default: 5)')
    parser.add_argument('--ckpts_interval', type=int, default='1', help='ckpts_interval (default: 1)')
    parser.add_argument('--ckpt_name', type=str, default='latest')

    # directory, file path
    parser.add_argument('--data_dir', type=str, default='/opt/ml/detection/dataset')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--valid_file', type=str, default='valid_split_0.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--train_dataset_path', type=int, default='0', help='select train_dataset config path (default: 0)\
                                                                            0(cfg.data.train),\
                                                                            1(cfg.train_dataset.dataset),\
                                                                            2(cfg.data.train.dataset)')
    parser.add_argument('--num_classes_path', type=int, default='0', help='select num_classes config path (default: 0)\
                                                                            0(cfg.model.roi_head.bbox_head),\
                                                                            1(cfg.model.bbox_head),\
                                                                            2(cfg.model.roi_head.bbox_head(2개 이상))')
    parser.add_argument('--config_dir', type=str, default='swin')
    parser.add_argument('--config_file', type=str, default='cascade_rcnn_swin-t-p4-w7_fpn_ms_mosaic_1x_coco_val.py')
    