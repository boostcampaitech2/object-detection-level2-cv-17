import os
import torch
import argparse
import pandas as pd
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pycocotools.coco import COCO
from GPUtil import showUtilization as gpu_usage

def empty_cache():
    """
    GPU cache를 비우는 함수
    """
    print("Initial GPU Usage") 
    gpu_usage() 
    print("GPU Usage after emptying the cache") 
    torch.cuda.empty_cache() 
    gpu_usage()

def save_csv(output, cfg, epoch):
    """
    Inference 결과를 csv 파일로 저장하는 함수
    """
    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    # for i, out in enumerate(output):
    for i, out in zip(img_ids, output):
        prediction_string = ''
        try:
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        except:
            continue
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)

def main(args):
    empty_cache()
    # config file 들고오기
    config_dir = args.config_dir
    config_file = args.config_file
    cfg = Config.fromfile(f'./configs/{config_dir}/{config_file}.py')

    epoch = args.ckpt_name

    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = args.batch_size

    cfg.seed=args.seed
    cfg.gpu_ids = [1]
    cfg.work_dir = os.path.join('./work_dirs', config_file)

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산

    save_csv(output, cfg, epoch) # save csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, nargs='?', default=1995, help='random seed (default: 1995)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=2, help='batch size (default: 2)')

    #checkpoint
    parser.add_argument('--ckpt_name', type=str, nargs='?', default='latest')

    # directory, file path
    parser.add_argument('--data_dir', type=str, nargs='?', default='/opt/ml/detection/dataset')

    parser.add_argument('--config_dir', type=str, nargs='?', default='swin')
    parser.add_argument('--config_file', type=str, nargs='?', default='cascade_rcnn_swin-t-p4-w7_fpn_ms_mosaic_1x_coco_val')

    args = parser.parse_args()

    # running
    main(args)