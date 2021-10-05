import warnings
warnings.filterwarnings(action='ignore')

import argparse
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data as data_


from dataset import TrainCustom, CustomDataset
from model import FasterRCNNVGG16, FasterRCNNTrainer
from utils import collate_fn

# epochs=14
# learning_rate = 1e-3
# weight_decay = 0.0005
# lr_decay = 0.1

# rpn_sigma = 3.     # sigma for l1_smooth_loss (RPN loss)
# roi_sigma = 1.     # sigma for l1_smooth_loss (ROI loss)

# data_dir = '../../dataset'   # 데이터 경로 
# train_load_path = None  # train시 checkpoint 경로

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    # Train dataset 불러오기
    # dataset = TrainDataset()
    annotation = os.path.join(data_dir,'train.json')
    dataset = TrainCustom(annotation, data_dir, transforms=True)
    print('load data')
    dataloader = data_.DataLoader(dataset, 
                                    batch_size=1,     # only batch_size=1 support
                                    shuffle=False, 
                                    pin_memory=False,
                                    num_workers=4,
                                    # collate_fn=collate_fn,
                                    )

    # faster rcnn 불러오기
    faster_rcnn = FasterRCNNVGG16(args.lr, args.weight_decay).cuda()
    print('model construct completed')

    # faster rcnn trainer 불러오기
    trainer = FasterRCNNTrainer(faster_rcnn, args.rpn_sigma, args.roi_sigma).cuda()

    # checkpoint load
    if args.train_load_path:
        trainer.load(args.train_load_path)
        print('load pretrained model from %s' % args.train_load_path)

    lr_ = args.lr
    best_loss = 1000
    device = torch.device("cuda")
    for epoch in range(args.epochs):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in enumerate(tqdm(dataloader)):
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda() ### 실패
            trainer.train_step(img, bbox, label, scale)
        
        losses = trainer.get_meter_data()
        print(f"Epoch #{epoch+1} loss: {losses}")
        if losses['total_loss'] < best_loss :
            trainer.save(save_path=f'{model_dir}/{args.name}.pth')
            
        if epoch == 9:
            trainer.faster_rcnn.scale_lr(args.lr_decay)
            lr_ = lr_ * args.lr_decay

        if epoch == 13: 
            break
if __name__ == '__main__':
    # epochs=14
    # learning_rate = 1e-3
    # weight_decay = 0.0005
    # lr_decay = 0.1

    # rpn_sigma = 3.     # sigma for l1_smooth_loss (RPN loss)
    # roi_sigma = 1.     # sigma for l1_smooth_loss (ROI loss)

    # data_dir = '../../dataset'   # 데이터 경로 
    # train_load_path = None  # train시 checkpoint 경로

    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train (default: 14)')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 1)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='lr decay (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay (default: 0.0005)')
    parser.add_argument('--rpn_sigma', type=float, default=3., help='rpn_sigma (default: 3.)')
    parser.add_argument('--roi_sigma', type=float, default=1., help='roi_sigma (default: 1.)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type=int, default=10, help='check early stopping point (default: 10)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/detection/dataset'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './checkpoints'))
    parser.add_argument('--train_load_path', type=str, default=os.environ.get('SM_CHECK_DIR', None))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
