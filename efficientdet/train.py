# 라이브러리 및 모듈 import
import torch
import os
import argparse
import random
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from importlib import import_module
from torch.optim.lr_scheduler import StepLR, OneCycleLR

from utils import collate_fn, Averager
from model import get_net
from dataset import CustomDataset

import wandb

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# train function
def train_fn(data_dir, model_dir, args):
    wandb.init(project='efficientdet')
    seed_everything(args.seed)
    annotation = os.path.join(data_dir,'train.json')
    train_dataset = CustomDataset(annotation, data_dir, args.img_size)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # model = get_net(checkpoint_path='/opt/ml/detection/object-detection-level2-cv-17/efficientdet/checkpoints/d5_e50_con/epoch_20.pth', box_weight=args.box_weight, img_size=args.img_size)
    model = get_net()
    model.to(device)

    if args.optimizer == 'Adam':
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
    elif args.optimizer == 'SGD':
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # scheduler = OneCycleLR(optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.002, epochs=args.epochs, steps_per_epoch=len(train_data_loader))
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    num_epochs = args.epochs

    loss_hist = Averager()
    loss_hist_box = Averager()
    loss_hist_cls = Averager()
    model.train()
    
    for epoch in range(num_epochs):
        loss_hist.reset()
        loss_hist_box.reset()
        loss_hist_cls.reset()
        
        for images, targets, image_ids in tqdm(train_data_loader):
            
            images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
            images = images.to(device).float()
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]
            target = {"bbox": boxes, "cls": labels}
            # print(target['cls'])
            # calculate loss
            loss, cls_loss, box_loss = model(images, target).values()
            loss_value = loss.detach().item()
            loss_value_box = box_loss.detach().item()
            loss_value_cls = cls_loss.detach().item()
            
            loss_hist.send(loss_value)
            loss_hist_box.send(loss_value_box)
            loss_hist_cls.send(loss_value_cls)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            
            optimizer.step()
        scheduler.step()    

        current_lr = get_lr(optimizer)
        print(f"Epoch #{epoch+1} lr: {current_lr} loss: {loss_hist.value} box_loss: {loss_hist_box.value} cls_loss: {loss_hist_cls.value}")
        save_path = f'./{model_dir}/{args.name}/epoch_{epoch+1}.pth'
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), save_path)
        if epoch > 0:
            wandb.log({'loss': loss_hist.value, 'box_loss': loss_hist_box.value, 'cls_loss': loss_hist_cls.value})



if __name__ == '__main__':
    
    # data_dir = '../../dataset'   # 데이터 경로 
    # train_load_path = None  # train시 checkpoint 경로

    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--model', type=int, default=0, help='select which model (0~7)')
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train (default: 14)')
    parser.add_argument('--batch_size', type=int, default=12, help='input batch size for training (default: 12)')
    parser.add_argument('--img_size', type=int, default=512, help='input image size for training (default: 512)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 1e-3)')
    parser.add_argument('--box_weight', type=int, default=50, help='box weight (default: 50)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay for SGD (default: 0.0005)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type=int, default=10, help='check early stopping point (default: 10)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/detection/dataset'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './checkpoints'))
    

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir


    loss = train_fn(data_dir, model_dir, args)

