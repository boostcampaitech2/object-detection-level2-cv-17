# 라이브러리 및 모듈 import
import torch
import os
import argparse
import random
import numpy as np


from torch.utils.data import DataLoader
from tqdm import tqdm
from importlib import import_module
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau

from utils import *
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
    # random.seed(seed)

# train function
def train_fn(data_dir, model_dir, args):
    if args.wandb:
        wandb.init(project='efficientdet')
    seed_everything(args.seed)
    
    k = random.randint(0,4) # k-fold 번호 (실행할 때마다 랜덤)

    createFolder(model_dir)
    save_dir = increment_path(os.path.join(model_dir, f'k{k}_{args.name}')) # 실행할 때 val 번호(k)를 알아야 나중에 metric 할 수 있다.
    createFolder(save_dir)

    annotation = os.path.join(data_dir,'train.json')
    
    train_group = stratified_split(annotation, k, False)  # train, val set stratified 하게 나누고 train set 불러오기
    train_dataset = CustomDataset(annotation, data_dir, train_group, args.img_size)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    valid_group = stratified_split(annotation, k, True)  # valid set 불러오기
    valid_dataset = CustomDataset(annotation, data_dir, valid_group, args.img_size)

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # model = get_net(checkpoint_path='/opt/ml/detection/object-detection-level2-cv-17/efficientdet/checkpoints/d5_e50_con/epoch_20.pth', box_weight=args.box_weight, img_size=args.img_size)
    model = get_net(img_size=args.img_size)
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
    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=args.patience)

    num_epochs = args.epochs
    best_loss = 1000
    loss_hist = Averager()
    loss_hist_box = Averager()
    loss_hist_cls = Averager()
    loss_hist_valid = Averager()
    loss_hist_box_valid = Averager()
    loss_hist_cls_valid = Averager()
    # model.train()
    
    for epoch in range(num_epochs):
        model.train()

        loss_hist.reset()
        loss_hist_box.reset()
        loss_hist_cls.reset()
        
        for images, targets, image_ids in tqdm(train_data_loader):
            
            images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
            images = images.to(device).float()
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]
            target = {"bbox": boxes, "cls": labels}
            
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
        
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            for images, targets, image_ids in tqdm(valid_data_loader):
                
                images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
                images = images.to(device).float()
                boxes = [target['boxes'].to(device).float() for target in targets]
                labels = [target['labels'].to(device).float() for target in targets]
                target = {"bbox": boxes, "cls": labels, "img_scale": None, "img_size": None}
                # print(target['cls'])
                # calculate loss
                loss, cls_loss, box_loss, detection = model(images, target).values()
                loss_value = loss.detach().item()
                loss_value_box = box_loss.detach().item()
                loss_value_cls = cls_loss.detach().item()
                
                loss_hist_valid.send(loss_value)
                loss_hist_box_valid.send(loss_value_box)
                loss_hist_cls_valid.send(loss_value_cls)
                
        print(f"Validation | lr: {current_lr} loss: {loss_hist_valid.value} box_loss: {loss_hist_box_valid.value} cls_loss: {loss_hist_cls_valid.value}")
        save_path = f'{save_dir}/epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), save_path)

        if loss_hist_valid.value < best_loss:
            best_save_path = f'{save_dir}/best.pth'
            torch.save(model.state_dict(), best_save_path)
            best_loss = loss_hist_valid.value

        if epoch > 0 and args.wandb:
            wandb.log({'loss': loss_hist.value, 'box_loss': loss_hist_box.value, 'cls_loss': loss_hist_cls.value,
            'valid_loss': loss_hist_valid.value, 'valid_box_loss': loss_hist_box_valid.value, 'valid_cls_loss': loss_hist_cls_valid.value})



if __name__ == '__main__':
    
    # data_dir = '../../dataset'   # 데이터 경로 
    # train_load_path = None  # train시 checkpoint 경로

    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--wandb', type=bool, default=True, help='use wandb or not')
    parser.add_argument('--model', type=int, default=0, help='select which model (0~7)')
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train (default: 14)')
    parser.add_argument('--batch_size', type=int, default=12, help='input batch size for training (default: 12)')
    parser.add_argument('--img_size', type=int, default=512, help='input image size for training (default: 512)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 1e-3)')
    parser.add_argument('--box_weight', type=int, default=50, help='box weight (default: 50)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay for SGD (default: 0.0005)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type=int, default=5, help='check early stopping point (default: 5)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/detection/dataset'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './checkpoints'))
    

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir


    loss = train_fn(data_dir, model_dir, args)

