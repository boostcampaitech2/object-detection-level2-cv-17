# 라이브러리 및 모듈 import
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import collate_fn, Averager
from model import get_net
from dataset import CustomDataset

import wandb

# train function
def train_fn(num_epochs, train_data_loader, optimizer, model, device, clip=35):
    wandb.init(project='efficientdet')

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                
                optimizer.step()

        print(f"Epoch #{epoch+1} loss: {loss_hist.value} box_loss: {loss_hist_box.value} cls_loss: {loss_hist_cls.value}")
        torch.save(model.state_dict(), f'epoch/epoch_{epoch+1}.pth')
        if epoch > 0:
            wandb.log({'loss': loss_hist.value, 'box_loss': loss_hist_box.value, 'cls_loss': loss_hist_cls.value})


annotation = '/opt/ml/detection/dataset/train.json'
data_dir = '/opt/ml/detection/dataset'
train_dataset = CustomDataset(annotation, data_dir)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = get_net(box_weight=50)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 50

loss = train_fn(num_epochs, train_data_loader, optimizer, model, device)