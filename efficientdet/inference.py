# 라이브러리 및 모듈 import
from pycocotools.coco import COCO
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import TestDataset, CustomDataset
from model import load_net
from utils import collate_fn, stratified_split


# valid function
def valid_fn(val_data_loader, model, device):
    outputs = []
    for images, image_ids in tqdm(val_data_loader):
        # gpu 계산을 위해 image.to(device)       
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        output = model(images)
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1]})
    return outputs

def main(name):
    annotation = '/opt/ml/detection/dataset/train.json'
    data_dir = '/opt/ml/detection/dataset'
    
    coco = COCO(annotation)
    group = stratified_split(coco, 1, True)
    mask = group[0]

    val_dataset = TestDataset(coco, data_dir, group, args.img_size)
    checkpoint_path = '/opt/ml/detection/object-detection-level2-cv-17/efficientdet/checkpoints/validation_k1/epoch_49.pth'
    score_threshold = 0.1
    val_data_loader = DataLoader(
        val_dataset,  
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = load_net(checkpoint_path, device, args.img_size)
    
    outputs = valid_fn(val_data_loader, model, device)
    
    prediction_strings = []
    file_names = []
    
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=mask[i]))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label)-1) + ' ' + str(score) + ' ' + str(box[0]*2) + ' ' + str(
                    box[1]*2) + ' ' + str(box[2]*2) + ' ' + str(box[3]*2) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f'csv/submission_{name}.csv', index=None)
    print(submission.head())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='exp', help='enter csv name (default: exp)')
    parser.add_argument('--img_size', type=int, default=512, help='input image size for training (default: 512)')
    args = parser.parse_args()
    print(args)
    main(args.name)