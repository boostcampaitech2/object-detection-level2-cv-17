# 라이브러리 및 모듈 import
from pycocotools.coco import COCO
import argparse
import torch
import os
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
    annotation = '/opt/ml/detection/dataset/test.json'
    data_dir = '/opt/ml/detection/dataset'
    
    coco = COCO(annotation)
    group = stratified_split(coco, args.k_num, True) if args.mode=='valid' else (0, 0)
    mask = group[0]

    val_dataset = TestDataset(coco, data_dir, group, args.img_size)
    checkpoint_path = '/opt/ml/detection/object-detection-level2-cv-17/efficientdet/checkpoints/k0_d4_lambda_lr01/best.pth'
    score_threshold = 0.1
    val_data_loader = DataLoader(
        val_dataset,  
        batch_size=2,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = load_net(checkpoint_path, device, args.img_size, args.version)
    
    outputs = valid_fn(val_data_loader, model, device)
    
    prediction_strings = []
    file_names = []
    
    for i, output in enumerate(outputs):
        prediction_string = ''
        m = mask[i] if args.mode=='valid' else i
        image_info = coco.loadImgs(coco.getImgIds(imgIds=m))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label)-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    
    
    if not os.path.exists('csv'):
        os.makedirs('csv')
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f'csv/submission_{name}.csv', index=None)
    print(submission.head())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='exp', help='enter csv name (default: exp)')
    parser.add_argument('--k_num', type=int, default=0, help='input fold number')
    parser.add_argument('--img_size', type=int, default=512, help='input image size for inference (default: 512)')
    parser.add_argument('--version', type=int, default=4, help='model ver [0~7] (default: 5)')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for inference (default: 4)')
    parser.add_argument('--mode', type = str, default='valid', help='select mode [vaild or test]')
    args = parser.parse_args()
    print(args)
    main(args.name)