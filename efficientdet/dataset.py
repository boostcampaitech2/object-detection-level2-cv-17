# 라이브러리 및 모듈 import
from albumentations.augmentations.transforms import Normalize
from numpy.lib.type_check import imag

import numpy as np
import cv2
import os
import torch
import albumentations as A

from pycocotools.coco import COCO
from torch.utils.data import  Dataset
from albumentations.pytorch import ToTensorV2

# Albumentation을 이용, augmentation 선언
def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_test_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(p=1.0)
    ])
# CustomDataset class 선언

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, coco, data_dir, group, img_size=512):
        super().__init__()
        self.data_dir = data_dir
        self.mask = group[0]
        # coco annotation 불러오기 (by. coco API)
        self.coco = coco
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        if group[1] == 'train':
            self.transforms = get_train_transform(img_size)
        else:
            self.transforms = get_valid_transform(img_size)

    def __getitem__(self, index: int):
        index = self.mask[index]
        image_id = self.coco.getImgIds(imgIds=index)

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # boxes (x, y, w, h)
        boxes = np.array([x['bbox'] for x in anns])

        # boxex (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # box별 label
        labels = np.array([x['category_id']+1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas,
                  'iscrowd': is_crowds}

        # transform
        if self.transforms:
            while True:
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    target['labels'] = torch.tensor(sample['labels'])
                    break
            
        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.mask)
        # return len(self.coco.getImgIds())

# TestDataset class 선언

class TestDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, coco, data_dir, group, img_size=512, mode='valid'):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = coco
        self.mode = mode
        self.mask = group[0]
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = get_test_transform(img_size)

    def __getitem__(self, index: int):
        if self.mode=='valid':
            index = self.mask[index]
        image_id = self.coco.getImgIds(imgIds=index)

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # 라벨 등 이미지 외 다른 정보 없기 때문에 train dataset과 달리 이미지만 전처리
        
        # transform
        if self.transforms:
            sample = self.transforms(image=image)

        return sample['image'], image_id
    
    def __len__(self) -> int:
        if self.mode=='valid':
            return len(self.mask)
        else:
            return len(self.coco.getImgIds())
