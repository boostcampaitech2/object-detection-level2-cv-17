import warnings
warnings.filterwarnings(action='ignore')

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils import data as data_
from pycocotools.coco import COCO

from dataset import TestCustom
from model import FasterRCNNVGG16

data_dir = '../../dataset'   # 데이터 경로 
inf_load_path = './checkpoints/faster_rcnn_scratch_checkpoints.pth' # inference시 체크포인트 경로

def eval(dataloader, faster_rcnn):
    outputs = []
    for ii, (imgs, sizes) in enumerate(tqdm(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        for out in range(len(pred_bboxes_)):
            outputs.append({'boxes':pred_bboxes_[out], 'scores': pred_scores_[out], 'labels': pred_labels_[out]})
    
    return outputs



# Test dataset 불러오기
#     testset = TestDataset()
annotation = os.path.join(data_dir,'train.json')
testset = TestCustom(annotation, data_dir)
test_dataloader = data_.DataLoader(testset,
                                    batch_size=1, # only batch_size=1 support
                                    num_workers=4,
                                    shuffle=False, 
                                    pin_memory=False
                                    )
# faster rcnn 불러오기
faster_rcnn = FasterRCNNVGG16().cuda()
state_dict = torch.load(inf_load_path)
if 'model' in state_dict:
    faster_rcnn.load_state_dict(state_dict['model'])
print('load pretrained model from %s' % inf_load_path)

# evaluation
outputs = eval(test_dataloader, faster_rcnn)
score_threshold = 0.05
prediction_strings = []
file_names = []

# submission file 작성
coco = COCO(os.path.join(data_dir, 'train.json'))
for i, output in enumerate(outputs):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
        if score > score_threshold:
            prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[1]) + ' ' + str(
                box[0]) + ' ' + str(box[3]) + ' ' + str(box[2]) + ' '
    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv("./faster_rcnn_scratch_submission.csv", index=False)

print(submission.head())        