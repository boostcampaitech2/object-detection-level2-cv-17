# 라이브러리 및 모듈 import
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def get_net(checkpoint_path=None, box_weight=50):
    
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = 10
    config.image_size = (1024,1024)
    
    config.soft_nms = False
    config.max_det_per_image = 25
    config.box_loss_weight = box_weight

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        
    return DetBenchTrain(net)
