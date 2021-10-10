# 라이브러리 및 모듈 import
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

def get_net(checkpoint_path=None, box_weight=50, img_size=512):
    
    config = get_efficientdet_config('tf_efficientdet_d4')
    config.num_classes = 10
    config.image_size = (img_size, img_size)
    
    config.soft_nms = False
    config.max_det_per_image = 100
    config.box_loss_weight = box_weight
    config.label_smoothing = 0.2
    # config.lagacy_focal = True
    # config.jit_loss = True
    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    net = DetBenchTrain(net)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)
        
    return net

# Effdet config를 통해 모델 불러오기 + ckpt load
def load_net(checkpoint_path, device, img_size):
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.num_classes = 10
    config.image_size = (img_size, img_size)
    
    config.soft_nms = True
    config.max_det_per_image = 100
    config.label_smoothing = 0.2
    
    
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    net = DetBenchPredict(net)
    net.load_state_dict(checkpoint)
    net.eval()

    return net.to(device)