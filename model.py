import torch
import torch.nn as nn

class X_NIRSNet(nn.Module):
    def __init__(self, channels=13, time_points=368, num_classes=2):
        super(X_NIRSNet, self).__init__()
        # 파라미터를 받아 모델 사이즈 결정
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(channels, 1)) 
        # ... (작성하신 모델 내용)
        
    def forward(self, x1, x2):
        # x1: Stream 1 (HbO), x2: Stream 2 (HbR)
        # 2-Stream Feature Extraction & Fusion 로직
        pass