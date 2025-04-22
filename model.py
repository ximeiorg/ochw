import torch
import torch.nn as nn

class HandwritingCNN(nn.Module):
    def __init__(self, num_classes=3740):
        super().__init__()

        self.features = nn.Sequential(
            # 输入: 3x64x64 (RGB)
            nn.Conv2d(3, 64, 3, padding=1),  # 关键修改：输入通道改为3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x32x32
            
            # 残差块增强特征提取
            ResidualBlock(64, 128, stride=2),  # 128x16x16
            ResidualBlock(128, 256, stride=2), # 256x8x8
            
            # 注意力机制
            SEBlock(256)  # 增强重要通道
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化替代Flatten
            nn.Flatten(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, in_channels//4, 3, padding=1)
        self.branch2 = nn.Conv2d(in_channels, in_channels//4, 3, padding=2, dilation=2)  # 扩大感受野
        self.branch3 = nn.Conv2d(in_channels, in_channels//4, 3, padding=4, dilation=4)
        self.conv1x1 = nn.Conv2d(3*(in_channels//4), in_channels, 1)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return self.conv1x1(torch.cat([b1, b2, b3], dim=1))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y