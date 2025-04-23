import torch
import torch.nn as nn
import torchvision
from torchvision.models.mobilenetv2 import MobileNetV2

class MobileNetV2_Chinese(MobileNetV2):
    def __init__(self, num_classes=3740):
        super().__init__()
        # 调整第一个卷积层的步长
        self.features[0][0].stride = (1, 1)
        # 重新定义分类器
        last_channel = 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def adapt_resnet_for_small_input(model):
    # --- 修改第一层卷积 ---
    # 原始：kernel_size=7, stride=2 → 改为 kernel_size=3, stride=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # --- 移除第一个最大池化层（原 stride=2）---
    model.maxpool = nn.Identity()
    
    # --- 调整后续下采样层的 stride ---
    # Layer2 的第一个 BasicBlock
    layer2 = model.layer2[0]
    layer2.conv1.stride = (1, 1)                # 原为 (2,2)
    layer2.downsample[0].stride = (1, 1)        # 原为 (2,2)
    
    # Layer3 的第一个 BasicBlock
    layer3 = model.layer3[0]
    layer3.conv1.stride = (1, 1)                # 原为 (2,2)
    layer3.downsample[0].stride = (1, 1)        # 原为 (2,2)
    
    # Layer4 的第一个 BasicBlock
    layer4 = model.layer4[0]
    layer4.conv1.stride = (1, 1)                # 原为 (2,2)
    layer4.downsample[0].stride = (1, 1)        # 原为 (2,2)
    
    return model

class ResNet18_Chinese(nn.Module):
    def __init__(self, num_classes=3740):
        super().__init__()
        # 加载预训练 ResNet18（ImageNet 权重）
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone = adapt_resnet_for_small_input(self.backbone)
        
        # 替换输入层适配小尺寸输入（原第一层卷积kernel_size=7→3）
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除原最大池化层（避免过早下采样）
        self.backbone.maxpool = nn.Identity()
        
        # 替换分类头
        self.backbone.fc = nn.Linear(512, num_classes)  # ResNet18 最终特征维度512
    
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = MobileNetV2_Chinese()
    # save to file
    print(model, file=open("model.txt", "w"))