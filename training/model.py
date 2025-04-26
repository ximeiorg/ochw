import torch
import torch.nn as nn
import torchvision
from torchvision.models.mobilenetv2 import MobileNetV2,mobilenet_v2
from torchvision.models.resnet import ResNet18_Weights,ResNet

class MobileNetV2_Chinese(nn.Module):
    def __init__(self, num_classes=4037):  # 确保与数据集类别数一致
        super().__init__()
        # 加载预训练模型
        self.base_model = mobilenet_v2(weights = torchvision.models.MobileNet_V2_Weights)
        # 替换分类器
        last_channel = self.base_model.last_channel
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(last_channel, num_classes),
        )
        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.classifier(x)
        return x


class ResNet18_Chinese(nn.Module):
    def __init__(self, num_classes=4037):
        super().__init__()

        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # 移除前面的下采样，让最后的特征图大一些(2x2) => (4x4) 这个操作直接让训练变慢7-8倍
        # self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.model.maxpool = nn.Identity()

        # 替换分类头
        self.model.fc = nn.Linear(512, num_classes)  # ResNet18 最终特征维度512
    
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    model = ResNet18_Chinese()
    # save to file
    print(model, file=open("model_resnet18.txt", "w"))