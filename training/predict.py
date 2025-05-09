import torch
from trainer import HandwritingTrainer
from PIL import Image
from torchvision import transforms
import os
import numpy as np


def get_labels():
    labels = []
    with open("./data/train/label.txt", "r", encoding="utf-8") as f:
        for line in f:
            # line: !	0
            line = line.strip()
            label = line.split("\t")[0]
            labels.append(label)
    return labels


if __name__ == "__main__":

    model = HandwritingTrainer.load_from_checkpoint(
        "./logs/mobilenetv2/version_0/checkpoint-epoch=11-val_loss=0.191.ckpt", model="mobilenetv2")
    model.eval()
    model = model.to("cuda")
    img = Image.open("../testdata/yi.jpeg")
    img = img.convert("RGB")
    img = img.resize((96, 96))
    trans = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.95], std=[0.2])
    ])
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.to("cuda")
    labels = get_labels()
    with torch.no_grad():
        output = model(img)
        output = torch.nn.functional.softmax(output, dim=1)
        # 获取top5的预测结果
        top5_prob, top5_idx = torch.topk(output, 5)
        top5_prob = top5_prob.cpu().numpy()
        top5_idx = top5_idx.cpu().numpy()
        for i in range(5):
            idx = top5_idx[0][i]
            print(f"Top {i+1} 预测标签: {labels[idx]}, 预测概率: {top5_prob[0][i]}")
