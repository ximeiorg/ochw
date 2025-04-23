import pytorch_lightning as pl
import torch
from dataset import HWDB1Dataset
from model import MobileNetV2_Chinese,ResNet18_Chinese
from torchvision import transforms
from torch.utils.data import DataLoader


class HandwritingTrainer(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = MobileNetV2_Chinese(4037)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = 128
        self.root_dir = "data"

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),lr=1e-2,weight_decay=1e-4,momentum=0.9)
        # optimizer = torch.optim.AdamW(
        #     self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100)
        return [optimizer], [scheduler]

    def common_transforms_compose(self,mode='train'):
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色扰动
                transforms.RandomGrayscale(p=0.1),  # 随机灰度化模拟不同墨色
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

    def train_dataloader(self):
        train_dataset = HWDB1Dataset(
            root_dir=self.root_dir, mode='train', transform=self.common_transforms_compose('train'))
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True

        )

    def val_dataloader(self):
        train_dataset = HWDB1Dataset(
            root_dir=self.root_dir, mode='test', transform=self.common_transforms_compose('test'))
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True

        )
    
    def compute_grad_norm(self):
        # 计算所有参数的梯度范数
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2 范数
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def on_after_backward(self):
        # 计算梯度范数
        grad_norm = self.compute_grad_norm()
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    
    def common_setps(self, batch, batch_idx,mode = "train"):
        x,y = batch
        pred = self.model(x)
        # print(f"Model output: {pred}")  # 打印模型输出

        loss = self.criterion(pred, y)

        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.common_setps(batch, batch_idx,mode = "train")
    
    def validation_step(self, batch, batch_idx):
        return self.common_setps(batch, batch_idx,mode = "val")

