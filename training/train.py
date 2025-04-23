from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import pytorch_lightning as pl
import random
import numpy as np
import torch

from trainer import HandwritingTrainer
from argparse import ArgumentParser

if __name__ == "__main__":

    parse =ArgumentParser()
    parse.add_argument('-m','--model', type=str,default="mobilenetv2")
    parse.add_argument('-b','--batch_size', type=int,default=32)
    args = parse.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='best-model',
        save_top_k=1,
        mode='min'
    )

    # 创建TensorBoardLogger实例并指定日志保存路径
    logger = TensorBoardLogger(save_dir="logs", name=args.model,log_graph=True)

    # 设置 ModelCheckpoint 回调  
    checkpoint_callback = ModelCheckpoint(  
        monitor='val_loss',  # 监控指标，例如验证损失  
        dirpath=logger.log_dir,  # 检查点保存路径  
        filename='checkpoint-{epoch:02d}-{val_loss:.3f}',  # 保存文件的命名格式  
        save_top_k=3,  # 只保存最好的k个模型  
        mode='min',  # 'min'表示监控指标越小越好  
        save_weights_only=True,  # 只保存权重而不是整个模型  
    ) 
    early_stopping = EarlyStopping('val_loss', mode="min")
    trainer_args = {
        'accelerator': 'gpu',
        'devices': [0],
        'callbacks': [checkpoint_callback,early_stopping],
        'max_epochs': 100,
    }
    trainer = pl.Trainer(logger=logger, **trainer_args, fast_dev_run=False)
    # 训练数据
    model = HandwritingTrainer(model=args.model,batch_size=args.batch_size)
    print(model)
    trainer.fit(model)
