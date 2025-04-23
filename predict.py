import torch
from trainer import HandwritingTrainer
from PIL import Image
from torchvision import transforms
import os
import numpy as np 
if __name__ == "__main__":
    
    model = HandwritingTrainer.load_from_checkpoint("logs/version_1/checkpoint-epoch=08-val_loss=0.161.ckpt")
    model.eval()
    model = model.to("cuda")
    img = Image.open("./wo.png")
    img = img.convert("RGB")
    img = img.resize((64,64))
    trans = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.to("cuda")
    with torch.no_grad():
        output = model(img)
        output = torch.nn.functional.softmax(output,dim=1)
        output = output.cpu().numpy()
        print(output)
        print(np.argmax(output))