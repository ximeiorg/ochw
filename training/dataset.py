from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

class HWDB1Dataset(Dataset):
    def __init__(self, root_dir:str, mode:str = 'train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.data = self.load_data()
        self.char_to_idx = self.build_label_to_idx()
        # label
        self.label_idx = self.load_label_to_idx(Path(self.root_dir) / self.mode / 'label.txt')
        
    
    def load_label_to_idx(self,label_path:Path):
        char_to_idx = {}
        with open(str(label_path), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                char = line[0]
                char_to_idx[char] = int(line[1])
        return char_to_idx

    def load_data(self):
        data = []
        txt_file = Path(self.root_dir) / self.mode / 'gt.txt'
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                img_path = line[0]
                label = line[1]
                data.append((img_path, label))
        return data
    
    def build_label_to_idx(self):
        char_to_idx = {}
        for _, label in self.data:
            char_to_idx[label] = False
        return char_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, char = self.data[index]
        img_path = Path(self.root_dir) / self.mode / 'images' / img_path
        img = Image.open(img_path)
        img = img.convert('RGB')

        if self.transform: img = self.transform(img)
        label_indices = self.label_idx[char]
        label_tensor = torch.tensor(int(label_indices), dtype=torch.long)
      
        # print(f"Image shape: {img.shape}, Label: {label_tensor}")

        return img, label_tensor
    

if __name__ == '__main__':
    # 查看图片的是rgb还是grayscale
    dataset = HWDB1Dataset(root_dir='./data', mode='train')
    print(dataset.char_to_idx)
    for img, label in dataset:
        print(label)
        break