from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd


class MyDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
            csv_file: 标签文件的路径.
            root_dir: 所有图片的路径.
            transform: 一系列transform操作
        """
        self.data_frame = pd.read_csv(csv_file)#默认第一行是列标签
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame) # 返回数据集长度

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,self.data_frame.iloc[idx, 0]) #获取图片所在路径
        img = Image.open(img_path).convert('RGB') # 防止有些图片是RGBA格式
        
        label_number = self.data_frame.iloc[idx, 1] # 获取图片的类别标签
        
        if self.transform:
            img = self.transform(img)

        return img, label_number # 返回图片和标签