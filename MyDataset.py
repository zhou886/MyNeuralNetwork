import torch
from torch.utils.data import Dataset
import cv2 as cv
import os
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root_path, train, transform):
        super().__init__()
        self.root_path = root_path                      # 保存根目录
        self.label_path = os.listdir(self.root_path)    # 获得所有标签
        self.imgAndLabel = []                           # 用于存放图像地址和对应标签
        self.transform = transform                      # 保存对图像进行的变换

        if train:
            # 如果用作训练集
            for label in self.label_path:
                path = os.path.join(self.root_path, label)
                img_path = os.listdir(path)

                # 将数据集的前80%放在imgAndLabel中用作训练集
                total_img_size = round(len(img_path)*0.8)
                for i in range(total_img_size):
                    self.imgAndLabel.append(
                        (os.path.join(self.root_path,
                         label, img_path[i]), label)
                    )
        else:
            # 如果用作测试集
            for label in self.label_path:
                path = os.path.join(self.root_path, label)
                img_path = os.listdir(path)

                # 将数据集的后20%放在imgAndLabel中用作测试集
                total_img_size = round(len(img_path)*0.8)
                for i in range(total_img_size, len(img_path)):
                    self.imgAndLabel.append(
                        (os.path.join(self.root_path,
                         label, img_path[i]), label)
                    )

    def __getitem__(self, index):
        # 获得图像路径和标签
        img_path, label = self.imgAndLabel[index]

        # 读取图像
        img = cv.imread(img_path)

        # 如果有变换就对图像应用变换
        if self.transform:
            img = self.transform(img)

        # 将标签转换为对应的tensor类型的数据方便使用
        label_tensor = 0
        for i in range(len(self.label_path)):
            if label == self.label_path[i]:
                label_tensor = torch.tensor(i)
                break

        return img, label_tensor

    def __len__(self):
        return len(self.imgAndLabel)

# 单元测试
if __name__ == "__main__":
    myDataset = MyDataset(r'.\dataset', train=False, transform=None)
    for data in myDataset:
        img, label = data
        print(img, '\n', label, '\n')
