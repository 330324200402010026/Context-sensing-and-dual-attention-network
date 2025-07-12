import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from torchvision import datasets

# 指定数据集路径
data_dir = r'/tmp/pycharm_project_899/UCMerced_LandUse/Images'

# 尝试加载数据集
try:
    dataset = datasets.ImageFolder(root=data_dir)
    print(f"成功加载 {len(dataset)} 个图像样本")
except FileNotFoundError as e:
    print(f"加载数据集时出错: {e}")
except Exception as e:
    print(f"发生错误: {e}")
