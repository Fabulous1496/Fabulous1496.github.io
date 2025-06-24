---
title: Pytorch简易入门教程
date: 2025-06-20
categories:
- Deep Learning
index_img: /Pictures/torch_basic/banner.jpg
banner_img: /Pictures/torch_basic/banner.jpg
---

# Pytorch简易入门教程

> 声明：本教程依照Bilibili Up主我是土堆出品系列视频编撰，有兴趣的读者可以去原视频进行详细学习。

## 1. 环境配置

使用Anaconda作为包管理工具，创建初始环境、安装torch，cuda和cudnn。

推荐安装Jupyter notebook解锁新天地。

使用代码检测环境是否配置成功：

```python
import torch
print("Pytorch version：")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is :")
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
```

笔者配置的环境如下：

```tex
Pytorch version：
2.5.1
CUDA Version: 
12.4
cuDNN version is :
90100
True
```

在学习过程中，我们遇到问题往往会有多种信息来源进行参考，那么这时我们就会面临一个信息的筛选。建议在学习时依照官方文档进行对照，常用的函数为`dir()`以及`help()`，后者在例如Pycharm、Visuial Studio Code等编辑器中可以用快捷键`Ctrl+左键`进行替代。



## 2. PyTorch数据加载初认识

在PyTorch中，读取数据主要涉及两种方式：**Dataset**以及**Dataloader**。

对于Dataset，它是对于数据集的抽象，定义了数据的类型、来源和格式，并且进行编号。通常Dataset只处理单条数据。

而对于Dataloader，它是一个数据加载器，用于控制数据的加载方式，主要负责将Dataset中的数据批量（batch）加载出来，同时提供了例如`shuffle`、`batch_size`等基本功能，返回一个可迭代对象，每次迭代返回一个`batch`。

这里我们以一个蚂蚁蜜蜂图片数据集为例（[下载链接](https://download.pytorch.org/tutorial/hymenoptera_data.zip)）进行具体的学习。

数据集的基本结构如下：

```Tex
hymenoptera_data/
|--train/
	|--ants/
	└--bees/
└--val/
	|--ants/
	└--bees/
```

其中，`ants`和`bees`目录下存放对应的图片。

下面我们来具体构造一个Dataset。构造一个Dataset一般只需要重写torch提供的类的三个函数：`__init__`，`__getitem__`和`__len__`，分别执行初始化、获得数据与标签、获取数据集长度的功能。

对于我们本案例中的数据集，将相同标签的所有数据条目放在同一目录下，因此文件夹名`ants`和`bees`为我们的二分类标签。在另一种数据集的组织方式中，每一条数据会有对应的同名txt文件，里面存放相关的标签信息。

```python
# 导入必要的库
from torch.utils.data import Dataset
from PIL import Image
import os
```

```python
# 创建Dataset类
class MyDataset(Dataset):
    # 初始化方法，接收根目录和标签目录
    # root_dir: 数据集根目录
    # label_dir: 标签目录（子目录名）
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_list = os.listdir(self.path)

    # getitem需要获得索引对应的图像和标签
    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.path, image_name)
        image = Image.open(image_path)
        label = self.label_dir
        return image, label
    
    def __len__(self):
        return len(self.image_list)
```

```python
# 创建一个测试用实例
root_dir = 'hymenoptera_data/train'
label_dir = 'ants'
ants_dataset = MyDataset(root_dir, label_dir)
image, label = ants_dataset[1]
print(f"Image: {image}, Label: {label}")
image.show()  

root_dir = 'hymenoptera_data/train'
label_dir = 'bees'
bees_dataset = MyDataset(root_dir, label_dir)
image, label = bees_dataset[5]
print(f"Image: {image}, Label: {label}")
image.show()

train_dataset = ants_dataset + bees_dataset
```

这样，我们就完成了一个基本数据集的加载。总而言之，加载数据集最为核心的部分便是分辨不同数据的格式（图片、语音、文本等），利用不同的工具进行加载，并同时加载对应的`label`信息。





