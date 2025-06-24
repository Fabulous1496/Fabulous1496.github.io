---
title: Pytorch基本语法
date: 2024-02-09 16:35:55
categories:
- Deep Learning
index_img: /Pictures/DL/01/f.jpg
banner_img: /Pictures/DL/01/f.jpg
---

# Pytorch的基本操作


## 张量的创建

Tensor张量：张量的概念类似于Numpy中的ndarray数据结构，最大的区别在于Tensor可以利用GPU的加速功能。

在使用Pytorch时，需要先引入模块：

```python
from __future__ import print_function
import torch
```

下面是Pytorch中创建张量的基本语法：

1. torch.tensor 根据指定数据创建张量
2. torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
3. torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

```python
from __future__ import print_function
import torch
import numpy as np
import random

# 1. 根据已有数据创建张量
def test01():
    # 1. 创建张量标量
    data = torch.tensor(10)
    print(data)
    # 2. numpy 数组, 由于 data 为 float64, 下面代码也使用该类型
    data = np.random.randn(2, 3)
    data = torch.tensor(data)
    print(data)
    # 3. 列表, 下面代码使用默认元素类型 float32
    data = [[10., 20., 30.], [40., 50., 60.]]
    data = torch.tensor(data)
    print(data)

# 2. 创建指定形状的张量
def test02():
    # 1. 创建2行3列的张量, 默认 dtype 为 float32
    data = torch.Tensor(2, 3)
    print(data)
    # 2. 注意: 如果传递列表, 则创建包含指定元素的张量
    data = torch.Tensor([10])
    print(data)
    data = torch.Tensor([10, 20])
    print(data)

# 3. 使用具体类型的张量
def test03():
    # 1. 创建2行3列, dtype 为 int32 的张量
    data = torch.IntTensor(2, 3)
    print(data)
    # 2. 注意: 如果传递的元素类型不正确, 则会进行类型转换
    data = torch.IntTensor([2.5, 3.3])
    print(data)
    # 3. 其他的类型
    data = torch.ShortTensor()  # int16
    data = torch.LongTensor()   # int64
    data = torch.FloatTensor()  # float32
    data = torch.DoubleTensor() # float64
```

创建线性和随机张量：
1. torch.arange 和 torch.linspace 创建线性张量
2. torch.random.init_seed 和 torch.random.manual_seed 随机种子设置
3. torch.randn 创建随机张量

- rand和randn：
1. torch.rand(): 这个函数生成一个张量，其中的值是在 $[0, 1)$ 区间内均匀分布的随机数。你可以通过指定张量的形状来生成不同形状的张量。
2. torch.randn(): 这个函数生成一个张量，其中的值是从标准正态分布（均值为0，标准差为1）中随机采样得到的。你同样可以通过指定张量的形状来生成不同形状的张量。
Pytorch的基本运算操作

```python
import torch
# 1. 创建线性空间的张量
def test01():
    # 1. 在指定区间按照步长生成元素 [start, end, step)
    data = torch.arange(0, 10, 2)
    print(data)
    # 2. 在指定区间按照元素个数生成
    data = torch.linspace(0, 11, 10)
    print(data)

# 2. 创建随机张量
def test02():
    # 1. 创建随机张量
    data = torch.randn(2, 3)  # 创建2行3列张量
    print(data)
    # 2. 随机数种子设置
    print('随机数种子:', torch.random.initial_seed())
    torch.random.manual_seed(100)
    print('随机数种子:', torch.random.initial_seed())
```

创建01张量
1. torch.ones 和 torch.ones_like 创建全1张量
2. torch.zeros 和 torch.zeros_like 创建全0张量
3. torch.full 和 torch.full_like 创建全为指定值张量

```python
import torch

def test01():
    # 1. 创建指定形状全0张量
    data = torch.zeros(2, 3)
    print(data)
    # 2. 根据张量形状创建全0张量
    data = torch.zeros_like(data)
    print(data)

# 2. 创建全1张量
def test02():
    # 1. 创建指定形状全0张量
    data = torch.ones(2, 3)
    print(data)
    # 2. 根据张量形状创建全0张量
    data = torch.ones_like(data)
    print(data)

# 3. 创建全为指定值的张量
def test03():
    # 1. 创建指定形状指定值的张量
    data = torch.full([2, 3], 10)
    print(data)
    # 2. 根据张量形状创建指定值的张量
    data = torch.full_like(data, 20)
    print(data)
```

张量元素类型转换
1. tensor.type(torch.DoubleTensor)
2. torch.double()
```python
import torch

def test():
    data = torch.full([2, 3], 10)
    # 将 data 元素类型转换为 float64 类型
    # 1. 第一种方法
    data = data.type(torch.DoubleTensor)
    # 转换为其他类型
    # data = data.type(torch.ShortTensor)
    # data = data.type(torch.IntTensor)
    # data = data.type(torch.LongTensor)
    # data = data.type(torch.FloatTensor)

    # 2. 第二种方法
    data = data.double()
    # 转换为其他类型
    # data = data.short()
    # data = data.int()
    # data = data.long()
    # data = data.float()
```

## 张量的基本运算

基本运算中，包括 add、sub、mul、div、neg 等函数，以及这些函数的 in-place 版本。

>注意：所有的in-place的操作函数都有一个下划线的后缀，如x.copy_(y)，都会直接改变x的值

阿达玛积指的是矩阵对应位置的元素相乘，用 * 表示。
点积运算指一般意义上的矩阵乘法，要求矩阵可乘，用运算符 @ 表示。
torch.matmul 对进行点乘运算的两矩阵形状没有限定，利用广播机制进行。

```python
import numpy as np
import torch

def test01():
    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])
    # 第一种方式
    data = torch.mul(data1, data2)

    # 第二种方式
    data = data1 * data2

def test02():
    data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    # 第一种方式
    data = data1 @ data2

    # 第二种方式
    data = torch.mm(data1, data2)

    # 第三种方式
    data = torch.matmul(data1, data2)
```

对于PyTorch的四则运算会优先进行元素级别的操作，即两个张量之间对应位置元素进行运算。这就要求张量满足同型或满足广播规则。

和一个常数进行运算时，会将张量中的每一个元素都进行运算，可以看作将常数广播为同型张量。

- **广播规则：**

广播规则是指在进行张量运算时，PyTorch会自动调整张量的形状，使得它们能够进行元素级别的操作。

具体来说，当两个张量的形状不完全匹配时，PyTorch会根据一组规则对它们进行扩展，使它们的形状能够对齐，从而进行运算。

广播规则的基本思想是，如果两个张量的形状在某个维度上相同，或者其中一个张量在某个维度上的长度为1，那么可以在该维度上进行广播。广播操作会在这些维度上复制张量，使其形状与另一个张量相匹配，从而进行运算。

具体来说，广播规则包括以下几点：

1. 维度数增加：如果两个张量的维度数不同，会在较小的张量的前面添加一个或多个维度，直到两个张量的维度数相同。

2. 维度长度为1的扩展：对于每个维度，如果两个张量在该维度上的长度不同，且其中一个张量在该维度上的长度为1，可以在该维度上对该张量进行扩展，使其长度与另一个张量相同。

3. 扩展之后形状相同：经过广播之后，两个张量的形状必须是相同的才能进行元素级别的操作。

```python
import torch
x = torch.tensor([1],[2],[3])
y = torch.tensor([[4,5,6]])
# 执行广播操作
broadcasted_x, broadcasted_y = torch.broadcast_tensors(x, y)
print("Broadcasted x:", broadcasted_x)
print("Broadcasted y:", broadcasted_y)
'''
Broadcasted x: 
tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])

Broadcasted y: 
tensor([[4, 5, 6],
        [4, 5, 6],
        [4, 5, 6]])
'''
```

## 指定运算设备
将张量移动到 GPU 上有两种方法: 
1. 使用 cuda 方法 
2. 直接在 GPU 上创建张量 
3. 使用 to 方法指定设备
```python
import torch

# 1. 使用 cuda 方法
def test01():
    data = torch.tensor([10, 20 ,30])
    data = data.cuda()

# 2. 直接将张量创建在 GPU 上
def test02():
    data = torch.tensor([10, 20, 30], device='cuda:0')
    # 使用 cpu 函数将张量移动到 cpu 上
    data = data.cpu()

# 3. 使用 to 方法
def test03():
    data = torch.tensor([10, 20, 30])
    data = data.to('cuda:0')
```

## 张量类型转换
1. 使用 Tensor.numpy 函数可以将张量转换为 ndarray 数组，但是共享内存，可以使用 copy 函数避免共享。
2. 使用 from_numpy 可以将 ndarray 数组转换为 Tensor，默认共享内存，使用 copy 函数避免共享。
3. 使用 torch.tensor 可以将 ndarray 数组转换为 Tensor，默认不共享内存。
4. 对于只有一个元素的张量，使用 item 方法将该值从张量中提取出来。

## 张量拼接操作

1. torch.cat 函数可以将两个张量根据指定的维度拼接起来。
```python
import torch

# 生成形状为3，5，4的三维张量
# 第0维（也就是维度索引为0的维度）是大小为3，通常表示这个张量中包含3个样本或者批次。
# 第1维（维度索引为1的维度）是大小为5，代表每个样本中的特征数量。
# 第2维（维度索引为2的维度）是大小为4，代表每个特征的维度或者特征向量的长度。
data1 = torch.randint(0, 10, [3, 5, 4])
data2 = torch.randint(0, 10, [3, 5, 4])
# 按第0维拼接
new_data = torch.cat([data1, data2], dim=0)
```

2. torch.stack 函数可以将两个张量根据指定的维度叠加起来，会增加新的维度。

两个[2, 2] 的张量堆叠，产生大小为 [2, 2, 2] 的张量。

## 张量索引操作
1. 简单行、列索引
```python
import torch

data = torch.randint(0, 10, [4, 5])
data[0]    # 第一行
data[:, 0]    # 第一列
data[0, 1]    # 返回(0, 1) 位置即第一行第二列的元素，返回一个张量
data[[[0], [1]], [1, 2]]    # 返回 0、1 行的 1、2 列共4个元素
```

2. 范围索引
```python
# 前3行的前2列数据
data[:3, :2]
# 第2行到最后的前2列数据
data[2:, :2]
```

3. 布尔索引
```python
# 第三列大于5的行数据，输出整行
data[data[:, 2] > 5]
# 第二行大于5的列数据
data[:, data[1] > 5]
```

## 张量形状操作
reshape 函数可以在保证张量数据不变的前提下改变数据的维度，将其转换成指定的形状，前提是数据个数不变。

ranspose 函数可以实现交换张量形状的指定维度, 例如: 一个张量的形状为 (2, 3, 4) 可以通过 transpose 函数把 3 和 4 进行交换, 将张量的形状变为 (2, 4, 3)

squeeze 函数用删除 shape 为 1 的维度，unsqueeze 在每个维度添加 1, 以增加数据的形状。

```python
data = torch.tensor(np.random.randint(0, 10, [1, 3, 1, 5]))
print('data shape:', data.size())

# 1. 去掉值为1的维度
new_data = data.squeeze()
print('new_data shape:', new_data.size())  # torch.Size([3, 5])

# 2. 去掉指定位置为1的维度，注意: 如果指定位置不是1则不删除
new_data = data.squeeze(2)
print('new_data shape:', new_data.size())  # torch.Size([3, 5])

# 3. 在2维度增加一个维度
new_data = data.unsqueeze(-1)
print('new_data shape:', new_data.size())  # torch.Size([3, 1, 5, 1])
```

输出：

```python
data shape: torch.Size([1, 3, 1, 5])
new_data shape: torch.Size([3, 5])
new_data shape: torch.Size([1, 3, 5])
new_data shape: torch.Size([1, 3, 1, 5, 1])
```

