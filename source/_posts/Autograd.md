---
title: Autograd
date: 2024-02-11 14:31:05
categories:
- Deep Learning
index_img: /Pictures/DL/02/cover.jpg
banner_img: /Pictures/DL/02/cover.jpg
---

# PyTorch中的autograd

在整个PyTorch框架中，所有的神经网络都是一个autograd package（自动求导工具包）

autograd package提供了一个对Tensor上的所有操作进行自动微分的功能

## 关于torch.Tensor

- torch.Tensor是整个package中的核心类，如果将属性 *.requires_grad* 设置为 true，他将追踪这个类上的所有操作，当代码要进行反向传播时，直接调用 *.backward()* 就可以自动计算所有的梯度，在这个Tensor上的所有梯度将被累加进属性 *.grad* 中。

- 如果想终止一个Tensor在计算图中的追踪回溯，只需要执行 *.detach()* 就可以将该Tensor从计算图中撤下，在未来的回溯计算中也不会在计算该Tensor。

- 除了 *.detach()* 如果想终止对计算图的回溯，也就是不再进行方向传播求导的过程，也可以采用代码块的形式 *with torch.no_grad():* ，这种操作非常适用于对模型进行预测的时候，因为预测阶段不需要对梯度进行计算。 

## 关于torch.Function

- Function类是和Tensor类同等重要的一个核心类,它和Tensor共同构成一个完整的类

- 每一个Tensor拥有一个 *grad.fn* 属性,代表引用了哪个具体的Function创建了该Tensor

```python
from __future__ import print_function
import torch

x1 = torch.ones(3, 3)
x2 = torch.ones(2, 2, requires_grad=True)
print(x1, '\n', x2)

'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]]) 
 tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
'''

y = x2 + 2
print(y)
print(x1.grad_fn, '\n', y.grad_fn)

'''
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
None 
 <AddBackward0 object at 0x0000020CC93A7F70>
'''
```

>用户自定义的Tensor,其grad_fn=None.

一些更复杂的操作:

```python
z = y * y * 3
out = z.mean()
print(z, out)

'''
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
'''
```

> *.mean()* 方法表示求均值

使用inplace操作符可以改变Tensor的 *requires_grad* 属性:

```python
a = torch.ones(2, 2)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)

'''
False
True
'''
```

## 关于梯度Gradients

在PyTorch中,反向传播是依靠 *.backward()* 实现的,下面是一个使用例子:

```python
out.backward()
print(x2.grad)

'''
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
'''
```

这里我们首先要弄清 $out$ 是如何得到的.

我们把 $out$ 视为一个多元函数,将 $x$ 视为一个多元变量,那么:
$$out = \frac{3(x+2)^{2}}{4}$$

对 $out$ 求 $x$ 的导数:
$$out' = \frac{3(x+2)}{2}$$

将x代入,便得到结果.

- 关于自动求导的属性设置:可以通过设置 *.requires_grad=True* 来执行自动求导,也可以通过代码块的限制来停止自动求导.

```python
print(x.requires_grad)
print((x ** 2).equires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

'''
True
True
False
'''
```

- 可以通过 *.detach()* 获得一个新的Tensor,拥有相同内容但不需要自动求导.

```python
print(x.requires_grad)
y = x.detach()
print(y,requires_grad)
print(x.eq(y).all)

'''
True
False
tensor(True)
'''
```