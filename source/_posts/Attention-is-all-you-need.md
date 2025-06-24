---
title: Attention is all you need
date: 2024-03-06 16:00:12
categories:
- Deep Learning
index_img: /Pictures/DL/Attention/cover.jpg
banner_img: /Pictures/DL/Attention/cover.jpg
---

# 注意力机制

我们观察事物时，之所以能够快速判断一种事物(当然允许判断是错误的), 是因为我们大脑能够很快把注意力放在事物最具有辨识度的部分从而作出判断，而并非是从头到尾的观察一遍事物后，才能有判断结果. 正是基于这样的理论，就产生了注意力机制.

>摘自论文原文：
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

注意力机制的核心公式为：

$$ Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt d_{k}})V $$

对于注意力机制来说，我们需要三个基本的输入： $Q(query), K(key), V(value)$ 。

在Transformer中Encoder使用的$ Q, K, V $ ，其实都是从输入矩阵 $X$ 经过线性变化得来的。

简单来说就是：

$$\begin{cases}
Q = XW^{Q} \\\\
K = XW^{K} \\\\
V = XW^{V}
\end{cases}$$

![img](/Pictures/DL/Attention/f2.png)

在这张图中，$Q$ 与 $K^{T}$ 经过MatMul，生成了相似度矩阵。对相似度矩阵每个元素除以 $\sqrt d_{k}$，其为 $K$ 的维度大小。这个除法被称为Scale。

注意力的计算过程：

![img](/Pictures/DL/Attention/f3.png)

1. query 和 key 进行相似度计算，得到一个query 和 key 相关性的分值
2. 将这个分值进行归一化(softmax)，得到一个注意力的分布
3. 使用注意力分布和 value 进行计算，得到一个融合注意力的更好的 value 值

为了增强拟合性能，Transformer对Attention继续扩展，提出了多头注意力（Multi-Head Attention）。

![img](/Pictures/DL/Attention/f4.png)

对于同样的输入 $X$ ，我们定义多组不同的 $W^{Q}, W^{K}, W^{V}$ ，计算得到多组 $Q,K,V$ ，然后学习到不同的数据。

比如我们定义8组参数，同样的输入 $X$ ，将得到8个不同的输出 $Z_{0} ~ Z_{7}$ ，在输出到下一层前，我们需要将8个输出拼接到一起，进行一次线性变换，将维度降低到我们想要的维度。

# Self-Attention

Self-attention就本质上是一种特殊的attention。

attention和self attention 其具体计算过程是一样的，只是计算对象发生了变化而已。

attention是source对target的attention，

而self attention 是source 对source的attention。

即输入的Q=K=V。

在翻译任务中，如果源句子≠目标句子，那么你用目标句子中的词去query源句子中的所有词的key，再做相应运算，这种方式就是Attention；如果你的需求不是翻译，而是对当前这句话中某几个词之间的关系更感兴趣，期望对他们进行计算，这种方式就是Self-Attention。

从范围上来讲，注意力机制是包含自注意力机制的。注意力机制给定K、Q、V，其中Q和V可以是任意的，而K往往等于V（不相等也可以）；而自注意力机制要求K=Q=V。

# Transformer
Transformer模型基于Encoder-Decoder架构。一般地，在Encoder-Decoder中，Encoder部分将一部分信息抽取出来，生成中间编码信息，发送到Decoder中。

![img](/Pictures/DL/Attention/f1.png)

我们可以将整个架构抽象为四个组成部分：

1. 输入部分
2. 输出部分
3. 编码器部分
4. 解码器部分

## 输入部分

![img](/Pictures/DL/Attention/f5.png)

- 源文本嵌入层及其位置编码器
- 目标文本嵌入层及其位置编码器

关于位置编码器 Positional Encoding：

Transformer模型的输入为一系列词，词需要转化为词向量。一般的语言模型都需要使用Embedding层，用以将词转化为词向量。Transformer没有采用RNN的结构，不能利用单词的顺序信息，但顺序信息对于NLP任务来说非常重要。在此基础上，Transformer增加了位置编码（Positional Encoding）。

$$ PE_{(pos,2i)} = sin(pos / 10000^{2i / d})$$
$$ PE_{(pos,2i+1)} = cos(pos / 10000^{2i / d})$$

$pos$ 代表单词在句子中的位置， $d$ 表示词向量的维度， $2i$ 表示偶数维度， $2i+1$ 表示奇数维度。生成的是[−1,1]区间内的实数。

### Embedding层和Positional Encoding层的代码实现：

x的大小为 (batch_size, sequence_length, embedding_dim)
pe的大小为 (max_len, embedding_dim)
这里的 sequence_length $\ne$ max_len，需要匹配形状。

```python
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        d_model: 指词嵌入的维度 
        vocab: 指词表的大小
        """
        super(Embeddings, self).__init__()
        # 调用nn中的预定义层Embedding, 获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 乘以缩放因子，通常为词嵌入的维度开根号
        return self.lut(x) * math.sqrt(self.d_model)
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        d_model: 词嵌入维度, 
        dropout: 置0比率, max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 实例化nn中预定义的Dropout层, 并将dropout传入其中, 获得对象self.dropout
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵, 它是一个0阵，矩阵的大小是max_len x d_model.
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵，用它的索引去表示词汇的绝对位置。
        # 首先使用arange方法获得一个连续自然数向量，然后使用unsqueeze方法拓展向量维度使其成为矩阵 
        position = torch.arange(0, max_len).unsqueeze(1)

        # 对应公式，将奇数维度和偶数维度分别对应初始化。
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 使用unsqueeze拓展维度。
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 适配张量大小
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
```

## 编码器部分
- 由N个编码器层堆叠而成
- 每个编码器层由两个子层连接结构组成
- 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
- 第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接

Add & Norm层由 Add 和 Norm 两部分组成。Add 类似 ResNet 提出的残差连接，以解决深层网络训练不稳定的问题。Norm 为归一化层，即 *Layer Normalization* ，通常用于 RNN 结构。

Feed Forward层由两个全连接层构成，第一层的激活函数为 ReLu，第二层不使用激活函数。

Multi-Head Attention 采用了 Mask 操作，即掩码张量,因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第i+1 个单词。

![Mask](/Pictures/DL/Attention/f7.png)

0到4即代表按顺序的前5个单词。

### Mask
```python
def subsequent_mask(size):
    """
    size是掩码张量最后两个维度的大小,形成一个方阵
    """
    attn_shape = (1, size, size)

    # 然后使用np.ones方法向这个形状中添加1元素,形成上三角阵 
    # 再使其中的数据类型变为无符号8位整形unit8 
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor, 内部做一个1 - 的操作, 即将上三角转为下三角。
    return torch.from_numpy(1 - subsequent_mask)
```

### 注意力的计算实现
```python
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    """
    输入分别是query, key, value
    mask: 掩码张量, 
    dropout：置零
    """
    # 取query的最后一维的大小, 等同于词嵌入维度
    d_k = query.size(-1)
    # 按照注意力公式得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法, 将掩码张量和scores张量每个位置一一比较, 如果掩码张量处为0则对应的scores张量用-1e9这个值来替换
        scores = scores.masked_fill(mask == 0, -1e9)

    # 进行softmax操作
    p_attn = F.softmax(scores, dim = -1)

    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn
```

### 多头注意力机制实现

```python
import copy

# 首先需要定义克隆函数, 因为在多头注意力机制的实现中, 用到多个结构相同的线性层.
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中。
def clones(module, N):
    """
    用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
    # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
    # 然后将其放在nn.ModuleList类型的列表中存放.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        head——头数
        embedding_dim——词嵌入的维度， 
        dropout——置0比率，默认是0.1
        """
        super(MultiHeadedAttention, self).__init__()

        # 判断h是否能被d_model整除
        # 这是因为我们之后要给每个头分配等量的词特征.也就是embedding_dim/head个.
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head
        self.head = head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用clones函数克隆四个，
        # 在多头注意力中，Q，K，V各需要一个，拼接的矩阵还需要一个，一共是四个.
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None.
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(0)

        batch_size = query.size(0)


        # [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for ...]: 在每次迭代中，首先使用 model(x) 对输入进行线性变换。然后使用 view 方法将结果重塑为 (batch_size, -1, self.head, self.d_k) 的形状，其中 batch_size 表示批处理大小，-1 表示自动计算该维度大小，self.head 表示头的数量，self.d_k 表示每个头的维度。这样就将线性变换后的结果按照头的数量进行了分割。

        # transpose(1, 2): 最后，使用 transpose 方法将第1和第2维进行转置。在多头注意力中，这样做是为了使代表句子长度和词向量维度的维度能够相邻，以便后续的注意力计算可以正确处理输入数据。具体地，该操作将形状从 (batch_size, seq_length, head, d_k) 转换为 (batch_size, head, seq_length, d_k)。

        query, key, value = \
        [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                     for model, x in zip(self.linears, (query, key, value))]


        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 先将维度复原，再由多头转为单头。
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        return self.linears[-1](x)
```

{% note info %}
contiguous() 是 PyTorch 中的一个方法，用于返回一个具有连续内存的新张量，即将张量的存储重新排列为连续的内存块，使得张量的元素在内存中的布局是连续的。
在上面，由于转置操作，储存内存变得不连续了，所以需要重新规划。
{% endnote %}

### 前馈全连接层

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model——线性层的输入维度，也是第二个线性层的输出维度
        d_ff——第二个线性层的输入维度和第一个线性层的输出维度
        dropout=0.1
        """
        super(PositionwiseFeedForward, self).__init__()

        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活,
        # 之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果.
        return self.w2(self.dropout(F.relu(self.w1(x))))
```

### 规范化层

```python

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        features——表示词嵌入的维度,
        eps——它是一个足够小的数, 在规范化公式的分母中出现,防止分母为0.默认是1e-6.
        """
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2
```

{% note info %}
self.a2 是一个用 nn.Parameter 封装的可学习参数张量，它的形状为 (features,)，其中 features 表示输入特征的维度。这个参数控制归一化后的结果的缩放比例。在初始化时，我们将其初始化为一个全为1的张量，表示初始时不进行缩放。

self.b2 同样是一个用 nn.Parameter 封装的可学习参数张量，形状也为 (features,)。这个参数控制归一化后的结果的平移偏移。在初始化时，我们将其初始化为一个全为0的张量，表示初始时不进行平移。

在进行 Layer Normalization 过程中，我们先计算输入张量 x 沿着最后一个维度的均值和标准差，然后对输入进行归一化。归一化的结果为 (x - mean) / (std + eps)，其中 eps 是一个足够小的数，用于防止分母为0的情况。然后，我们将归一化后的结果乘以 self.a2（缩放）并加上 self.b2（平移），得到最终的归一化结果。

{% endnote %}

### 子层连接结构(Add)
```python

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):

        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

### 编码器
```py
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        size，词嵌入维度的大小，它也将作为编码器层的大小
        self_attn，多头自注意力子层实例化对象,自注意力机制 
        feed_froward,前馈全连接层实例化对象
        """
        super(EncoderLayer, self).__init__()

        # 首先将self_attn和feed_forward传入其中.
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 把size传入其中
        self.size = size

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):

        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

## 解码器部分
- 由N个解码器层堆叠而成
- 每个解码器层由三个子层连接结构组成
- 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
- 第二个子层连接结构包括一个多头注意力子层和规范化层以及一个残差连接
- 第三个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接

### 解码器层

Decoder Block 的第一个 Multi-Head Attention 采用了 Mask 操作，第二个 Multi-Head Attention 主要的区别在于 Attention 的 K, V 矩阵不是来自上一个 Decoder Block 的输出计算的，而是来自Encoder的编码信息矩阵C。

```python
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        '''
        size——词嵌入的维度大小，解码器的尺寸
        self_attn——多头自注意力对象（Q=K=V）
        src_attn——多头注意力对象（Q!=K=V）
        feed_forward——前馈全连接层
        dropout——置零比率
        '''
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 克隆三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # memory——来自编码器层的语义存储变量

        # 第一层——自注意力机制
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 第二层——常规注意力机制
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, target_mask))
        # 第三层——前馈全连接层
        return self.sublayer[2](x, self.feed_forward)
```

### 解码器

```python
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

```

## 输出部分

![output](/Pictures/DL/Attention/f6.png)

### 线性层&softmax层
通过对上一步的线性变化得到指定维度的输出,并将向量进行归一化操作。

```python
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)
```

## 最终模型构建

```python
def make_model(source_vocab, target_vocab, N=6, 
               d_model=512, d_ff=2048, head=8, dropout=0.1):

    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```
{% note info %}
nn.Sequential 是 PyTorch 中的一个容器，用于按顺序组合多个神经网络模块（如层、激活函数等），形成一个整体的神经网络模型。它可以简化模型的构建过程，使代码更加简洁易读。

具体地，nn.Sequential 接受一个包含多个神经网络模块的列表或序列作为参数，然后将这些模块按顺序组合在一起，形成一个完整的神经网络模型。当输入数据进入 nn.Sequential 时，它会按照列表中模块的顺序依次进行前向传播，将每个模块的输出作为下一个模块的输入，直到所有模块都被处理完毕，最终得到整个模型的输出。
{% endnote %}

最终我们得到的模型(下方较长)：

```
EncoderDecoder(
  (encoder): Encoder(
    (layers): ModuleList(
      (0-5): 6 x EncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0-1): 2 x SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0-5): 6 x DecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w1): Linear(in_features=512, out_features=2048, bias=True)
          (w2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0-2): 3 x SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )
  (src_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(11, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (tgt_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(11, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (generator): Generator(
    (project): Linear(in_features=512, out_features=11, bias=True)
  )
)
```
---
参考：
1. https://lulaoshi.info/deep-learning/attention/transformer-attention.html#self-attention%E4%B8%AD%E7%9A%84q%E3%80%81k%E3%80%81v
2. https://juejin.cn/post/7125629962769399838
3. http://121.199.45.168:13008/04_mkdocs_transformer/3%20%E8%BE%93%E5%85%A5%E9%83%A8%E5%88%86%E5%AE%9E%E7%8E%B0.html