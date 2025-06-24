---
title: 线性回归
date: 2024-01-30 15:37:35
tags:
- MCM
- RA
categories:
- MCM
index_img: /Pictures/MCM/RA/01.jpg
banner_img: /Pictures/MCM/RA/01.jpg
---

# 一元线性回归

一般的，我们称由 $y=\beta_{0}+\beta_{1}x+\varepsilon $ 确定的模型为一元线性回归模型，记为

$$\begin{cases}
y=\beta_{0}+\beta_{1}x+\varepsilon \\\\
E(\varepsilon)=0,D(\varepsilon)=\sigma^{2}
\end{cases}$$

固定的未知参数 $\beta_{0},\beta_{1}$ 称为回归系数，自变量x称为回归变量。

$Y=\beta_{0}+\beta_{1}x$ 称为**y对x的回归直线方程。**

一元线性回归的主要任务：

1. 用试验值（样本值）对 $\beta_{0},\beta_{1},\varepsilon$ 作估计
2. 用回归系数作假设检验
3. 在 $x=x_{0}$ 处作预测得出 $\hat y$ ，对y作区间估计


## 普通最小二乘法（OLS）

在已知数据的基础上，拟合出函数曲线进行预测。

评判回归精度的标准是：样本回归线上的点与真实观测点的误差应该尽可能小。

OLS给出的判断标准是：二者之差的平方和最小，即

$$Q=\sum_{i=1}^{n}(Y_{i}-\hat Y_{i})^(2)=\sum_{i=1}^{n}(Y_{i}-(\hat \beta_{0}+\beta_{1}X_{i}))^{2}$$

由于Q是关于 $\hat \beta_{0},\hat \beta_{1}$ 的二次函数且非负，故极小值存在。

$$\begin{cases}
\frac{\partial Q}{\partial \hat \beta_{0}}=0 \\\\
 \\\\
\frac{\partial Q}{\partial \hat \beta_{1}}=0
\end{cases}$$

结果为：

$$\begin{cases}
\hat \beta_{1}=\frac{n\sum Y_{i}X_{i}-\sum Y_{i}\sum X_{i}}{n\sum X_{i}^{2}-(\sum X_{i})^{2}} \\\\
 \\\\
\hat \beta_{0}=\bar Y-\hat \beta_{1} \bar X
\end{cases}$$

>其实就是高中的最小二乘法，别被吓到了。

离散形式的版本：

$$\begin{cases}
\bar X=\frac{1}{n} \sum X_{i} \\\\
\bar Y=\frac{1}{n} \sum Y_{i} \\\\
x_{i}=X_{i}-\bar X \\\\
y_{i}=Y_{i}-\bar Y
\end{cases}$$

$$\begin{cases}
\hat \beta_{1}=\frac{\sum x_{i}y_{i}}{\sum x_{i}^{2}} \\\\
 \\\\
\hat \beta_{0}=\bar Y-\hat \beta_{1} \bar X
\end{cases}$$


## 随机误差项方差的估计量

记 $e_{i}=Y_{i}-\hat Y_{i}$ 为残差，则随机误差项方差的估计量为：

$$\hat \delta_{e}^{2}=\frac{\sum e_{i}^{2}}{n-2}$$

## 回归方程的显著性检验

对回归方程的显著性检验，归结为对假设

$$H_{0}:\beta_{1}=0,H_{1}:\beta_{1}\ne 0$$

若 $H_{0}$ 不成立，说明回归方程通过检测。

1. *F*检验法

$$F=\frac{U}{Q_{e}/(n-2)} ~ F(1,n-2)$$

其中 $U=\sum_{i=1}^{n}(\hat y_{i}-\bar y)^{2}$ （回归平方和）

$Q_{e}=\sum_{i=0}^{n}(y_{i}-\hat y_{i})^{2}$ （残差平方和）

$1-\alpha$ 为置信度（即多大把握假设成立）

若 $F>F_{1-\alpha}(1,n-2)$ ，零假设不成立。

2. r检验法

$$r=\frac{\sum_{i=1}^{n}(x_{i}-\bar x)(y_{i}-\bar y)}{\sqrt{\sum_{i=1}^{n}(x_{i}-\bar x)^{2}\sum_{i=1}^{n}(y_{i}-\bar y)^{2}}}$$

查表，大于时，零假设不成立。


# 多元线性回归

我们一般称模型 
$$\begin{cases} Y=X\beta+\varepsilon \\\\ E(\varepsilon)=0,COV(\varepsilon,\varepsilon)=\sigma^{2}I_{n} \end{cases}$$

为高斯-马尔可夫线性模型（k元线性回归模型），简记为 $(Y,X\beta,\sigma^{2}I_{n})$

![img](/Pictures/MCM/RA/plot01.png)

$y=\beta_{0}+\sum_{i=1}^{n}\beta_{i}x_{i}$ 称为回归平面方程

---

多项式回归

设变量X、Y的回归模型为：

$$Y=\beta_{0}+\beta_{1}x+\beta_{2}x^{2}+\ldots+\beta_{p}x^{p}+\varepsilon$$

其中p已知， $\varepsilon$ 服从正态分布。

令 $x_{i}=x^{i}$ ，可以转化为多元线性回归模型。

## 模型参数估计

用最小二乘法求 $\beta_{0},\ldots,\beta_{k}$ 的估计量，作离差平方和：

$$Q=\sum_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1}x_{i1}-\ldots-\beta_{k}x_{ik})$$

选择合适的 $\beta_{0},\ldots,\beta_{k}$ 使Q取最小值

解得估计值为 $\hat \beta=(X^{T}X)^{-1}(X^{T}Y)$ （向量式）

将求出的 $\hat \beta_{i}$ 代入平面方程得到回归方程。


## 线性模型和回归系数的检验

我们依然可使用*F*检验法。

$$F=\frac{U/k}{Q_{e}/(n-k-1)} ~ F(k,n-k-1)$$

其中 $U=\sum_{i=1}^{n}(\hat y_{i}-\bar y)^{2}$ （回归平方和）

$Q_{e}=\sum_{i=0}^{n}(y_{i}-\hat y_{i})^{2}$ （残差平方和）

$1-\alpha$ 为置信度（即多大把握假设成立）

若 $F>F_{1-\alpha}(1,n-2)$ ，零假设不成立。


# 逐步回归分析

在建立回归方程是，我们可以经过筛选，剔除那些对Y影响不显著的变量，而保留影响显著的变量来建立回归方程。

选择的过程有以下几种：

1. 从所有的变量组合的回归方程中选择最优者

2. 从包含全部变量的回归方程中逐次剔除不显著因子

3. 从一个变量开始，把变量逐个引入方程

4. 有进有出逐步回归分析

一般我们选用第四种方法。

