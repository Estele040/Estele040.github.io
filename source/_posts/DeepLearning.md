---
title: DeepLearning
date: 2023-08-07 23:19:01
tags:
categories:
- deep learning
mathjax: true
---

# DeepLearning

关于 dl 一些笔记，或是不懂的问题的记录……:happy:

<!--more-->



## 多层感知机



### 1.1 线性模型可能会出错

例如，线性意味着单调假设：任何特征的增大都有可能导致模型输出的增大（如果相应的权重为正）；或导致模型权重的减小（如果相应的权重为负）。

此外，数据的表示可能考虑到特征之间的相关交互作用。在此表示的基础上建立一个线性模型可能会是合适的， 但我们不知道如何手动计算这么一种表示。 对于深度神经网络，我们使用观测数据来联合学习隐藏层表示和应用于该表示的线性预测器。

### 1.2 在网络中加入隐藏层

我们可以考虑在网络中加入一个或多个隐藏层来克服线性模型的限制，能使其处理更普遍函数之间的关系。要做到这一点，最简单的方法就是将许多全连接层堆叠到一起。每一层都输出到上面的层，直到生成最后的输出。我们可以把前面的$L-1$层看作表示，把最后一层看作线性预测器。这种架构通常称为多层感知机。

### 1.3 从线性到非线性

我们通过$X \in R^{n×h}$来表示n个样本的小批量，其中每个样本具有d个输入特征。对于具有$h$个隐藏单元的单层隐藏多层感知机，用$H\in R^{n*h}$表示隐藏层的输出，称为隐藏表示。在数学或代码中，$H$也被称为隐藏层变量（hidden-layer variable）或隐藏变量（hidden variable）。因为隐藏层和输出层都是全连接的，所以我们具有隐藏层权重$W \in R^{d×h}$和隐藏层偏置$b^{(1)} \in R^{1*h}$以及输出层权重$W^{(2)} \in R^{h×q}$和输出层偏置$b^{(2)} \in R^{1×q}$。形式上我们按如下方式计算单隐藏层多层感知机的输出$O \in R^{n×q}$：
$$
H = XW^{ (1) } + b^{ (1) }，
O = HW^{ (2) }  + b^{ (2) },
\tag{1}
$$
注意在添加隐藏层之后，模型现在需要跟踪和更新额外的参数。 可我们能从中得到什么好处呢？在上面定义的模型里，我们没有好处！ 原因很简单：上面的隐藏单元由输入的仿射函数给出， 而输出（softmax操作前）只是隐藏单元的仿射函数。 仿射函数的仿射函数本身就是仿射函数， 但是我们之前的线性模型已经能够表示任何仿射函数。

为了发挥多层架构的潜力，我们还需要一个额外的关键因素：在仿射变换之后对每个隐藏单元应用非线性激活函数（activation function）$\sigma$。激活函数的输出（例如，$\sigma(.)$）被称为活性值（activation）。一般来说，有了激活函数，就不可能再将我们的多层感知机退化成现行模型：
$$
H = \sigma(XW^{(1)} + b^{ (1) } )
O = HW^{(2)} + b^{ (2) } \tag{2}
$$
由于$X$中的每一行都对应于小批量中的一个样本，处于记号习惯的考量，我们定义非线性函数$\sigma$也以按行的方式作用于其输入，即一次计算一个样本。但是本节应用于隐藏层的激活函数通常不按行进行操作，也按元素操作。

这意味着，在计算每一层的线性部分之后，我们可以计算每个活性值，而不需要查看其他隐藏单元所取的值。对于大多数激活函数都是这样。

### 1.4 通用近似定理

多层感知机可以通过隐藏神经元，捕捉到输入之间复杂的相互作用， 这些神经元依赖于每个输入的值。 我们可以很容易地设计隐藏节点来执行任意计算。例如，在一对输入上进行基本的逻辑操作，多层感知机是通用近似器。即使网络只有一个隐藏层，给足够的神经元和足够的权重，我们可以对任意函数建模，尽管实际中学习该函数是很困难的神经网络有点像C语言。 C语言和任何其他现代编程语言一样，能够表达任何可计算的程序。 但实际上，想出一个符合规范的程序才是最困难的部分。

而且，虽然一个单隐层网络能学习任何函数， 但并不意味着我们应该尝试使用单隐藏层网络来解决所有问题。 事实上，通过使用更深（而不是更广）的网络，我们可以更容易地逼近许多函数。 我们将在后面的章节中进行更细致的讨论。

### 1.5 激活函数

激活函数（activate function）通过计算加权和并加上偏置来确定神经元是否应该被激活，他们将输入信号转换为输出的可微运算，大多数激活函数都是非线性的。由于激活函数是深度学习的基础，下面介绍一些简单的激活函数。

### 1.6 ReLU函数

最受欢迎的激活函数是线性修正单元，因为它实现简单，同时在各种预测任务中表现良好。ReLU提供了一种非常简单的线性变换。给定元素$x$，ReLU函数被定义为该元素于0的最大值：
$$
ReLU(x) = max(x,0)
$$
通俗地说， ReLU 函数通过将相应的活性值设置为0，仅保留正元素，并丢弃所有负元素。为了直观的感受一下，我们可以画出函数的曲线图。正如从图中所看到，激活函数是分段线性的。

```python
y = torch.arange(-8.0, 8.0, 0.1, requires_grad = True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
#返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
```

> 这样我们就会继续使用这个新的`tensor进行计算，后面当我们进行`反向传播时，到该调用detach()的`tensor`就会停止，不能再继续向前进行传播。
>
> 注意：使用detach()返回的Tensor和原始的tensor共用一个内存，即一个修改另一个也会跟着改变。
>
> 当使用detach()分离tensor但是没有更改这个tensor时，并不会影响backward()。
>
> 当使用detach()分离tensor，然后用这个分离出来的tensor去求导数，会影响backward()，会出现错误。
>
> 当使用detach()分离tensor并且更改这个tensor时，即使再对原来的out求导数，会影响backward()，会出现错误。

![8](/home/xxfs/study/recording/deep_learning/photos/2023-08-14 20-48-23 的屏幕截图.png)

当输入为负数时，ReLU导数为0，当输入为正数时，ReLU函数的导数为1。注意，输入值精确等于0时，ReLU函数不可导。在此时，我们默认使用左边导数，即当输入0的导数为0。我们可以忽略这种情况，因为输入可能永远都不会是0。

```python
y.backward(torch.ones_like(x), retrain_graph=True)
d2l.plot(x.detach(), x.grad(), 'x', 'grad of relu', figsize = (5, 2.5))
```



下面我们绘制ReLU函数的导数。

![7](/home/xxfs/study/recording/deep_learning/photos/2023-08-14 20-53-33 的屏幕截图.png)下面我们绘制ReLU函数的导数。

使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且**ReLU减轻了困扰以往神经网络的梯度消失问题。**

注意，ReLU函数有很多变体，包括参数化ReLU函数。改变体为ReLU添加了一个线性项，因此即使参数是负的，某些信息仍然可以通过：
$$
pReLU(x) = max(0,x) + \alpha min(0, x)
$$

### 1.7 sigmoid函数

对一个定义域在$R$上的输入，sigmoid函数将输入变换为区间$(0,1)$上的输出。因此，sigmoid函数通常称为挤压函数：它将范围$(-inf, inf)$中的任意输入压缩到区间$(0,1)$中的某个值：
$$
sigmoid(x) = \frac{1} {1 + exp(-x)}
$$
 注意，当输入接近0时，sigmoid函数接近线性变换。

```python
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize = (5, 2.5))
```

sigmoid的导数为以下公式$\frac{d} {dx}sigmoid(x) = \frac{exp(-x)}{ { (1+exp(-x)) }^2} = sigmoid(x)(1-sigmoid(x))$ 

sigmoid函数的导数图像如下。注意，当输入为0时，sigmoid函数的导数最大可以达到0.25；而输入在任意方向上越远离0时，导数越接近0。

```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

![6](/home/xxfs/study/recording/deep_learning/photos/2023-08-14 21-18-27 的屏幕截图.png)

### 1.8 tanh 函数

与sigmoid函数类似，tanh （双曲正切）函数也能将其输入压缩转换到区间$(-1,1)$上。tanh 函数的公式如下：
$$
tanh(x) = \frac{1 - exp(-2x)}{1 + exp(-2x)}
$$
下面我们绘制tanh 函数。注意，当输入在0附近时，tanh函数接近线性变换。函数的形状类似于sigmoid函数，不同的是tanh函数关于坐标系远点中心对称。

```python
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

![5](/home/xxfs/study/recording/deep_learning/photos/2023-08-14 21-52-57 的屏幕截图.png)

tanh 的物理导数是：
$$
\frac{d}{dx}tanh(x) = 1 - tanh^2(x)
$$
tanh 函数的导数图像如下所示。 当输入接近0时，tanh函数的导数接近最大值1。 与我们在sigmoid函数图像中看到的类似， 输入在任一方向上越远离0点，导数越接近0。

```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

![4](/home/xxfs/study/recording/deep_learning/photos/2023-08-14 22-07-56 的屏幕截图.png)





## 模型选择，过拟合和欠拟合

### 误差

#### 训练误差

训练误差是指模型在训练集上的错分样本比率，说白了就是在训练集上训练完毕后在训练集本身上进行预测得到了错分率

#### 泛化误差

*泛化误差*（generalization error）是指， 模型应用在同样从原始样本的分布中抽取的无限多数据样本时，模型误差的期望。

问题是我们永远不能准确的计算出泛化误差。这是因为无限多的数据样本是一个虚构的对象。在实际中，我们只能通过模型应用于一个独立的测试集来估计泛化误差，该测试集由随机选取的，未曾在训练集中出现的样本构成。

泛化误差的意义，其实就是在模型训练后查看模型是否具有代表性。

泛化误差的公式为:$E_G(\omega) = \sum_{x \in X}p(x)(\hat f (x|\omega) - f(x))^2$，即全集X中x出现的概率乘以其相对应的训练误差。

但是过分追求低训练误差会使得模型过拟合于训练集反而不使用于其他数据。

因此在样本集划分时，如果得到的训练集与测试集的数据没有交集，此时测试误差基本等同于泛化误差。



### 系统学习理论

#### 独立同分布

假设训练数据和测试数据都是从相同的分布中独立提取的，这通常被称为独立同分布假设，这意味这对数据进行采样的过程没有进行"记忆"。

影响模型泛化的因素：

1. 可调整参数的数量。当可调整参数的数量（有时称为*自由度*）很大时，模型往往更容易过拟合。
2. 参数采用的值。当权重的取值范围较大时，模型可能更容易过拟合。
3. 训练样本的值。即使模型很简单，也很容易过拟合只包含一两个样本的数据集。而过拟合一个有数百万个样本的数据集则需要一个极其灵活的模型。

#### 模型选择

在机器学习中，在我们确定所有超参数之前，我们不希望用到测试集。如果我们在模型选择过程中使用测试数据，有可能会过拟合测试数据的风险，那就麻烦大了。如果我们过拟合来训练数据，还可以在测试数据上的评估来判断过拟合。但是如果我们拟合了测试数据集，我们又该怎么知道呢?

因此，我们决不能靠测试数据进行模型的选择。然而，我们也不能依靠训练模型来选择模型，因为我们无法估计训练数据的泛化误差。

在实际应用中，情况变得更加复杂。虽然理想情况下，我们只会使用测试数据一次，以评估最好的模型或比较一些模型的效果，但现实是测试数据很少在使用一次后被丢弃。我们很少能有充足的实验来对每一轮实验才用全新的测试集。

解决此问题的常见做法是将我们的数据分成三份，除了训练集和测试集外，还增加依一个验证数据集，也叫验证集（validation dataset）。但现实是验证数据和测试数据之间模糊地令人担忧。除非另有明确说明，否则在本书的实验中，我们实际上实在使用应该被正确地称为训练数据和验证数据的数据集，并没有真正的测试数据集。因此，文中每次实验报告的准确度都是验证集准确度，而不是测试集准确度。

#### K折交叉验证

当训练数据稀缺时，我们甚至可能无法提供足够的数据来构成一个适合的验证集。这个问题的一个流行解决方案是采用K折交叉验证。这里，原始训练数据被分成K个不重叠的子集。然后执行K次模型训练和验证，每次在$K-1$个子集上进行训练，并在剩余一个子集（该轮中没有用于训练的子集）进行验证。最后，通过对K次实验的结果取平均值来估计训练和验证的误差。



### 欠拟合&&过拟合

#### 欠拟合

欠拟合是指模型不能在训练集上获得足够低的误差。换句换说，就是模型复杂度低，模型在训练集上就表现很差，没法学习到数据背后的规律。

当我们比较训练和验证误差时，我们要注意两种常见的情况。

#### 如何解决欠拟合

欠拟合基本上都会发生在训练刚开始的时候，经过不断训练之后欠拟合应该不怎么考虑了。但是如果真的还是存在的话，可以通过**增加网络复杂度**或者在模型中**增加特征**，这些都是很好解决欠拟合的方法。



#### 过拟合

过拟合是指训练误差和测试误差之间的差距太大。换句换说，就是模型复杂度高于实际问题，**模型在训练集上表现很好，但在测试集上却表现很差**。模型对训练集"死记硬背"（记住了不适用于测试集的训练集性质或特点），没有理解数据背后的规律，**泛化能力差**。



#### **为什么会出现过拟合现象？**

造成原因主要有以下几种：
1、**训练数据集样本单一，样本不足**。如果训练样本只有负样本，然后那生成的模型去预测正样本，这肯定预测不准。所以训练样本要尽可能的全面，覆盖所有的数据类型。
2、**训练数据中噪声干扰过大**。噪声指训练数据中的干扰数据。过多的干扰会导致记录了很多噪声特征，忽略了真实输入和输出之间的关系。
3、**模型过于复杂。**模型太复杂，已经能够“死记硬背”记下了训练数据的信息，但是遇到没有见过的数据的时候不能够变通，泛化能力太差。我们希望模型对不同的模型都有稳定的输出。模型太复杂是过拟合的重要因素。



#### 如何防止过拟合？

通过正则化：修改学习算法，使其降低泛化误差而非训练误差。

常用的正则化方法根据具体的使用策略不同可以分为：

1. 直接提供正则化约束的参数正则化方法，如$L1/L2$正则化；
2. 通过工程上的技巧来实现更低泛化误差的方法，如提前终止（early stopping）和（Drop）
3. 不直接提供约束的隐式正则化方法，如数据增强等等。



**1. 获取和使用更多的数据（数据集增强） -----解决过拟合的根本性方法**

让机器学习或深度学习模型泛化能力更好的办法就是使用更多的数据进行训练。但是，在实践中，我们拥有的数据量是有限的。解决这个问题的一种方法就是**创建“假数据”并添加到训练集中——数据集增强**。通过增加训练集的额外副本来增加训练集的大小，进而改进模型的泛化能力。

我们以图像数据集举例，能够做：旋转图像、缩放图像、随机裁剪、加入随机噪声、平移、镜像等方式来增加数据量。另外补充一句，在物体分类问题里，**CNN在图像识别的过程中有强大的“不变性”规则，即待辨识的物体在图像中的形状、姿势、位置、图像整体明暗度都不会影响分类结果**。我们就可以通过图像平移、翻转、缩放、切割等手段将数据库成倍扩充。



**2. 采用适合的模型（控制模型的复杂度）**

对于过于复杂的模型会带来过拟合问题1。对于模型的设计，目前公认的一个深度学习的规律是"deeper is better"。比如许多大牛通过实验和竞赛发现，对于CNN来说，层数越多，效果越好，但也更容易产生过拟合，并且计算所耗费的时间也越长。

**对于模型的设计而言，我们应该选择简单、合适的模型解决复杂的问题。**



**3.降低特征的数量**

对于一些特征工程而言，可以降低特征的数量——删除冗余特征，人工选择保留哪些特征。这种方法也可以解决过拟合问题。



**4. L1/L2正则化**

**(1) L1正则化**

在原始的损失函数后面加上一个L1正则化项

首先，我们要注意这样的情况：

1. 训练误差和验证误差都很严重；

2. 训练误差和验证误差之间仅有一点差距。

如果模型不能降低训练误差，这可能意味着模型过于简单（即表达能力不足），无法捕获试图学习的模式。此外，由于我们的训练和验证误差之间的泛化误差很小，我们有理由相信可以用一个更复杂的模型降低训练误差。这种现象被称为欠拟合（underfitting）。

另一方方面，当我们的训练误差明显小于验证误差时要小心，这表明严重的过拟合（overfitting）。 注意，*过拟合*并不总是一件坏事。 特别是在深度学习领域，众所周知， 最好的预测模型在训练数据上的表现往往比在保留（验证）数据上好得多。 最终，我们通常更关心验证误差，而不是训练误差和验证误差之间的差距。

**过拟合或欠拟合的因素：**

1. 模型的复杂性；
2. 训练数据集的大小。

**模型的复杂性**

为了说明一些关于过拟合和模型复杂性的经典直觉，我们给出一个多项式的例子。给定由单个特征$x$和和对应实数标签$y$组成的训练数据，我们试图找到下面的$d$阶多项式来估计标签$y$。
$$
\hat{y} = \sum \limits_{i=0}^d x^i \omega_i
$$
由于这是一个线性回归问题，我们可以用平方误差作为我们的损失函数。

高阶函数比低阶函数复杂得多，高阶函数的参数较多，模型的选择范围较广。因此在固定训练数据集的情况下，高阶多项式函数相对于低阶多项式的的训练误差应该始终更低（最坏也是相等）。事实上，当数据样本包含了$x$的不同值时，函数阶数等于样本数据量的多项式函数可以完美拟合训练集。下图中我们直观描述了过拟合和欠拟合的关系。

![3](/home/xxfs/study/recording/deep_learning/photos/2023-08-18 10-37-40 的屏幕截图.png)

**数据集大小**

另一个重要因素是数据集的大小。 训练数据集中的样本越少，我们就越有可能（且更严重地）过拟合。 随着训练数据量的增加，泛化误差通常会减小。 此外，一般来说，更多的数据不会有什么坏处。 对于固定的任务和数据分布，模型复杂性和数据集大小之间通常存在关系。 给出更多的数据，我们可能会尝试拟合一个更复杂的模型。 能够拟合更复杂的模型可能是有益的。 如果没有足够的数据，简单的模型可能更有用。 对于许多任务，深度学习只有在有数千个训练样本时才优于线性模型。 从一定程度上来说，深度学习目前的生机要归功于 廉价存储、互联设备以及数字化经济带来的海量数据集。



### 多项式回归

我们现在可以通过多项式拟合来探索这些概念。

```python
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
```



#### 生成数据集

给定$x$，我们将使用以下三阶多项式来生成训练和测试数据的标签：
$$
y = 5 + 1.2x -3.4\frac{x^2}{2!} + 5.6\frac{x^3}{3!} + \epsilon \quad where \quad \epsilon ～ N(0,0.1^2).
$$
 噪声$\epsilon$服从均值为0，标准差为1的正太分布。在优化的过程中，我们通常希望避免非常大的梯度值或损失值。这就是我们将特征从$x^i$调整为$\frac{x^i}{i!}$的原因，这样可以避免很大的$i$带来特别大的指数值。我们将训练集和测试集各生成100个样本。

```python
max_degree = 20 #多项式的最大阶数
n_train, n_test = 100 #训练和测试数据集将大小
true_w = np.zeros(max_degree) # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    ploy_features[:,i] /= math.gamma(i+1) #gamma(n) = (n-1)!
# labels的维度（n_train + n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale = 0.1, size = labels.shape)
```

同样，存储在ploy_features中的单项式由gamma函数重新缩放，其中$\Gamma(n) = (n-1)!$。从生成的数据集中查看一下前两个样本，第一个值是与偏置相对应的常量特征。

```python
# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

features[:2], poly_features[:2, :], labels[:2]
```

#### 对模型进行训练和测试

```python
def train(train_features, test_features, train_labels, test_labels
         num_epochs = 400):
    loss = nn.MESLoss(reduction='none')
    input_shape = train_features.shape[-1]
    #不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                               batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                              batch_size, is_train = False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', y.scale='log',
                           xlim=[1,num_epochs], ylim = [1e-3, 1e2],
                           legend = ['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch==0 or (epoch + 1)%20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter,loss),
                                    evaluate_loss(net, test_iter,loss)))
            
    print('weight:', net[0].weight.data.numpy())
```



#### 三阶多项式函数拟合

我们将首先使用三阶多项式函数，它与数据生成函数的阶数相同。 结果表明，该模型能有效降低训练损失和测试损失。学习到的模型参数也接近真实值$\omega=[5, 1.2, -3.4, 5.6]$。

```python
#从多项式特征中选取前四个维度，即1, x, x^2/2!, x^3/3!
train(poly_features[:n_train, :4], ploy_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

```python
weight: [[ 4.993645   1.2287872 -3.3972282  5.559377 ]]
```

![2](/home/xxfs/study/recording/deep_learning/photos/2023-08-18 14-51-23 的屏幕截图.png)



#### 线性函数拟合（欠拟合）

让我们再看看线性函数拟合，减少该模型的训练损失相对困难。 在最后一个迭代周期完成后，训练损失仍然很高。 当用来拟合非线性模式（如这里的三阶多项式函数）时，线性模型容易欠拟合。

```python
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

```python
weight: [[2.5148914 4.2223625]]
```

![2023-08-18 14-54-34 的屏幕截图](/home/xxfs/study/recording/deep_learning/photos/2023-08-18 14-54-34 的屏幕截图.png)

#### 高阶多项式拟合（过拟合）

现在，让我们尝试使用一个阶数过高的多项式来训练模型。 在这种情况下，没有足够的数据用于学到高阶系数应该具有接近于零的值。 因此，这个过于复杂的模型会轻易受到训练数据中噪声的影响。 虽然训练损失可以有效地降低，但测试损失仍然很高。 结果表明，复杂模型对数据造成了过拟合。

```python
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

![1](/home/xxfs/study/recording/deep_learning/photos/2023-08-18 14-56-01 的屏幕截图.png)





## 权重衰减

### L1/L2正则化和权重衰减

L2范数也被称为欧几里得范数，可以简单理解为向模长。

范数定义的公式如下：
$$
||x||_p :=  (\sum_{i = 1}^{n}|x_i|^p)^{\frac{1}{p}}
$$

#### L1范数

$p= 1$时称为$L1$范数(L1-norm)：
$$
||x||_1 := \sum^n_{i = 1}|x_i|
$$
$L1$范数是一组数的绝对值累加和。

#### L2范数

$p = 2$时，称为$L2$范数：
$$
||x||_2 := (\sum_{i =1}^n x^{(i)})^{\frac{1}{2}}
$$
可以理解为空间或平面内某一点到原点的距离。



#### L1/L2正则化和权重衰减

通过在loss上增加了$L1$或$L2$范数项，达到参数惩罚的作用，即实现了正则化效果，从而称为$L1/L2$正则化。

{% asset_img example.jpg This is an example image %}

![2023-09-15 21-17-12 的屏幕截图](/source/images/2023-09-15 21-17-12 的屏幕截图.png)

由于其高次项参数的使用，使得模型对训练数据过分拟合，导致对未来更一般的数据预测性大大下降，为了缓解这种过拟合的现象，我们可以采用L2正则化。 使用$L2$范数的一个原因是它对权重向量的大分量施加了巨大的惩罚。 这使得我们的学习算法偏向于在大量特征上均匀分布权重的模型。具体来说就是在原有的损失函数上添加L2正则化项(l2-norm的平方)：

原来的损失：
$$
Q(\theta) = \frac{1}{2n} \sum_{i=1}^n (\hat{y} - y)^2
$$
加上$L2$正则化项后的损失：
$$
J(\theta) = Q(x) + \frac{1}{2n} \lambda \sum_{j=1}^{n} \theta_j^2
$$
这里，通过正则化系数$\lambda$可以较好地惩罚高次项的特征，从而起到降低过拟合，正则化的效果。

添加$L2$正则化修正以后的模型：

![2023-09-15 21-39-05 的屏幕截图](source/images/2023-09-15 21-39-05 的屏幕截图.png)

### 权重衰减

权重衰减weight decay，并不是一个规范的定义，而只是俗称而已，可以理解为削减/惩罚权重。在大多数情况下weight dacay 可以等价为L2正则化。L2正则化的作用就在于削减权重，降低模型过拟合，其行为即直接导致每轮迭代过程中的权重weight参数被削减/惩罚了一部分，故也称为权重衰减weight decay。从这个角度看，不论你用L1正则化还是L2正则化，亦或是其他的正则化方法，只要是削减了权重，那都可以称为weight dacay。从这个角度看，不论你用$L1$正则化还是$L2$正则化，亦或是其他的正则化方法，只要是削减了权重，那都可以称为weight dacay。

设：

- 参数矩阵为p（包括weight和bias）；
- 模型训练迭代过程中计算出的loss对参数梯度为d_p；
- 学习率lr；
- 权重衰减参数为decay

则不设dacay时，迭代时参数的更新过程可以表示为：
$$
p = p - lr × d\_p
$$
增加weight_dacay参数后更新过程可以表示为：
$$
p = p - lr × （d\_p + p × dacay)
$$

### 代码实现

在深度学习框架的实现中，可以通过设置weight_decay参数，直接对weight矩阵中的数值进行削减（而不是像L2正则一样，通过修改loss函数）起到正则化的参数惩罚作用。二者通过不同方式，同样起到了对权重参数削减/惩罚的作用，实际上在通常的随机梯度下降算法(SGD)中，通过数学计算L2正则化完全可以等价于直接权重衰减。（少数情况除外，譬如使用Adam优化器时，可以参考：[L2正则=Weight Decay？并不是这样](https://zhuanlan.zhihu.com/p/40814046)）

正因如此，深度学习框架通常实现weight dacay/L2正则化的方式很简单，直接指定weight_dacay参数即可。

在pytorch/tensorflow等框架中，我们可以方便地指定weight_dacay参数，来达到正则化的效果，譬如在pytorch的sgd优化器中，直接指定weight_decay = 0.0001：

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001)
```



```python
import torch
from torch import nn
from d2l import torch as d2l

#0.05 + 0.01X + e where e \in N(0, 0.01^2)

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

#定义一个函数来随机初始化参数模型
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w,b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#定义训练代码的实现
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
    

#忽略正则化直接进行训练
train(lambd=0)

#使用权重衰减
train(lambd=3)
```



### 简洁实现

由于权重衰减在神经网络优化中很常用， 深度学习框架为了便于我们使用权重衰减， 将权重衰减集成到优化算法中，以便与任何损失函数结合使用。 此外，这种集成还有计算上的好处， 允许在不增加任何额外的计算开销的情况下向算法中添加权重衰减。 由于更新的权重衰减部分仅依赖于每个参数的当前值， 因此优化器必须至少接触每个参数一次。

```python
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
```





## 暂退法 Dropout

#### 重新审视过拟合

当面对更多的特征而样本不足时，线性模型往往会过拟合。相反，当给出更多样本而不是特征，通常线性模型不会过拟合。 不幸的是，线性模型泛化的可靠性是有代价的。 简单地说，线性模型没有考虑到特征之间的交互作用。 对于每个特征，线性模型必须指定正的或负的权重，而忽略其他特征。

泛化性和灵活性之间的权衡被描述为**偏差-方差权衡**。线性模型有很高的偏差：它们只能表示一小类函数。然而，这些模型的方差很低：它们在不同的随机数据样本上可以得出相似的结果。

深度学习网络位于偏差-方差谱的另一端。于线性模型不同，神经网络并不局限于查看每个特征，而是学习特征之间的交互。

在探究泛化之前，我们先来定义以下什么是“好”的预测模型？我们期待好的预测模型能在未知的数据上有很好的表现， 经典泛化理论认为，为了缩小训练和测试性能之间的差距，应该以简单的模型为目标。

简单性的另一个度量角度是平滑性，即函数不应该对其输入的微小变化而敏感 例如，当我们对图像进行分类时，我们预计向像素添加一些随机噪声应该是基本无影响的。在2014年，斯里瓦斯塔瓦等人就如何将毕晓普的想法应用于网络的内部层提出了一个想法： 在训练过程中他们建议在计算后续层之前向网络的每一层注入噪声。 因为当训练一个有多层的深层网络时，注入噪声只会在输入-输出映射上增强平滑性。

这个想法被称为暂退法。暂退法在前向传播过程中，计算每一层内部的同时注入噪音，这已经成为训练神经网络的常用技术。这种方法之所以被称为暂退法，因为我们表面上看是在训练过程中丢弃的一些神经元。在整个训练过程的每一次迭代中，标准暂退法包括在计算下一层之前将当前层中的一些节点置零。

需要说明的是，暂退法的原始论文提到了一个关于有性繁殖的类比： 神经网络过拟合与每一层都依赖于前一层激活值相关，称这种情况为“共适应性”。 作者认为，暂退法会破坏共适应性，就像有性生殖会破坏共适应的基因一样。

那么关键的挑战就是如何注入这种噪声。 一种想法是以一种*无偏向*（unbiased）的方式注入噪声。 这样在固定住其他层时，每一层的期望值等于没有噪音时的值。

可以考虑将高斯噪声加入到线性模型的输入中。在没次训练中，他将从均值为零的分布$\epsilon ～ N（0,\sigma)$采样噪声添加到输入$x$，从而产生扰动点$x' = x + \epsilon$，期望是$E[x'] = x$。

在标准暂退法正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差。 换言之，每个中间活性值ℎ以*暂退概率$p$由随机变量$ℎ′$替换，如下所示：
$$
h' = 
\left\{
\begin{array}{**lr**}  
0 \quad 概率为0
\\
\frac{h}{1-p} \quad 其他情况
\end{array}  
\right.  
$$
根据此模型的设计，其期望值保持不变，即$E[x'] = x$。

#### 总结

dropout相当于给出一个概率$p$，比方说$p=40\%$，那么就是说有$40\%$的文件要被删除，只留下$60%$的神经元，那么这就是我们的表面理解。对于程序来说，就是将这40%的神经元赋值0，那么可以想一下一个神经元等于0了，那么他对下一层还能产出结果吗，0乘多少权重都是0，相当于这个被dropout选中的神经元没价值了，那他就相当于被删了。



#### 实践中的暂退法

带有1个隐藏层和5个隐藏单元的多层感知机。 当我们将暂退法应用到隐藏层，以$p$的概率将隐藏单元置为零时， 结果可以看作一个只包含原始神经元子集的网络。假设隐藏单元为$h1,h2,h3,h4,h5$，我们删除了$h2,h5$，因此输出的计算不依赖$h2,h5$并且它们各自的梯度在之执行反向传播也会消失。这样，输出层的计算不能过度依赖$h1,...,h5$中的任意一个元素。

#### 从零开始实现

要实现单层的暂退法函数，我们从均匀分布$U[0,1]$中抽取样本，样本数于这层神经网络的维度一致。然后我们保留那些对应样本大于$p$的节点，把剩下的丢弃。

在下面的代码中，我们实现dropout_layer函数，该函数以dropout的概率丢弃丢弃张量输入X中的元素，如上述重新缩放剩余部分：将剩余部分除以1.0-dropout。

```python
import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

#### 定义模型参数

引入Fashion-MNIST数据集。我们定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元。

```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

#### 定义模型

我们可以将暂退法应用于每个隐藏层的输出（在激活函数之后）， 并且可以为每一层分别设置暂退概率： 常见的技巧是在靠近输入层的地方设置较低的暂退概率。 下面的模型将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5， 并且暂退法只在训练期间有效。

```python
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

#### 训练和测试

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

#### 简洁实现

```python
import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```



## 前向传播、反向传播和计算图

我们已经学习了如何用小批量随机梯度下降训练模型。 然而当实现该算法时，我们只考虑了通过*前向传播*（forward propagation）所涉及的计算。 在计算梯度时，我们只调用了深度学习框架提供的反向传播函数，而不知其所以然。

梯度的自动计算（自动微分）大大简化了深度学习算法的实现。 在自动微分之前，即使是对复杂模型的微小调整也需要手工重新计算复杂的导数， 学术论文也不得不分配大量页面来推导更新规则。 本节将通过一些基本的数学和计算图， 深入探讨*反向传播*的细节。 首先，我们将重点放在带权重衰减（ $L2$ 正则化）的单隐藏层多层感知机上。

#### 前向传播

前向传播指的是：按顺序（从输入层到输出层）计算和存储神经网络中每层的结果。

我们将一步一步研究单隐藏层神经网络的机制，为简单起见，我们假设输入样本是$x \in R^d$，并且我们的隐藏层不包括偏置项。这里的中间变量是：
$$
z = W^{(1)}x
$$
其中$W^{(1)} \in R^{h*d}$是隐藏层的权重参数。将中间变量$z \in R^h$通过激活函数$\phi$，我们得到长度为$h$的隐藏激活向量：
$$
h = \phi(z)
$$
隐藏变量$h$也是一个中间变量。假设输出层的参数只有权重$W^{(2)} \in R^{q*h}$，我们可以得到输出层的变量，它是一个长度为$q$的向量：
$$
o = W^{(2)}h
$$
假设损失函数为$l$，样本标签为$y$，我们可以单个计算数据样本的损失项，$L = l(o,y)$

根据$L2$正则化的定义，给定超参数$\lambda$，正则化项为
$$
s = \frac{\lambda}{2}(||W||_F^2 + ||W||_F^2)
$$
其中矩阵的Frobenius范数是将矩阵展平为向量后应用的$L2$范数。最后，模型在给定数据样本上的正则化损失为：
$$
J = L + s
$$
在下面讨论中，我们将$J$称为目标函数。

#### 反向传播

反向传播指的是计算神经网络参数梯度的方法。简言之，该方法根据微积分中的链式规则，按相反的顺序从输出层到输入层遍历网络。该算法存储了计算某些参数梯度时所需的任何中间变量（偏导数）。假设我们有函数$Y=f(X)$和$Z = g(X)$，其中输入和输出为$X,Y,Z$是任意形状的张量。利用链式法则，我们可以计算$Z$关于$X$的导数
$$
\frac{\partial Z}{\partial L} = prod(\frac{\partial Z}{\partial Y}, \frac{\partial{Y}}{\partial X})
$$
这里我们使用prod运算符在执行必要的操作（如换位和交换输入位置）后将其参数相乘。对于向量，这很简单，它只是矩阵-矩阵乘法。对于高维向量，我们使用适当的对因项。运算符prod代指了所有的这些符号。

回想以下，在计算图中的单隐藏层简单网络的参数是$W^{(1)}$和$W^{(2)}$。反向传播的目的是计算度$\partial J/\partial W^{(1)}$和$\partial J/ \partial W^{(2)}$。为此，我们应用链式法则，依次计算每个中间变量和参数的梯度。计算的顺序与前向传播中执行的顺序相反，因为我们需要从计算图的结果开始，并朝着参数的方向努力。第一步是计算目标函数$J = L + s$相对于损失项L和正则项$s$的梯度。
$$
\frac{\partial J}{\partial o}= prod(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial o}) = \frac{\partial L}{\partial o} \in R^q
$$
接下来，我们计算正则化项两个参数的梯度：

$\frac{\partial s}{\partial W^{(1)}} = \lambda W^{(1)} and \frac{\partial s}{\partial W^{(2)}} = \lambda W^{(2)}$

现在我们可以计算最接近输出层的模型的梯度$\frac{\partial J}{\partial W^{(2)}} \in R^{q*h}$。使用链式法则得出：
$$
\frac{\partial J}{\partial W^{(2)}} = prod(\frac{\partial J}{\partial o}, \frac{\partial o}{\partial W^{(2)}}) + prod(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial W^{(2)}}) = \lambda W^{(2)}
$$
为了获得关于$W^{(1)}$的梯度，我们需要继续沿着输出层到隐藏层反向传播。关于隐藏层输出的梯度$\partial J/ \partial h \in R^h$由下式给出：
$$
\frac{\partial J}{\partial h} = prod(\frac{\partial J}{\partial o}) = W^{(2)^T} \frac{\partial J}{\partial o}
$$
由于激活函数$\phi$是按元素计算的，计算中间变量$z$的梯度$\partial J/ \partial z \in R^n$需要使用按元素乘法运算符，我们用$\odot$来表示：
$$
\frac{\partial J}{\partial z} = prod(\frac{\partial J}{\partial h}, \frac{\partial h}{\partial z}) = \frac{\partial J}{\partial h } \odot \phi'(z)
$$
最后，我们可以得到最接近输入层的的模型参数的梯度$\partial J / \partial W^{(1)} \in R^{h*d}$。根据链式法则，我们得到：
$$
\frac{\partial J}{\partial W^{(1)}} = prod(\frac{\partial J}{\partial z},\frac{\partial z}{\partial W^{(1)}}) + prod(\frac{\partial J}{\partial s}, \frac{s}{W^{(1)}}) = \frac{\partial J}{\partial z}x^T + \lambda W^{(1)}
$$

#### 训练神经网络

在训练神经网络时，前向传播和反向传播相互依赖。 对于前向传播，我们沿着依赖的方向遍历计算图并计算其路径上的所有变量。 然后将这些用于反向传播，其中计算顺序与计算图的相反。

以上述简单网络为例：一方面，在前向传播期间计算正则项取决于模型参数$W^{(1)}$和$W^{(2)}$的当前值。它们是由优化算法根据最近迭代的反向传播给出的。另一方面，反向传播期间参数的梯度计算，取决于由前向传播给出的隐藏层变量$h$的当前值。

因此，在训练神经网络时，在初始化模型参数后， 我们交替使用前向传播和反向传播，利用反向传播给出的梯度来更新模型参数。 注意，反向传播重复利用前向传播中存储的中间值，以避免重复计算。 带来的影响之一是我们需要保留中间值，直到反向传播完成。 这也是训练比单纯的预测需要更多的内存（显存）的原因之一。 此外，这些中间值的大小与网络层的数量和批量的大小大致成正比。 因此，使用更大的批量来训练更深层次的网络更容易导致*内存不足*（out of memory）错误。



## 数值稳定和模型初始化

### 前言



### part1：为什么要用梯度更新

在介绍梯度消失以及爆炸之前，先简单的说一说梯度消失的根源——深度神经网络和反向传播。目前深度学习方法中，深度神经网络的发展造就了我们可以构建更深层网络完成复杂任务，深度网络比如深度卷积网络，LSTM等等，而且最终的结果表明，在处理复杂任务上，深度网络比浅层的网络具有更好的效果。但是，目前优化神经网络的方法都是基于反向传播的思想，即根据损失函数计算的误差通过梯度反向传播的方式，指导深度网络权值的更新优化。这样做是有一个原因的，首先，深层网络有许多非线性层堆叠而来，每一层非线性层都可以视为是一个非线性函数$f(x)$（非线性来自于非线性激活函数数），因此整个深度网络可以视为是一个复合的非线性多元函数。
$$
F(x) = f_n(...f_3(f_2(f_1(x)*\theta_1 + b)*\theta_2 + b)...)
$$
我们最终的目的是希望这个多元函数可以很好的完成输入到输出之间的映射，假设不同的输入，输出的最优解是$g(x)$，那么，优化深度网络就是为了寻找到适合的权值，满足$Loss = L(g(x), F(x))$取得极小值点，比如最简单的损失函数
$$
Loss = ||g(x) - f(x)||_2^2
$$
假设损失函数的数据空间是下图这样的，我们最优的权值就是为了寻找下图中的最小值点，对于这种数学寻找最小值问题，采用梯度下降的方法再适合不过了:

![](/home/xxfs/repository/blog/source/_posts/DeepLearning/v2-dca032791074e6660129fa3d7304d1b6_720w.webp)

### part2：梯度消失、梯度爆炸

梯度消失和梯度爆炸其实是一种情况，看接下来的文章就知道了。两种情况下梯度消失经常出现，一是在深层网络中，二是采用了不适合的损失函数，比如sigmoid。梯度爆炸一般出现在深层网络和权值初始化值太大的情况下，下面分别从这两个角度分析梯度消失和梯度爆炸的原因。

#### 1.深层网络角度

比较简单的深层网络如下：

![](/home/xxfs/repository/blog/source/_posts/DeepLearning/v2-94d5576a89767793a67e6976775a6b13_720w.webp)

假设是一个四层的全连接网络，假设每一层网络激活后的输出为$fi(x)$，其中$i$为第$i$层，$x$代表第$i$层的输入，也就是$i-1$层输入，$f$是激活函数，那么得出$f_{i+1} = f(f_i * w_{i+1} + b_{i+1})$，简单记为$f_{i+1} = f(f_i * w_{i+1})$。

BP算法

基于梯度下降策略，以目标的负梯度方向对参数进行调整，参数的更新为$w + \Delta w \to w$，给定学习率$\alpha$，得出$\Delta w = -\alpha \frac{\alpha Loss}{\alpha w}$。如果要更新第二隐藏层的权值信息，根据链式求导法则，更新梯度信息：
$$
\Delta w_2 = \frac{\partial Loss} {\partial w_2} = \frac{\partial Loss}{\partial f4} \frac{\partial f3}{\partial f2} \frac{\partial f3}{\partial w2}
$$
很容易看出来$\frac{\partial f2}{\partial w2} = \frac{\partial f2}{\partial (f1 * w2)}f1$，即第二层隐藏层的输入。

所以说，$\frac{\partial f4}{\partial w2} × w4$就是对激活函数进行求导。如果此部分大于1,那么层数增多的时候，最终的求出的梯度更新将以指数的形式增加，即发生梯度爆炸，如果此部分小于1,那么随着层数增多，**最终的求出的梯度更新将以指数形式衰减**，即发生了梯度消失。

#### 2. 激活函数角度

其实也注意到了，上文中提到计算权值更新信息的时候需要计算前层偏导信息，因此如果激活函数选择不合适，比如使用sigmoid，梯度消失就会很明显了，原因看下图，左图是sigmoid的损失函数图，右边是其导数的图像，如果使用sigmoid作为损失函数，其梯度是不可能超过0.25的，这样经过链式求导之后，很容易发生梯度消失，sigmoid函数数学表达式为$sigmoid(x) = \frac{1}{1 + e^{-X}}$

![](/home/xxfs/repository/blog/source/_posts/DeepLearning/v2-b35406c5d4f3cf2f6cceb3bbe63a8de1_720w.webp)

![](/home/xxfs/repository/blog/source/_posts/DeepLearning/v2-f7322895a6b2883a111284d42608ac82_720w.webp)

同理，tanh作为激活函数，它的导数图如下，可以看出，tanh比sigmoid要好一些，但是它的导数仍然是小于1的。tanh数学表达为：
$$
tanh(x) = \frac{e^x - e^{-x} }{e^x + e^{-x} }
$$

#### 第三部分：梯度消失、爆炸的解决方案

**2.1  方案一-预训练加微调**

此方法来自Hinton在2006年发表的一篇论文，Hinton为了解决梯度的问题，提出采取无监督逐层训练方法，其基本思想是每次训练一层隐节点，训练时将上一层节点的输出作为输入，而本层隐节点的输出作为下一层隐节点的输入，此过程就是逐层"预训练"；在预训练之后，在对整个网络进行”微调“。Hinton在训练深度信念网络（Deep Belief Networks中，使用了这个方法，在各层预训练完成之后，再利用BP算法对整个网络进行训练）。此思想是寻找全局最优解，此方法有一定的好处，但是目前应用的不是很多了。

**2.2 方案2-梯度剪切、正则**

梯度剪切这个方案主要是针对梯度爆炸提出的，其思想是设置一个梯度剪切的阈值，然后更新梯度的时候，如果梯度超过这个阈值，那么就将其强制限制在这个范围内。这可以防止梯度爆炸。

另外一种解决梯度爆炸的手段是采用权重正则化（weithts regularization）比较常见的是 $L1$ 正则，和 $L2$ 正则，在各个深度框架中都有相应的API可以使用正则化，若搭建网络的时候已经设置了正则化参数，则调用以下代码可以直接计算出正则损失：

**2.3 方案3-relu、leakrelu、elu等激活函数**

如果激活函数导数为1,那么就不存在梯度消失爆炸的问题了，每层的网络都可以得到相同的更新梯度，relu就这样应运而生。先看一下relu的数学表达式：
$$
Relu(x) = max(x,0)= 
\left\{  
             \begin{array}{**lr**}
             0,x<0
             \\
             x,x>0
              \end{array}  
\right.
$$
对relu函数求导可以得到，relu函数的导数在正数部分是恒等于1的，因此在深度网络中使用relu激活函数就不会导致梯度消失和爆炸问题。

relu的主要贡献在于：

1. 解决了梯度消失、梯度爆炸问题
2. 计算方便、计算速度快
3. 加速了网络的计算

relu的缺点：

1. 由于负数部分恒为0,会导致一些神经元无法激活（可通过设置小学习率部分来解决）
2. 输出不是以0为中心的



**leakrelu**

leakrelu就是为了解决relu和0区间带来的影响，其数学表达为：leakrelu  = max(k*x,x)其中k是leak系数，一般选择0.01或0.02,或者通过学习而来

![](/home/xxfs/repository/blog/source/_posts/DeepLearning/v2-368d66932489e813fa6b074b5c5794c0_720w.webp)

leakrelu解决了0区间带来的影响，其数学表达式为：
$$
\left\{  
             \begin{array}{**lr**}
             x, \qquad if \quad x>0
             \\
             \alpha (e^x-1), \quad otherwize
              \end{array}  
\right.
$$
其函数的导数为：



![](/home/xxfs/repository/blog/source/_posts/DeepLearning/v2-f31157be664bda8b0d23356775286572_720w.webp)

**2.4 解决方案4-batchnorm**

Batchnorm具有加速网络收敛速度，提升训练稳定性的效果，Batchnorm本质上是解决反向传播过程的梯度问题。batchnorm全名是batch normalization，简称BN，即批规范化操作将输出信号x规范化保证网络的稳定性。

