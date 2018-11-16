---
layout: post
title: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"
tag: 文献阅读
---

### 问题备注

基于图像的序列识别；

自然语言场景文本识别；







### 方法

#### 优点

提出了一种将特征提取，序列建模和转录整合到统一框架中的新型神经网络架构。与以前的场景文本识别系统相比所提出的架构具有四个不同的特性：

- （1）与大多数现有的组件需要单独训练和协调的算法相比，它是端对端训练的。
- （2）它自然地处理任意长度的序列，不涉及字符分割或水平尺度归一化。
- （3）它不仅限于任何预定义的词汇，并且在无词典和基于词典的场景文本识别任务中都取得了显著的表现。
- （4）它产生了一个有效而小得多的模型，这对于现实世界的应用场景更为实用。

> 端到端：分词、词性标注、句法分析、语义分析等多个独立步骤，每个步骤是一个独立的任务，其结果的好坏会影响到下一步骤，从而影响整个训练的结果，这是非端到端的。端到端则相反。

在包括IIIT-5K，Street View Text和ICDAR数据集在内的标准基准数据集上的实验证明了提出的算法比现有技术的更有优势。此外，提出的算法在基于图像的音乐得分识别任务中表现良好，这显然证实了它的泛化性。

CRNN与传统神经网络模型相比具有一些独特的优点：

- 1）可以直接从序列标签（例如单词）学习，不需要详细的标注（例如字符）；
- 2）直接从图像数据学习信息表示时具有与DCNN相同的性质，既不需要手工特征也不需要预处理步骤，包括二值化/分割，组件定位等；
- 3）具有与RNN相同的性质，能够产生一系列标签；
- 4）对类序列对象的长度无约束，只需要在训练阶段和测试阶段对高度进行归一化；
- 5）与现有技术相比，它在场景文本（字识别）上获得更好或更具竞争力的表现[23,8]。
- 6）它比标准DCNN模型包含的参数要少得多，占用更少的存储空间。

#### 具体方法

提出了一种卷积循环神经网络（CRNN）

是DCNN+RNN结合





### 所用模型

![Figure 1](http://upload-images.jianshu.io/upload_images/3232548-7e18716d1bae7aeb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 1) 卷积层，从输入图像中提取特征序列；
- 2) 循环层，预测每一帧的标签分布；
- 3) 转录层，将每一帧的预测变为最终的标签序列。
- 通过不同网络结构的组合，可以只使用一个损失函数，进行联合训练。

#### 特征序列提取（卷积层）

convolutional layers, which extract a feature sequence from the input image 

> CNN中的卷积层和最大池化层

在CRNN模型中，通过采用标准CNN模型（去除全连接层）中的卷积层和最大池化层来构造卷积层的组件。

这样的组件用于从输入图像中提取序列特征表示。在进入网络之前，所有的**图像需要缩放到相同的高度**。

然后从卷积层组件产生的特征图中提取特征向量序列，这些特征向量序列作为循环层的输入。

具体地，特征序列的每一个特征向量在特征图上按列从左到右生成。**这意味着第i个特征向量是所有特征图第i列的连接。在我们的设置中每列的宽度固定为单个像素。**

As the layers of convolution, max-pooling, and elementwise activation function operate on local regions, they are translation invariant.  由于卷积层，最大池化层和元素激活函数在局部区域上执行，因此它们是平移不变的。因此，特征图的每列对应于原始图像的一个矩形区域（称为感受野）

- 对于每一个输入的图片，自动提取出特征序列

![Figure 2](http://upload-images.jianshu.io/upload_images/3232548-064909c42851b28b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 感受野。提取的特征序列中的每一个向量关联输入图像的一个感受野，可认为是该区域的特征向量。

Being robust, rich and trainable, deep convolutional features have been widely adopted for different kinds of visual recognition tasks 由于强大，丰富和可训练，深度卷积特征已被广泛用于不同类型的视觉识别任务。

Some previous approaches have employed CNN to learn a robust representation for sequence-like objects such as scene text一些以前的方法已经聘请CNN学习强大的代表性类似序列的对象，例如场景文本。

However,these approaches usually extract holistic representation of the whole image by CNN, then the local deep features are collected for recognizing each component of a sequencelike object. 然而，这些方法通常提取整体表示CNN的整个形象，然后是当地的深层功能收集用于识别序列式对象的每个组件。

Since CNN requires the input images to be scaled to a fixed size in order to satisfy with its fixed input dimension, it is not appropriate for sequence-like objects due to their large length variation.  由于CNN需要输入图像缩放到固定大小以满足其固定输入维度，它不适合类序列对象由于它们的长度变化很大。

In CRNN, we convey deep features into sequential representations in order to be invariant to the length variation of sequence-like objects. 在CRNN，我们传达将特征深入到顺序表示中对于类序列对象的长度变化不变。

#### 



#### 循环层（序列标注）

recurrent layers, which predict a label distribution for each frame 对于从卷积层输出的每个特征序列的架构，自动进行一个预测。

A deep bidirectional Recurrent Neural Network is built
on the top of the convolutional layers, as the recurrent layers.  顶部是一个深度双向神经网络，作为循环层

预测特征序列中的每一个帧的标签分布

优点有一下三个方面：

- RNN has a strong capability of capturing contextual information within a sequence. 
- RNN can back-propagates error differentials to its input RNN 可以将误差差值反向传播到其输入，即卷积层，从而允许我们在统一的网络中共同训练循环层和卷积层。
- RNN is able to operate on sequences of arbitrary lengths, traversing from starts to ends. RNN能够对任意序列进行操作长度，从头到尾遍历

![Figure 3](http://upload-images.jianshu.io/upload_images/3232548-bb7ebc9a9bbc1c0c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图。(a) 基本的LSTM单元的结构。LSTM包括单元模块和三个门，即输入门，输出门和遗忘门。（b）我们论文中使用的深度双向LSTM结构。合并前向（从左到右）和后向（从右到左）LSTM的结果到双向LSTM中。在深度双向LSTM中堆叠多个双向LSTM结果。

#### 转移层

transcription layer, which translates the per-frame predictions into the final label sequence. 

位于CRNN的顶部，对于循环层的预测结果，再进行一个转换，经历一个标签序列。

Transcription is the process of converting the per-frame predictions made by RNN into a label sequence. 转录是将RNN所做的每帧预测转换成标签序列的过程。

Mathematically, transcription is to find the label sequence with
the highest probability conditioned on the per-frame predictions.  数学上，转录是根据每帧预测找到具有最高概率的标签序列。

In practice, there exists two modes of transcription, namely the lexicon-free and lexicon-based transcriptions.  存在两种转录模式，即无词典转录和基于词典的转录。

A lexicon is a set of label sequences that prediction
is constraint to, e.g. a spell checking dictionary. In lexiconfree mode, predictions are made without any lexicon. In
lexicon-based mode, predictions are made by choosing the
label sequence that has the highest probability. 词典是一组标签序列，预测受拼写检查字典约束。在无词典模式中，预测时没有任何词典。在基于词典的模式中，通过选择具有最高概率的标签序列进行预测。

##### 标签序列的概率

![](https://ws1.sinaimg.cn/large/e93305edgy1fx9yl8pvvej20jf0cwwgn.jpg)



### 实验





### 结论





### 启发





### 参考文献







