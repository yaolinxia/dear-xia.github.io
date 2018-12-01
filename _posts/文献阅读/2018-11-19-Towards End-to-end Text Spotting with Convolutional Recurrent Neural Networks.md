---
layout: post
title: "Towards End-to-end Text Spotting with Convolutional Recurrent Neural Networks"
tag: 文献阅读
---

### 问题备注
论文提出一种统一的网络结构模型，这种模型可以直接通过一次**前向计算**就可以同时实现对图像中文本定位和识别的任务。这种网络结构可以直接以end-to-end的方式训练。

#### 输入

- 图像
- 图像中文本的bbox 
- 文本是标签信息

#### 优点

- 可以学习到更加丰富的特征信息
- 所需时间更少，因为在文本检测和识别的时候，只需要计算一次图像的特征，这种特征是同时别文本检测和识别所共享的。


### 方法

检测和识别统一到一个模型里面，进行end-to-end训练

#### 优点

- 由于检测和识别是高度相关的，因此将检测和识别统一到一个模型里面，就使得图像的feature可以被共享利用。
- 检测和识别这两种任务可以是互补的，更好的检测结果可以提升识别的准确率，识别的信息也可以被用来精修检测的结果。



### 论文所做贡献

- end-to-end方式训练出来的模型可以学习到更丰富的图像特征，并且这种特征可以被两种不同任务所共享，可以有效的节省时间。
- 论文中提出了一种全新的**region feature**抽取方法。这种feature抽取方法可以很好的兼容文本bbox原始长宽比以及避免图像的扭曲，而且**ROI pooling**可以生成具有不同长度的feature maps。
- 提出了一种类似**课程学习策略**的方法用一种逐渐增加图像复杂性的数据集来训练模型。

### 所用模型

![img](https://img-blog.csdn.net/20170818200734278?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDIzODUyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- 首先用修改后的VGG16网络对图像进行特征提取。
- 用TPN对提取的CNN特征进行**region proposals**的生成。
- 然后用由LSTM组成的**RFE**将region proposals编码成固定长度的表征序列。
- 将fixed-length的representation输入到TDN，计算每个region proposals属于文本区域的置信度与坐标偏移量。
- 然后RFE再计算TDN所提供的bboxes内的fixed-length representation。
- 最后TRN基于检测到的bbox（Bounding Box Offsets边界框偏移）的representation来识别单词。

#### VGG网络

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180117142931666?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjU3MzcxNjk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

由上图所知，VGG一共有五段卷积，每段卷积之后紧接着最大池化层，作者一共实验了6种网络结构。分别是VGG-11，VGG-13,VGG-16,VGG-19，网络的输入是224*224大小的图像，输出是图像分类结果（本文只针对网络在图像分类任务上，图像定位任务上暂不做分析） 
接下来开始对VGG做详细的分析，首先VGG是基于Alexnet网络的，VGG在Alexnet基础上对深度神经网络在深度和宽度上做了更多深入的研究，业界普遍认为，更深的网络具有比浅网络更强的表达能力，更能刻画现实，完成更复杂的任务。 

VGG与Alexnet相比，具有如下改进几点：

- 去掉了LRN层
- 采用更小的卷积核
- 池化核变小

**这样解决的原因**

- 更深的网络意味更多的参数，训练更加困难，使用大卷积核尤其明显
- VGG最后使用了三层全连接层，最终接一个softmax
- VGG是一个优良的特征提取器
- 之所以VGG是一个很好的特征提取器，除了和它的网络结构有关，我认为还和它的训练方式有关系，VGG并不是直接训练完成的，它使用了逐层训练的方法。

#### TPN

![img](https://img-blog.csdn.net/20170818200757969?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMDIzODUyMA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

这种结构是由faster rcnn中的RPN结构改进得到。为了适应文本区域的不同程度的长宽比以及尺度的不同，TPN结构采用了相比于RPN更多的anchors（24个），包括4种尺度（16^2，32^2，64^2，80^2），6种长宽比（1：1，2：1，3：1，5：1，7：1，10：1）.同时使用了两种256维的卷积核(5x3,3x1)，分别抽取feature maps中局部和上下文的信息.这种长方形的filters更加适用于本身具有不同长宽比的单词bbox.


#### 参考网址

- <https://blog.csdn.net/qq_25737169/article/details/79084205>
- <https://blog.csdn.net/errors_in_life/article/details/65950699>
- <https://blog.csdn.net/loveliuzz/article/details/79135546>
- <https://blog.csdn.net/u010238520/article/details/77386939>

### 实验





### 结论





### 启发



### 参考网址

- <https://blog.csdn.net/u010238520/article/details/77386939>

### 参考文献

- 





