---
layout: post
title: "Enhancing RNN Based OCR by Transductive Transfer Learning From Text to Images"
tag: 文献阅读
---

[Yang He](https://dblp.uni-trier.de/pers/hd/h/He_0003:Yang), [Jingling Yuan](https://dblp.uni-trier.de/pers/hd/y/Yuan:Jingling), [Lin Li](https://dblp.uni-trier.de/pers/hd/l/Li:Lin):
Enhancing RNN Based OCR by Transductive Transfer Learning From Text to Images. [AAAI 2018](https://dblp.uni-trier.de/db/conf/aaai/aaai2018.html#HeYL18): 8083-8084

基于文本到图像的再加工学习

### 摘要

- this paper presents a novel approach for optical character  recognition (ocr) on acceleration and to avoid underfitting  by text.

  本文提出了一种新的基于加速和避免文本重复的光学字符识别方法。

- previously proposed ocr models typically take  much time in the training phase and require large amount of  labelled data to avoid underfitting.

  以前提出的模型在训练阶段通常需要很长的时间，并且需要大量的标签数据来避免。

- in contrast, our method  does not require such condition. this is a challenging task  related to transferring the character sequential relationship  from text to ocr.

  相反，我们的方法不需要这样的条件。这是一项具有挑战性的任务，涉及到将字符顺序关系从文本转移到文本。

- we build a model based on transductive  transfer learning to achieve domain adaptation from text to  image.

  我们建立了一种基于转导转移学习的模型，以实现从文本到图像的域自适应。

- we thoroughly evaluate our approach on different  datasets, including a general one and a relatively small one.

  我们从不同的角度对我们的方法进行了彻底的评估，包括一般的和相对较小的。

- we also compare the performance of our model with the  general ocr model on different circumstances. we show  that (1) our approach accelerates the training phase 20-30%  on time cost; and (2) our approach can avoid underfitting  while model is trained on a small dataset

  我们还将我们的模型与一般OCR模型在不同情况下的性能进行比较。我们展示：（1）我们的方法加快了培训阶段20-30%的时间成本；（2）我们的方法可以避免模型在小数据集中接受培训时的拟合不足



### Introduction

- as a major application of pattern recognition and machine  learning, optical character recognition (ocr) is widely  used for converting text from visual documents into digitally text to facilitate document management for search and  information retrieval. in this work, we concentrate on a  basic sequence-to-sequence ocr model (breuel et al.  2013), called dl model for short, aiming at designing a  method that generalizes well to different model structures.  although existing advanced approaches about dl model  has been incessantly come up with, the gist of these models  remain constant, because additional networks just change  the size of data dl model handles.

  光学字符识别作为模式识别和机器学习的一个重要应用，被广泛应用于将文本从视觉文档转换为文本，以便于检索和检索的文档管理。在本文中，我们主要研究了一个基本的序列到序列模型。(简称2013)，旨在设计一种能够很好地适应不同模型结构的方法。虽然已有的先进的模型方法已经提出，但是这些模型仍然是不变的，因为额外的网络只是改变数据模型处理的大小。

- in this model, there is a typical phenomenon in training.  before the accuracy of ocr model rapidly increases, there  lies a period, we called it the latent period, at the beginning  of training that model’s accuracy stays nearly constant,  occupying much time.

  在该模型中，存在一个典型的训练现象。在模型精度迅速提高之前，存在一个时期，我们称之为潜伏期，在训练开始时，模型的精度几乎保持不变，占用了大量的时间。

- in addition, there lies a passive relationship between the  volume of training data and the length of latent period.  while trained on a huge dataset, the latent period has suppressed by the overwhelming amount of data. with the  decrease of training data, the latent period lags and model's  accuracy wanes. when it comes to a small dataset whose  training data are smaller than its test data, dl model will  fall into underfitting, and its accuracy fluctuates round a  low level.

  此外，训练数据的数量与潜伏期的长短之间存在着被动的关系。在庞大的数据集上进行训练时，潜藏期被海量的数据所抑制。随着训练数据的减少，训练的潜伏期和模型的精度也随之降低。对于训练数据小于其测试数据的小数据集，模型的精度将降到较低的水平。

- to efface two problems mentioned above, we build a  model based on transductive transfer learning, called ttl  model for short. our hypothesis of this idea is that, the  image is a counterpart of corresponding text in high dimensional space, namely image is an embodiment presentation of text in 2-dimension and text is a projection of  image towards a lower dimension. like dimensionality  reduction, no matter what dimension data will be projected  to, the relationship between data remains constant. in ocr,  the relationship is character sequential relationship, an order probability between continuous characters. with transfer learning, the shared relationship can be transferred from  text to images

  针对上述两个问题，我们建立了一个基于转移学习的模型，简称为转移学习模型。我们的假设是，图像是高维空间中对应文本的对应体，即图像是文本在二维上的体现，文本是图像向低维的投影。与降维一样，无论预测到什么维数据，数据之间的关系仍然是不变的。在关联关系中，关系是字符序列关系，是连续字符之间的有序概率。通过迁移学习，共享关系可以从文本传递到图像。

- in this paper, we consider to combine these two problems together, as they are both resulted from the capacity  of dataset. we design a novel approach to eradicate the two  problems mentioned above. our main contributions are:  1) acceleration of training phase.  2) avoiding underfitting in a small dataset.

  在本文中，我们考虑将这两个问题结合在一起，因为它们都是数据集容量的结果。我们设计了一种新的方法来消除上述两个问题。我们的主要贡献是：1)加快培训阶段。2)在小数据集中避免重复。

### 方法

- our approach follows the combination architecture of a  teacher network, matrix extension and a student network.  in the teacher network, given an input sequence of sen￾tences, the corresponding sequence of one-hot encoding is  fed as the input of the teacher network with the output of  probability matrix over characters. then we extend these  probability matrices to the same size of images by the rule  of repeating tensors of matrix for certain times, being fed  enhancing rnn based ocr by transductive  transfer learning from text to images  the thirty-second aaai conference on artificial intelligence (aaai-18) 8083 as the input of the student network.

  该方法采用教师网络、矩阵扩展和学生网络相结合的体系结构。在教师网络中，给定SEN的输入序列，将相应的一次热编码序列作为教师网络的输入，输出字符上的概率矩阵。在此基础上，我们将这些概率矩阵扩展到相同大小的图像上，通过重复矩阵的规则在一定的时间内对图像进行增强，通过将文本学习传递到图像增强的方法，将第三十二届人工智能会议(-18)作为学生网络的输入。






### 所用模型

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/微信截图_20190305153411.png)



### 启发

- 角度新颖，从OCR模型训练的速度提高上，加速了OCR的识别，以及避免了重复



### 参考文献







