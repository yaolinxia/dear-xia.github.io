---
layout: post
title: "Statistical learning for OCR error correction"
tag: 文献阅读
---

[Jie Mei](https://dblp.uni-trier.de/pers/hd/m/Mei:Jie), [Aminul Islam](https://dblp.uni-trier.de/pers/hd/i/Islam:Aminul), [Abidalrahman Moh'd](https://dblp.uni-trier.de/pers/hd/m/Moh=d:Abidalrahman), [Yajing Wu](https://dblp.uni-trier.de/pers/hd/w/Wu:Yajing), [Evangelos E. Milios](https://dblp.uni-trier.de/pers/hd/m/Milios:Evangelos_E=):
Statistical learning for OCR error correction. [Inf. Process. Manage. 54(6)](https://dblp.uni-trier.de/db/journals/ipm/ipm54.html#MeiIMWM18): 874-887 (2018)



### 问题备注

- OCR后处理

- OCR矫正



### 方法

- 使用丰富的语料库，结合丰富的语义特征集合
- 通过互相依存的学习管道，模型可以产生连续的错误检测和候选矫正的建议。continuously refines the error detection and suggestion of candidate corrections.不断完善误差检测和候选修正的建议。





### 所用模型

#### 标记tokenization

- 实验姑姑n-gram标记，并且包含额外的规则，处理特殊的标记

#### 检测特征集

- 负责识别错误标记
- 从以下四种类型的特征中推断标记的正确性
  - 单词的有效性
  - 字符是否存在
  - 准确的语境连贯；一个正确的单词，应该在它的上下文中是连贯的。
  - 近似语境连贯

#### 错误分类

- 给定一个标记好的文本，分类器必须能够分类出OCR错误的情况，考虑到错误的单词都不会在错误的句子中进行处理，所以，最初该分类器的目标是降低

#### 候选集选择

- 为了在候选集中包含负责的错误模式，我们采纳了两种选择的不同方法

#### 候选特征集合

- 呈现了每一个候选的集合，总结了6种特征类型

#### 候选集修剪





#### 候选置信度预测   

#### 错误重复选择





### 数据

- 网络规模的语料库



### 实验

- 实验基于基于一个历史的生物书有着复杂的错误模式



### 结论

- 在自动模式下，我们的模型由于各种基线方法， 并且再涉及最少的用户交互时，显示出更大的优势。



### 启发





### 参考文献







