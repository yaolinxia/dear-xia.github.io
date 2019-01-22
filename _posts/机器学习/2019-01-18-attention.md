---
layout: post
title: "attention"
tag: 机器学习
---

# 简介

- 用于提升基于RNN（LSTM或GRU）的Encoder + Decoder模型的效果的的机制（Mechanism）

- 一种资源分配模型，在某个特定时刻，你的注意力总是集中在画面中的某个焦点部分，而对其它部分视而不见。

# 适用领域

机器翻译、语音识别、图像标注（Image Caption）等很多领域

# 优点

- Attention给模型赋予了区分辨别的能力

- 在机器翻译、语音识别应用中，为句子中的每个词赋予不同的权重，使神经网络模型的学习变得更加灵活（soft），同时Attention本身可以做为一种对齐关系，解释翻译输入/输出句子之间的对齐关系，解释模型到底学到了什么知识，为我们打开深度学习的黑箱，提供了一个窗口，如图1所示。
- 与人类对外界食物的观察机制类似，当人类观察外界事物的时候，一般不会把事物当成一个整体去看，往往倾向于根据需要选择性的去获取被观察事物的某些重要部分。

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\v2-016f8210eb05fb1b32bc75f4131a7dbf_hd.jpg)

# 原理

## Encoder-Decoder

- 可以把它看作适合处理由一个句子（或篇章）生成另外一个句子（或篇章）的通用处理模型。对于句子对<X,Y>，我们的目标是给定输入句子X，期待通过Encoder-Decoder框架来生成目标句子Y。X和Y可以是同一种语言，也可以是两种不同的语言。而X和Y分别由各自的单词序列构成：![](E:\yaolinxia\workspace\practice\practice\images\文献阅读\20160120181636077.jpg)

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\v2-2a5ba93492e047b60f4ebc73f2862fae_hd.jpg)



![20160120181545780](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\20160120181545780.jpg)

- 把一个变成的输入序列x1，x2，x3....xt编码成一个固定长度隐向量（背景向量，或上下文向量context）c，c有两个作用：1、做为初始向量初始化Decoder的模型，做为decoder模型预测y1的初始向量。2、做为背景向量，指导y序列中每一个step的y的产出。Decoder主要基于背景向量c和上一步的输出yt-1解码得到该时刻t的输出yt，直到碰到结束标志（<EOS>）为止。

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\微信截图_20190118164217.png)

> X: 句子
>
> X={x1,x2,...,x3}
>
> X --> C(背景向量、语义编码、上下文向量)

- 句子中任意单词对生成目标单词yi， 影响力都是一样的， 没有任何区别
- 所以谷歌提出S2S时，发现输入句子逆序输入做翻译效果更好

### 常用模型

- CNN/RNN/BiRNN/GRU/LSTM/Deep LSTM

### 应用场景

- Encoder-Decoder是个创新游戏大杀器，一方面如上所述，可以搞各种不同的模型组合，另外一方面它的应用场景多得不得了，比如对于机器翻译来说，<X,Y>就是对应不同语言的句子，比如X是英语句子，Y是对应的中文句子翻译。再比如对于文本摘要来说，X就是一篇文章，Y就是对应的摘要；再比如对于对话机器人来说，X就是某人的一句话，Y就是对话机器人的应答；再比如……总之，太多了。哎，那位施主，听老衲的话，赶紧从天台下来吧，无数创新在等着你发掘呢。

## attention

- 在encoder-decoder基础上，引入AM模型
- 给定一个概率分布值`（Tom,0.3）(Chase,0.2)(Jerry,0.5)`
- 每个单词的概率代表了在当前单词下，注意力分配给不同英文单词的注意力大小。对于正确翻译目标单词是有帮助的，因为引入了新的信息。
- 每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。
- 这意味着在生成每个单词Yi的时候，原先都是相同的中间语义表示C会替换成根据当前生成单词而不断变化的Ci。

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\微信截图_20190118171943.png)

- 相应的生成目标句子单词的过程如下：

  ![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\微信截图_20190118172058.png)



- 而每个Ci可能对应着不同的源语句子单词的注意力分配概率分布，比如对于上面的英汉翻译来说，其对应的信息可能如下：

  ![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\微信截图_20190118172227.png)

  > f2: 代表Encoder对于输入英文单词的某种变换函数（如果Encoder用的是RNN, f2的结果就是指某个时刻输入xi后隐层节点的状态值）
  >
  > g:代表Encoder根据单词的中间表示合成整个句子中间语义表示的变换函数， 一般， g函数就是构成元素的加权求和

### 公式

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\微信截图_20190118173449.png)

- Ci ：Ci中那个i就是上面的“汤姆”、
- Tx：对应上面的3，代表输入句子的长度
- h1=f(“Tom”)，h2=f(“Chase”),h3=f(“Jerry”)
- 注意力模型权值分别为0.6， 0.2， 0.2

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\20160120182034485.png)

### 概率分布值

**存在问题**：如何知道AM模型所需要输入的句子单词注意力概率分布值？

> eg:（Tom,0.6）(Chase,0.2)  (Jerry,0.2）是如何得到的

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\20160120182108205.png)

如上图，非AM模型的Encoder-Decoder框架进行细化，Encoder采用RNN模型，Decoder也采用RNN模型，这是比较常见的一种模型配置

**概率分布值通用计算**：

![](E:\yaolinxia\workspace\yaolinxia.github.io\images\文献阅读\20160120182219236.png)

> 上图对应输入值为Tom时， 对应的输入句子的对齐概率。
>
> 可以用i时刻的隐层节点状态Hi去一一和输入句子中每个单词对应的RNN隐层节点状态hj进行对比，即通过函数F(hj,Hi)来获得目标单词Yi和每个输入单词对应的对齐可能性
>
> F函数在不同论文里可能会采取不同的方法，然后函数F的输出经过Softmax进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值

### 物理含义

- 一般会把AM模型看作时单词对齐模型

- 目标句子生成的每个单词对应输入句子单词的概率分布可以理解为输入句子单词和这个目标生成单词的对齐概率
- 传统的统计机器翻译一般在做的过程中会专门有一个短语对齐的步骤，而注意力模型其实起的是相同的作用
- AM模型理解成影响力模型也是合理的，就是说生成目标单词的时候，输入句子每个单词对于生成这个单词有多大的影响程度。这种想法也是比较好理解AM模型物理意义的一种思维方式

# 参考网址、文献

- 《Sequence to Sequence Learning with Neural Networks》
- 《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》
- <https://blog.csdn.net/xiewenbo/article/details/79382785>