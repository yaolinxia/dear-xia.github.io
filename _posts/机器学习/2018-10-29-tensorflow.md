---
layout: post
title: "Tensorflow"
tag: 机器学习
---

## 一、 参数介绍

​	梯度下降法是迭代的，也就是说我们需要多次计算结果，最终求得最优解。梯度下降的迭代质量有助于使输出结果尽可能拟合训练数据。在训练模型时，如果训练数据过多，无法一次性将所有数据送入计算，那么我们就会遇到epoch，batchsize，iterations这些概念。为了克服数据量多的问题，我们会选择将数据分成几个部分，即**batch**，进行训练，从而使得每个批次的数据量是可以负载的。将这些batch的数据逐一送入计算训练，更新神经网络的权值，使得网络收敛。

- Epoch

  一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程。由于一个epoch常常太大，计算机无法负荷，我们会将它分成几个较小的batches。

  在训练时，将所有数据迭代训练一次是不够的，需要反复多次才能拟合收敛。在实际训练时，我们将所有数据分成几个batch，每次送入一部分数据，梯度下降本身就是一个迭代过程，所以单个epoch更新权重是不够的。

- Batch Size

  所谓Batch就是每次送入网络中训练的一部分数据，而Batch Size就是每个batch中训练样本的数量.

- Iterations

  所谓iterations就是完成一次epoch所需的batch个数。batch numbers就是iterations。

  简单一句话说就是，我们有2000个数据，分成4个batch，那么batch size就是500。运行所有的数据进行训练，完成1个epoch，需要进行4次iterations。



## 二、 相关函数



## 参考网址

- <https://www.jianshu.com/p/e5076a56946c>