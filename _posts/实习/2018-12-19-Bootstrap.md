---
layout: post
title: "Bootstrap"
tag: 实习
---

### 简介

- 总体永远都无法知道，我们知道的只有样本
- 问题就是，如何利用样本
- Bootstrap： 既然样本是抽出来的，就从样本中再抽样
- Bootstrap的一般的抽样方式都是“有放回地全抽”（其实样本量也要视情况而定，不一定非要与原样本量相等），意思就是抽取的Bootstrap样本量与原样本相同，只是在抽样方式上采取有放回地抽，这样的抽样可以进行B次，每次都可以求一个相应的统计量/估计量，最后看看这个统计量的稳定性如何（用方差表示）。
- 以原始数据为基础的模拟抽样统计推断法
- 用于研究一组数据的某统计量的分布特征，特别适用于那些难以用常规方法导出对参数的区间估计、假设检验等问题
- 在原始数据的范围内作有放回的再抽样，样本容量仍为n，原始数据中每个观察单位每次被抽到的概率相等，为1/n。所得样本为Bootstrap样本。

### 详细解释

![](https://ws1.sinaimg.cn/large/e93305edgy1fyc7crfu1rj20g60e7wgm.jpg)

### 步骤

- 通过重采样（有放回的采样），抽取一定数量的新样本
- 基于产生的新样本，计算需要估计的统计量

![](https://ws1.sinaimg.cn/large/e93305edgy1fyc7q156qnj20fx01l0sr.jpg)

- 重复上面步骤n次，每次计算一个统计量a
- 最后估计均值以及方差

![](https://ws1.sinaimg.cn/large/e93305edgy1fyc7sw5y8gj20b905vweo.jpg)

![](https://ws1.sinaimg.cn/large/e93305edgy1fyc7vorthkj20fz087t9f.jpg)





![img](https://pic4.zhimg.com/80/v2-f2b47f4339db21da24d46b1091778a47_hd.png)











### 参考网址

- <https://zhuanlan.zhihu.com/p/24851814>
- 