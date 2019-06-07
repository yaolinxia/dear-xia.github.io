---
layout: post
title: "ShopSign: a Diverse Scene Text Dataset of Chinese Shop Signs in Street Views"
tag: 文献阅读
---

- Zhang C, Peng G, Tao Y, et al. ShopSign: a Diverse Scene Text Dataset of Chinese Shop Signs in Street Views[J]. arXiv preprint arXiv:1903.10412, 2019.

  shopsign: a diverse scene text dataset of chinese shop signs in street views

  展示牌：一种多样化的中国商店街景招牌场景文本数据集

# **摘要**

- in this paper, we introduce the shopsign dataset, which is a newly developed natural scene text dataset of chinese shop signs in street views

  本文介绍了一种新开发的自然场景文本数据集-中国商店街景标志数据集-ShopsignDataSet。

- although a few scene text datasets are already publicly available (e.g. icdar2015, coco-text)

  虽然一些场景文本数据集已经公开了(例如icda 2015，co-text)

- 但是这些公开的在这些数据集中包含的中文文本和字符的图片很少

- 因此，我们收集并且标注了招牌数据以促进在中文自然场景文本中监测和识别的发展



- 新的数据集有以下三种特性：

  - 大语料库

  - 多样性

  - difficulty: the dataset is very sparse稀疏 and imbalanced失衡. It also includes five categories of hard images (mirror, wooden, deformed, exposed and obscure) 

    mirror, wooden, deformed, exposed and obscure

    木制、变形、暴露和模糊的镜子

  - to illustrate the challenges in shopsign, we run baseline experiments using state-of-the-art scene text detection methods (including ctpn, textboxes++ and east), and crossdataset validation to compare their corresponding performance on the related datasets such as ctw, rctw and icpr 2018 mtwi challenge dataset

    为了说明设计中的挑战，我们使用最先进的场景文本检测方法(包括ctpn、文本框和东)和交叉数据集验证进行基线实验，以比较它们在ctw、rctw和icpr 2018 mtwi挑战数据集上的相应性能。

- baseline 使用经典的文本检测方法，包括CTPN，TextBoxes++，EAST, 并且使用交叉数据集验证，包括CTW, RCTW, ICPR 