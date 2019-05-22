---
layout: post
title: "[开源期刊C以下]Deep Learning-Aided OCR Techniques for Chinese Uppercase Characters in the Application of Internet of Things"
tag: 文献阅读
---

- [Yue Yin](https://dblp.uni-trier.de/pers/hd/y/Yin:Yue), [Wei Zhang](https://dblp.uni-trier.de/pers/hd/z/Zhang:Wei), [Sheng Hong](https://dblp.uni-trier.de/pers/hd/h/Hong:Sheng), [Jie Yang](https://dblp.uni-trier.de/pers/hd/y/Yang_0027:Jie), [Jian Xiong](https://dblp.uni-trier.de/pers/hd/x/Xiong:Jian), [Guan Gui](https://dblp.uni-trier.de/pers/hd/g/Gui:Guan):
  **Deep Learning-Aided OCR Techniques for Chinese Uppercase Characters in the Application of Internet of Things.**[IEEE Access 7](https://dblp.uni-trier.de/db/journals/access/access7.html#YinZHYXG19): 47043-47049 (2019)

- IEEE Access： 开源期刊，审稿速度快

- **【参考网址：】**<https://www.zhihu.com/question/66432478>

# **摘要**

1. OCR在计算机视觉领域是一个重要的技术， 它可以简单地从物联网的各种图片中获取数字信息（**背景**）

2. 然而已经存在的OCR技术在中文上面的识别较差（**现有问题**）

   however, existing ocr techniques pose a big challenge in the recognition of the chinese uppercase characters due to their poor performance.

   然而，现有的OCR技术由于其性能差，对汉字大写字符的识别提出了很大的挑战。

3. 为了解决这个问题，提出了半自动的深度学习（**解决方法**）

4. first, we generate a database of the chinese uppercase characters to train four neural networks: a convolution neural network (cnn), a visual geometry group, a capsule network, and a residual network.

   首先，我们生成一个中文大写字符库来训练四个神经网络：卷积神经网络(CNN)、视觉几何组、胶囊网络和残差网络。

5. second, the four networks are tested on the generated dataset in terms of accuracy, network weight, and test time.

   其次，在生成的数据集上对这四个网络进行了精度、网络权重和测试时间的测试。

6. finally, in order to reduce test time and save computational resources, we also develop a lightweight cnn method to prune the network weight by 96.5% while reducing accuracy by no more than 1.26%.

   最后，为了减少测试时间并节省计算资源，我们还开发了一种轻量级CNN方法，在不超过1.26%的情况下，将网络权重修剪为96.5%。

# **简介**

- existing ocr techniques perform very well in the recognition of english words as well as arabic numerals. however, the accuracy of these techniques is not high for recognizing chinese characters due to different language families [9].

  现有的OCR技术在识别英语单词和阿拉伯数字方面表现很好。然而，由于语言家族的不同，这些技术对于汉字识别的准确性并不高[9]。