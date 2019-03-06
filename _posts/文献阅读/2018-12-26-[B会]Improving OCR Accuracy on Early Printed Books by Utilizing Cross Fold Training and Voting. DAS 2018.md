---
layout: post
title: "[B会]Improving OCR Accuracy on Early Printed Books by Utilizing Cross Fold Training and Voting. DAS 2018 "
tag: 文献阅读
---

[Christian Reul](https://dblp.uni-trier.de/pers/hd/r/Reul:Christian), [Uwe Springmann](https://dblp.uni-trier.de/pers/hd/s/Springmann:Uwe), [Christoph Wick](https://dblp.uni-trier.de/pers/hd/w/Wick:Christoph), [Frank Puppe](https://dblp.uni-trier.de/pers/hd/p/Puppe:Frank):
Improving OCR Accuracy on Early Printed Books by Utilizing Cross Fold Training and Voting. [DAS 2018](https://dblp.uni-trier.de/db/conf/das/das2018.html#ReulSWP18): 423-428

### 问题备注，摘要

- 利用交叉训练和投票，改善在早期打印书本上的OCR准确率
- 执行几个训练过程，每个训练过程，产生一个特定的OCR模型
- 通过这些模型产生出的OCR文本，再经过投票，决定最终的输出（通过可识别的字符，他们的选择，每个字符的信息值）
- 在三个早期打印书本上的实验，通过将误差减少50%或者更多，该方法的性能，能够明显优于标准方法

### 方法

- our method shows considerable differences compared to the work presented above. not only is it applicable to some of the earliest printed books, but it also works with only a single open source ocr engine. furthermore, it can be easily adapted to practically any given book using even a small amount of gt without the need for excessive data to train on (60 to 150 lines of gt corresponding to just a few pages will suffice for most cases)

  与上述工作相比，我们的方法显示出相当大的差异。它不仅适用于一些最早的印刷书籍，而且只适用于单一的开源引擎。此外，它可以很容易地适应几乎任何特定的书籍，即使使用少量的GT，而不需要对过多的数据进行培训(相当于几页的60到150行GT就足以满足大多数情况)。





### 所用模型





### 实验

#### **数据**

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/微信截图_20190305161244.png)





### 结论





### 启发





### 参考文献







