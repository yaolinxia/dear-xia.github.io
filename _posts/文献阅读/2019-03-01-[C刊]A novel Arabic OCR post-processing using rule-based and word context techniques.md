---
layout: post
title: "[C刊]Improving OCR Accuracy on Early Printed Books by Utilizing Cross Fold Training and Voting  DAS 2018: 423-428"
tag: 文献阅读
---

[Christian Reul](https://dblp.uni-trier.de/pers/hd/r/Reul:Christian), [Uwe Springmann](https://dblp.uni-trier.de/pers/hd/s/Springmann:Uwe), [Christoph Wick](https://dblp.uni-trier.de/pers/hd/w/Wick:Christoph), [Frank Puppe](https://dblp.uni-trier.de/pers/hd/p/Puppe:Frank):
Improving OCR Accuracy on Early Printed Books by Utilizing Cross Fold Training and Voting. [DAS 2018](https://dblp.uni-trier.de/db/conf/das/das2018.html#ReulSWP18): 423-428

### 问题备注，摘要

- optical character recognition (ocr) is the process of recognizing characters automatically from scanned documents for editing, indexing, searching, and reducing the storage space.

  光学字符识别(OCR)是自动从扫描的文档中识别字符以便进行编辑、索引、搜索和减少存储空间的过程。

- the resulted text from the ocr usually does not match the text in the original document. in order to minimize the number of incorrect words in the obtained text, ocr post-processing approaches can be used. correcting ocr errors is more complicated when we are dealing with the arabic language because of its complexity such as connected letters, different letters may have the same shape, and the same letter may have different forms.

  最终得到的文本通常与原始文档中的文本不匹配。为了最大限度地减少所获得的文本中不正确的字数，可以使用更精确的后处理方法。当我们处理阿拉伯语时，纠正错误是比较复杂的，因为它的复杂性，例如连接字母，不同的字母可能有相同的形状，同一字母可能有不同的形式。

- this paper provides a statistical arabic language model and post-processing techniques based on hybridizing the error model approach with the context approach

  本文提出了一种基于误差模型与上下文方法杂交的统计阿拉伯语模型和后处理技术。

- the proposed model is language independent and non-constrained with the string length. to the best of our knowledge, this is the first end-to-end ocr post-processing model that is applied to the arabic language. in order to train the proposed model, we build arabic ocr context database which contains 9000 images of arabic text. also, the evaluation of the ocr post-processing system results is automated using our novel alignment technique which is called fast automatic hashing text alignment.

  所提出的模型是与字符串长度无关且不受约束的语言。对于我们所知，这是应用于阿拉伯语的第一个端到端OCR后处理模型。为了训练所提出的模型，我们构建了阿拉伯文OCR上下文数据库，它包含9000个阿拉伯文文本的图像。此外，OCR后处理系统结果的评估是使用我们的新的对齐技术自动进行的，该技术被称为快速自动散列文本对齐。

- our experimental results show that the rule-based system improves the word error rate from 24.02% to become 20.26% by using a training data set of 1000 images. on the other hand, after this training, we apply the rule-based system on 500 images as a testing dataset and the word error rate is improved from 14.95% to become 14.53%. the proposed hybrid ocr post-processing system improves the results based on using 1000 training images from a word error rate of 24.02% to become 18.96%. after training the hybrid system, we used 500 images for testing and the results show that the word error rate enhanced from 14.95 to become 14.42. the obtained results show that the proposed hybrid system outperforms the rule-based system.

  实验结果表明，基于规则的系统通过使用1000幅图像的训练数据集，将单词错误率从24.0%提高到了20.26%。另一方面，经过这一训练，我们将基于规则的系统应用于500幅图像作为测试数据集，并将单词错误率从14.95%提高到了14.53%。本文提出的混合后处理系统在使用1000幅训练图像的基础上，将训练图像的误差率从 24.02%提高到了18.96%。在对混合系统进行训练后，我们使用了500幅图像进行测试，结果表明，该系统的词汇误差率由原来的更高到更高。结果表明，所提出的混合系统是基于规则的系统.

### 方法

- in this section, the new arabic ocr database is presented. after that, a new alignment algorithm is proposed, called fast automatic hashing text alignment (fahta). this algorithm is used to align the misrecognized word forms with the corresponding ground truth word forms.

  本节介绍了新的阿拉伯语数据库。在此基础上，提出了一种新的对齐算法-快速自动文本对齐算法。该算法用于对齐对应的地面真实词格式。

- finally, the two arabic ocr post-processors are implemented and compared

  最后,实施并比较了两个阿拉伯文OCR后处理处理器,





### 所用模型

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/微信截图_20190305163701.png)

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/微信截图_20190305164245.png)



### 实验

#### **数据**

- 阿拉伯语

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/微信截图_20190305163518.png)







### 结论





### 启发

- 基于阿拉伯语的第一个端到端的OCR后处理模型
- 统计了阿拉伯语的特征



### 参考文献







