---
layout: post
title: "A knowledge-based table recognition method for Chinese bank statement images"
tag: 文献阅读
---

[Liang Xu](https://dblp.uni-trier.de/pers/hd/x/Xu:Liang), [Wei Fan](https://dblp.uni-trier.de/pers/hd/f/Fan:Wei), [Jun Sun](https://dblp.uni-trier.de/pers/hd/s/Sun_0004:Jun), [Xin Li](https://dblp.uni-trier.de/pers/hd/l/Li:Xin), [Satoshi Naoi](https://dblp.uni-trier.de/pers/hd/n/Naoi:Satoshi):
A knowledge-based table recognition method for Chinese bank statement images. [ICIP 2016](https://dblp.uni-trier.de/db/conf/icip/icip2016.html#XuFSLN16): 3279-3283 [CCF C会]



### 问题备注

- 对大容量的中国银行的扫描文件进行自动处理





### 传统方法缺陷

- Conventional methods can not well handle the following challenges of this problem: various layout styles, noises, and especially requirement of fast speed for large Chinese character set. （大面板，噪音，速度）

- direct ocr on table image will fail since table cells are semantically interrelated.

  由于表格单元格在语义上是相互关联的，因此表映像上的直接OCR将失败。

### 本方法优点

- This paper proposes a knowledge based table recognition method to meet fast speed requirement with good accuracy 基于知识的图表识别，可以达到高速度高准确率
- Two kinds of knowledge are utilized to accelerate the identification of digit columns and the cell recognition: i) geometric knowledge about column alignment and quasi equal digit width, and ii) semantic knowledge about prior format based on the results from an optical character recognition (OCR) engine of digits.  两种知识用于加速数字列的识别和细胞识别：i）几何知识关于列对齐和准相等数字宽度，以及ii）基于结果的先前格式的语义知识来自数字的光学字符识别（OCR）引擎。

- i) table detector that finds regions corresponding to tables and ii) table structure recognizer that identifies the relationships from the detected table to derive the logical structure.

  (I)表检测器，它查找与表相对应的区域；以及(Ii)表结构识别器，该表结构识别器从检测到的表中识别出关系，以导出

### 方法

![](https://ws1.sinaimg.cn/large/e93305edgy1fxpvrc7id5j20i40b2tae.jpg)

- the focus of our work in this paper is the problem of table structure recognition, which is an indispensable step in table processing system.

  本文的工作重点是表格结构识别问题，这是表格处理系统中不可缺少的一个环节。

- **物理面板分析**：输入预处理图片；文本线提取； 主体区域检测；列分开

- **逻辑面板分析**：目标列的提取；基于列的字符串识别；Postprocessing(后期处理)





### 所用模型





### 实验





### 结论





### 启发





### 参考文献







