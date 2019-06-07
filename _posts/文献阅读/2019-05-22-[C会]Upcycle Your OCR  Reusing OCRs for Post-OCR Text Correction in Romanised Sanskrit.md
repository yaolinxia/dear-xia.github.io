---
layout: post
title: "Upcycle Your OCR: Reusing OCRs for Post-OCR Text Correction in Romanised Sanskrit"
tag: 文献阅读
---

- [Amrith Krishna](https://dblp.uni-trier.de/pers/hd/k/Krishna:Amrith), [Bodhisattwa Prasad Majumder](https://dblp.uni-trier.de/pers/hd/m/Majumder:Bodhisattwa_Prasad), [Rajesh Shreedhar Bhat](https://dblp.uni-trier.de/pers/hd/b/Bhat:Rajesh_Shreedhar), [Pawan Goyal](https://dblp.uni-trier.de/pers/hd/g/Goyal:Pawan):
  **Upcycle Your OCR: Reusing OCRs for Post-OCR Text Correction in Romanised Sanskrit.** [CoNLL 2018](https://dblp.uni-trier.de/db/conf/conll/conll2018.html#KrishnaMBG18): 345-355 【C会】

- upcycle

  vt.
  升级改造。即; 用创新的方式将某样东西改造以让发挥新的功效。与recycle（循环利用）不同; upcycle不会对原材料进行任何再处理; 只是换个方式利用它们。;

- reusing ocrs for post-ocr text correction in romanised sanskrit

  在罗马梵文中重用OCR后的文本校正

# **摘要**

- we propose a post-ocr text correction approach for digitising texts in romanised sanskrit.

  我们提出了一种后OCR文本校正方法，以数字化的文本在浪漫梵语。

- owing to the lack of resources our approach uses ocr models trained for other languages written in roman.

  由于缺乏资源，我们的方法使用了在罗马编写的其他语言培训的OCR模型。

- currently, there exists no dataset available for romanised sanskrit ocr.

  目前，没有可用于罗马化的梵文OCR的数据集。

- so, we bootstrap a dataset of 430 images, scanned in two different settings and their corresponding ground truth.

  因此，我们引导430个图像的数据集，扫描在两个不同的设置和他们相应的地面真相。

- for training, we synthetically generate training images for both the settings.

  对于训练，我们为这两个设置综合地生成训练图像。

- we find that the use of copying mechanism (gu et al., 2016) yields a percentage increase of 7.69 in character recognition rate (crr) than the current state of the art model in solving monotone sequence-tosequence tasks (schnober et al., 2016).

  我们发现，使用复制机制(Gu等，2016)在字符识别率(CRR)中产生比现有技术模型在求解单调序列到序列任务中的当前状态的百分比增加7.69(Schnober等，2016)。

## **资源**

- https://github.com/majumderb/sanskrit-ocr 

# **简介**

- sanskrit used to be the ‘lingua franca’ for the scientific and philosophical discourse in ancient india with literature that **spans** more than 3 millennia.

  梵语曾是古代印度的科学哲学话语的“语言文化”，其文学**跨度**超过3年。

- sanskrit primarily had an oral tradition, and the script used for writing sanskrit varied widely across the time spans and regions.

  梵文主要有口述传统，用于书写梵文的剧本在时间跨度和地区上差异很大。

- sanskrit primarily had an oral tradition, and the script used for writing sanskrit varied widely across the time spans and regions.

  梵文主要有口述传统，用于书写梵文的剧本在时间跨度和地区上差异很大。

- with standardisation of romanisation using iast in 1894 (monier-williams, 1899), printing in sanskrit was extended to roman scripts as well.

  随着浪漫主义的标准化使用在1894年(莫尼尔-威廉姆斯，1899年)，梵文印刷也扩展到罗马文字。

- in this work, we propose a model for postocr text correction for sanskrit written in roman.

  在此工作中，我们提出了一种在罗马字中编写梵文的posterr文本修正模型。

- post-ocr text correction, which can be seen as a special case of spelling correction (schnober et al., 2016), is the task of correcting errors that tend to appear in the output of the ocr in the process of converting an image to text.

  后OCR文本更正，可以看作是拼写更正的一个特例(schnober等人，2016)，是纠正错误的任务，这些错误往往出现在OCR的输出过程中将图像转换为文本。

- the errors incurred from ocr can be quite high due to numerous factors including typefaces, paper quality, scan quality, etc.

  由于字体、纸张质量、扫描质量等诸多因素，OCR产生的错误可能相当高。

- the text can often be eroded, can contain noises and the paper can be bleached or tainted as well

  文字经常会被腐蚀，可能含有噪音，纸张也可能被漂白或污染。

- figure 1 shows the sample images we have collected for the task. hence it is beneficial to perform a post-processing on the ocr output to obtain an improved text.

  图1显示了为该任务收集的示例映像。因此，对OCR输出执行后处理以获得改进的文本是有益的。

- the word based sequence labelling approaches were further extended to use neural architectures, especially using rnns and its variants such as lstms and grus

  基于单词的序列标记方法进一步扩展到使用神经结构，特别是使用rnns及其变种，如lstms和grus。

- where the authors found that using words within a context window of 5 for a given input word worked particularly well for the post-ocr text correction in sanskrit.

  作者发现，对于梵文中的OCR后文本校正，在上下文窗口5内使用给定输入词特别有效。

## **贡献**

- contrary to what is observed in schnober et al. (2016), an encoder-decoder model, when equipped with copying mechanism (gu et al., 2016), can outperform a traditional sequence labelling model in a monotone sequence labelling task. our model outperforms schnober et al. (2016) in the post-ocr text correction for romanised sanskrit task by 7.69 % in terms of crr

  与Schnober等人所观察到的相反。(2016)具有复制机制的编码器-解码器模型(GU等人，2016)在单调序列标记任务中优于传统的序列标记模型。我们的模型优于schnober等人。(2016)罗曼史梵语任务的后OCR文本更正中，CRR为7.69%

- by making use of digitised sanskrit texts, we generate images as synthetic training data for our models.

  通过使用数字化梵语文本，我们生成图像作为我们的模型的综合训练数据。

- we systematically incorporate various distortions to those images so as to emulate the settings of the original images.

  我们系统地将各种失真结合到这些图像中，以便模拟原始图像的设置。

- through a human judgement experiment, we asked the participants to correct the mistakes from a predicted output from the competing systems. we find that participants were able to correct predictions from our system more frequently and the corrections were done much faster than the crf model by schnober et al. (2016). we observe that predictions from our model score high on acceptability (lau et al., 2015) than other methods as well.

  通过一个人类判断实验，我们要求参与者纠正来自竞争系统的预测输出的错误。我们发现参与者能够更频繁地纠正来自我们系统的预测，并且修正的速度比schnober等人的CRF模型快得多。(2016年)。我们观察到，我们的模型的预测在可接受性上得分很高(刘等人，2015年)，也高于其他方法。

# **模型体系**

- we hypothesise this to be due to lack of enough font styles available in our collection, in spite of using a site with the richest collection of sanskrit fonts

  我们假设这是因为我们的集合中缺乏足够的字体样式，尽管我们使用的是梵语字体最丰富的网站。

# **实验**

## **数据集**





## **Synthetic Generation of training set **





## **Baselines **



## **结果**

### **System performances for various input lengths **





### **Error type analysis  **







