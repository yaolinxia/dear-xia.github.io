---
layout: post
title: "PoCoTo - an open source system for efficient interactive postcorrection of OCRed historical texts"
tag: 文献阅读
---

- [Thorsten Vobl](https://dblp.uni-trier.de/pers/hd/v/Vobl:Thorsten), [Annette Gotscharek](https://dblp.uni-trier.de/pers/hd/g/Gotscharek:Annette), [Ulrich Reffle](https://dblp.uni-trier.de/pers/hd/r/Reffle:Ulrich), [Christoph Ringlstetter](https://dblp.uni-trier.de/pers/hd/r/Ringlstetter:Christoph), [Klaus U. Schulz](https://dblp.uni-trier.de/pers/hd/s/Schulz:Klaus_U=):
  **PoCoTo - an open source system for efficient interactive postcorrection of OCRed historical texts.**[DATeCH 2014](https://dblp.uni-trier.de/db/conf/datech/datech2014.html#VoblGRRS14): 57-61

> 下载要钱

# **摘要**

- When applied to historical texts, OCR engines often produce a non-negligible number of OCR errors. For research in the Humanities, text mining and retrieval, the option is important to improve the quality of OCRed historical texts using interactive postcorrection. We describe a system for interactive postcorrection of OCRed historical documents developed in the EU project IMPACT. Various advanced features of the system help to efficiently correct texts. Language technology used in the background takes orthographic variation in historical language into account. Using this knowledge, the tool visualizes possible OCR errors and series of similar possible OCR errors in a given input document. Error series can be corrected in one shot. Practical user tests in three major European libraries have shown that the system considerably reduces the time needed by human correctors to eliminate a certain number of OCR errors. The system has been published as an open source tool under GitHub.

  当应用于历史文本时，OCR引擎通常会产生不可忽略的OCR错误。 对于人文科学，文本挖掘和检索的研究，选项对于使用交互式后期校正来提高OCR历史文本的质量非常重要。 我们描述了在欧盟项目IMPACT中开发的OCR历史文档的交互式后期修正系统。 系统的各种高级功能有助于有效地纠正文本。 背景中使用的语言技术考虑了历史语言中的正交变异。 利用这些知识，该工具可视化给定输入文档中可能的OCR错误和一系列类似的可能OCR错误。 错误系列可以一次性纠正。 三个主要欧洲图书馆的实际用户测试表明，该系统大大减少了人工纠正器消除一定数量的OCR错误所需的时间。 该系统已作为GitHub下的开源工具发布。

- 使用了命名实体识别的方法，和一种形态学的方法，因为中文不会被空格分开
- 使用了最大熵分类器，n-gram语言模型，单词是否在词典中出现