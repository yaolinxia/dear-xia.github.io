---
layout: post
title: "Semi-Supervised Sequence Modeling with Cross-View Training"
tag: 文献阅读
---

- [Kevin Clark](https://dblp.uni-trier.de/pers/hd/c/Clark:Kevin), [Minh-Thang Luong](https://dblp.uni-trier.de/pers/hd/l/Luong:Minh=Thang), [Christopher D. Manning](https://dblp.uni-trier.de/pers/hd/m/Manning:Christopher_D=), [Quoc V. Le](https://dblp.uni-trier.de/pers/hd/l/Le:Quoc_V=):
  **Semi-Supervised Sequence Modeling with Cross-View Training.** [EMNLP 2018](https://dblp.uni-trier.de/db/conf/emnlp/emnlp2018.html#ClarkLML18): 1914-1925

# **摘要**

- unsupervised representation learning algorithms such as word2vec and elmo improve the accuracy of many supervised nlp models, mainly because they can take advantage of large amounts of unlabeled text. however, the supervised models only learn from taskspecific labeled data during the main training phase.

  无监督表示学习算法如Word2vec和Elmo提高了许多有监督的NLP模型的准确性，主要是因为它们可以利用大量的未标记文本。然而，在主训练阶段，监督模型仅从特定任务的标记数据中学习。

- we therefore propose cross-view training (cvt), a semi-supervised learning algorithm that improves the representations of a bi-lstm sentence encoder using a mix of labeled and unlabeled data.

  因此，我们提出了交叉视点训练(CVT)，这是一种半监督学习算法，它使用标记和未标记数据的混合改进了双LSTM语句编码器的表示。

- 