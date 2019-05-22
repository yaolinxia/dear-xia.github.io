---
layout: post
title: "[C刊]Web Knowledge Base Improved OCR Correction for Chinese Business Cards"
tag: 文献阅读
---

[Xiaoping Wang](https://dblp.uni-trier.de/pers/hd/w/Wang:Xiaoping), [Yanghua Xiao](https://dblp.uni-trier.de/pers/hd/x/Xiao:Yanghua), [Wei Wang](https://dblp.uni-trier.de/pers/hd/w/Wang_0009:Wei):
Web Knowledge Base Improved OCR Correction for Chinese Business Cards. [WAIM 2015](https://dblp.uni-trier.de/db/conf/waim/waim2015.html#WangXW15): 573-576 【**C会**】

### 问题备注，摘要

- knowledge base was applied to ocr correcting system from the perspective of linked knowledge.

  从关联知识的角度将知识库应用于OCR校正系统。

- a pipelined method integrating selectivity-aware pre-filtering, text-level and image-level comparison was explored to identify the best candidate with better efficiency and accuracy.

  探讨了一种结合选择性感知预滤波、文本级和图像级比较的流水线方法，以更好的效率和准确性确定最佳候选。

- for more reliable comparison of company, the weighted coefficients derived from wikipedia were applied to distinguish the different importance.

  为了更可靠地比较公司，应用维基百科的加权系数来区分不同的重要性。

- moreover, traditional levenshtein distance was generalized to image-based levenshtein measure to better distinguish strings with similar text similarity.

  此外，将传统的levenshtein距离推广到基于图像的levenshtein测度，以更好地区分文本相似的字符串。

### **Introduction**

- the performance of optical character recognition (ocr) is often impacted by weak illumination, noise, and skew.

  光学字符识别(OCR)的性能往往受到弱光照、噪声和偏斜等因素的影响。

- most of the error correction methods address on natural language processing (nlp) or machine learning techniques.

  大多数纠错方法都是关注于自然语言处理(NLP)或机器学习技术。

- however, they might not perform well for business card.

  然而，他们在名片方面可能表现不佳。

- for instance, it’s difficult to determine the correct address among various candidates valid for models without resorting to additional information.

  例如，如果不使用其他信息，就很难确定对模型有效的各种候选人的正确地址。

- nowadays, there exist diverse knowledge bases (kbs) such as web encyclopedias (e.g. wikipedia1), and point of interest (poi) database. it is plausible that the “linked” knowledge within the kbs be explored to improve the ocr accuracy further.

  目前，已有多种知识库(如网络百科全书1)和兴趣点数据库(POI)。为了进一步提高OCR的准确性，有可能探索KBS内部的“关联”知识。

- the main contributions of this work were: based on our knowledge, it’s the first work for the ocr correction of chinese business cards from the point of view of kb.

  本文的主要贡献是：基于我们的知识，这是第一份从kb的角度对中国名片进行OCR校正的工作。

- a pipelined framework of correction method including selectivityaware **pre-filtering, text-level correction** and image-**level correction** was proposed.

  提出了一种流水线结构的校正方法，包括选择感知预滤波、文本级校正和图像级校正.

- selectivity-aware pre-filtering was tested to exclude irrelevant records quickly while retain the possible candidates.

  选择性感知的预过滤测试，以迅速排除无关的记录，同时保留可能的候选人。

- to be flexible for address integrity, a robust similarity measuring method integrating dynamic time warping (dtw) with jaccard/levenshtein measure was developed.

  为了灵活地实现地址完整性，提出了一种结合动态时间翘曲(DTW)和Jaccard/levenshtein测度的鲁棒相似性度量方法。

- to make the company comparison more reliable, a strategy of weighting importance based on wikipedia kb was conducted.

  为了使公司比较更加可靠，采用了一种基于维基百科kb的加权重要性策略。

- to distinguish the candidates with similar text similarity, traditional levenshtein distance was generalized to image-based levenshtein measure.

  为了区分文本相似性相似的候选对象，将传统的levenshtein距离推广到基于图像的levenshtein测度。

### **方法**

- we focused on the correction of company-address pair. the biggest challenge was similarity computation to handle the format diversity or importance imbalance. related key techniques were emphasized below.

  我们把重点放在公司地址对的修正上。最大的挑战是相似性计算，以处理格式多样性或重要性不平衡。下文强调了相关的关键技术。

- to compute the similarity reasonably for addresses with different integrity, we decomposed address by its hierarchy at first.

  为了合理地计算具有不同完整性的地址的相似度，我们首先根据其层次结构对地址进行分解。

- then, dtw and jaccard/levenshtein measure were performed.

  DTW和Jaccard/levenshtein方法被提出来。

- since different part within company name played different roles in comparison, part of speech tagging was applied for segmentation and feasible weights were then assigned using inverse document frequency(idf) derived from wikipedia kb.

  j经对比之后发现，公司名称中的不同部分扮演着不同的角色，词性标注的方法被采用，然后利用维基百科kb中的反向文档频率(下手频率)分配可行的权重。

- the details of weight computing is illustrated in formula.1. an example was shown in table.1.

  重量计算的细节在公式A.1中说明。表1中显示了一个例子。

- text-level/image-level correction and pipelined framework

  文本级/图像级校正和流水线框架

- although the incorrectly recognized characters are different from the correct ones in text, they might be similar in image.

  虽然不正确识别的字符与文本中的正确字符不同，但它们在图像上可能是相似的。

- hence, a pipelined correction method using both text similarity and image similarity was applied.

  因此，本文提出了一种基于文本相似度和图像相似度的流水线校正方法。

- for text-level comparison, dtw combined with jaccard/levenshtein measure was applied.

  文本级比较采用DTW与Jaccard/levenshtein相结合的方法。

- for image-level comparison, 2d discrete cosine transform features and intersecting features were applied.

  图像级比较采用二维离散余弦变换特征和相交特征。

- 