---
layout: post
title: "Multi-Input Attention for Unsupervised OCR Correction"
tag: 文献阅读
---

[David Smith](https://dblp.uni-trier.de/pers/hd/s/Smith:David), [Rui Dong](https://dblp.uni-trier.de/pers/hd/d/Dong:Rui):
Multi-Input Attention for Unsupervised OCR Correction. [ACL (1) 2018](https://dblp.uni-trier.de/db/conf/acl/acl2018-1.html#SmithD18): 2363-2372

### 问题备注

- 无监督的OCR矫正

### 摘要

- we propose a novel approach to ocr post-correction that exploits repeated texts in large corpora both as a source of noisy target outputs for unsupervised training and as a source of evidence when decoding.

  我们提出了一种新的OCR后校正方法，该方法利用大语料库中的重复文本作为无监督训练的噪声目标输出源和解码时的证据来源。

- a sequence-to-sequence model with attention is applied for single-input correction, and a new decoder with multi-input attention averaging is developed to search for consensus among multiple sequences.

  提出了一种基于注意的序列到序列模型进行单输入校正，并设计了一种新的多输入注意平均译码器，以寻找多序列间的一致性。

- we design two ways of training the correction model without human annotation, either training to match noisily observed textual variants or bootstrapping from a uniform error model.

  设计了两种无需人工标注的校正模型训练方法，一种是训练来匹配噪声观测到的文本变体，另一种是从统一的错误模型中进行引导。

- on two corpora of historical newspapers and books, we show that these unsupervised techniques cut the character and word error rates nearly in half on single inputs and, with the addition of multi-input decoding, can rival supervised methods

  在两个历史报刊和书籍的语料库中，我们证明了这些非监督技术在单一输入上将字符和单词错误率降低了近一半，加上多输入解码，可以与有监督的方法相媲美。

### 目前存在的相关问题

- 目前ocr软件，对于低质量的文本扫描存在问题

- 很多都是采用人工进行矫正

- 对于商业图书馆的数字化的产生阻碍

- the scale of these projects not only makes it dif- ficult to adapt ocr models to their diverse layouts and typefaces but also makes it impractical to present any ocr output other than a single-best transcript

  这些项目的规模不仅使得使ocr模型适应其多样化的布局和字体变得不现实，而且除了一份最好的成绩单之外，还使任何ocr的输出都变得不切实际。

### 目前解决方法

- existing methods for automatic ocr postcorrection are mostly supervised methods that correct recognition errors in a single ocr output

  现有的自动OCR后校正方法大多是对单个OCR输出中的识别错误进行修正的监督方法。

- 这些系统是不可扩展的，并且人工的成本很高

- 集成方法，将同一OCR扫描的结果，进行结合（another line of work is ensemble methods (lund et al., 2013, 2014) combining ocr results from multiple scans of the same document.

  另一项工作是集成方法(Lund等人，2013年，2014年)，将同一文档多次扫描的OCR结果结合起来。）

- most of these ensemble methods, however, require aligning multiple ocr outputs (lund and ringger, 2009; lund et al., 2011), which is intractable in general and might introduce noise into the later correction stage.

  然而，大多数这些集成方法都需要调整多个OCR输出(Lund和Ringger，2009年；Lund等人，2011年)，这在总体上是棘手的，可能会在后期的校正阶段引入噪声。

- furthermore, voting-based ensemble methods (lund and ringger, 2009; wemhoener et al., 2013; xu and smith, 2017) only work where the correct output exists in one of the inputs, while classification methods (boschetti et al., 2009; lund et al., 2011; al azawi et al., 2015) are also trained on human annotations.

  此外，基于投票的集成方法(Lund和Ringger，2009年；wemhoener等人，2013年；许和史密斯，2017年)只在一种输入中存在正确产出的情况下工作，而分类方法(Boschetti等人，2009年；Lund等人，2011年；al Azawi等人，2015年)也接受了关于人工注释的培训。

### 本实验所采用方法

- to address these challenges, we propose an unsupervised ocr post-correction framework both to correct single input text sequences and also to exploit multiple candidate texts by simultaneously aligning, correcting, and voting among input sequences.

  为了应对这些挑战，我们提出了一个无监督的OCR后校正框架，既可以纠正单个输入文本序列，也可以利用多个候选文本，同时对输入序列进行对齐、校正和投票。

- our proposed method is based on the observation that significant number of duplicate and near-duplicate documents exist in many corpora (xu and smith, 2017), resulting in ocr output containing repeated texts with various quality.

  我们提出的方法是基于大量的重复和近似重复的文档存在于许多语料库中的观察结果(XU和Smith，2017)，导致OCR输出包含不同质量的重复文本。

- as shown by the example in table 1, different errors (characters in red) are introduced when the ocr system scans the same text in multiple editions, each with its own layout, fonts, etc.

  如表1中的示例所示，当OCR系统在多个版本中扫描相同的文本时，会引入不同的错误(红色字符)，每个版本都有自己的布局、字体等。

- therefore, duplicated texts with diverse errors could serve as complementary information sources for each other.

  因此，具有不同错误的重复文本可以作为彼此互补的信息源。

![](https://ws1.sinaimg.cn/large/e93305edgy1fxx51fm1a8j20fn045q3h.jpg)

- we propose to map each erroneous ocr’d text unit to either its high-quality duplication or a consensus correction among its duplications via bootstrapping from an uniform error model.

  我们建议将每个错误的OCR文本单元映射到其高质量的复制或通过从统一的错误模型中引导来纠正其复制之间的协商一致。

- 

### 所用模型





### 实验





### 结论





### 启发





### 参考文献







