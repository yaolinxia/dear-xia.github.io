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

- we propose to map each erroneous ocr’d text unit to either its high-quality duplication or a consensus correction among its duplications via bootstrapping from an uniform error model.

  我们建议将每个错误的OCR文本单元映射到其高质量的复制或通过从统一的错误模型中引导来纠正其复制之间的协商一致。

- in this paper, we aim to train an unsupervised correction model via utilizing the duplication in ocr output。

  本文旨在利用OCR输出中的重复信息来训练一个无监督的校正模型。

- the baseline correction system is a sequence-to-sequence model with attention (bahdanau et al., 2015), which has been shown to be effective in text correction tasks (chollampatt et al., 2016; xie et al., 2016)

  基线校正系统是一个具有注意的序列到序列模型(bahdanau等人，2015年)，该模型已被证明在文本校正任务中是有效的(chollampatt等人，2016年；谢等人，2016年)。

- we also seek to improve the correction performance for duplicated texts by integrating multiple inputs.

  我们还试图通过集成多个输入来提高重复文本的校正性能。

- previous work on combining multiple inputs in neural translation deal with data from different domains

  以往在神经翻译中结合多个输入的工作都是针对不同领域的数据进行的。

- therefore, their models need to be trained on multiple inputs to learn parameters to combine inputs from each domain.

  因此，需要对它们的模型进行多输入方面的培训，以学习参数以组合来自每个领域的输入。

- given that the inputs of our task are all from the same domain, our model is trained on a single input and introduces multi-input attention to generate a consensus result merely for decoding.

  考虑到我们任务的输入都来自同一领域，我们的模型被训练成一个单一的输入，并引入多输入注意来产生一个一致的结果，仅仅是为了解码。

- it does not require learning extra parameters for attention combination and thus is more efficient to train.

  它不需要学习额外的参数为注意力组合，因此是更有效的训练。

- furthermore, average attention combination, a simple multi-input attention mechanism, is proposed to improve both the effectiveness and efficiency of multi-input combination on the ocr post-correction task.

  此外，为了提高多输入组合在OCR后校正任务中的有效性和效率，提出了一种简单的多输入注意组合机制&平均注意组合。

### 实验

- 分别在监督和非监督训练集上训练

- we experiment with both supervised and unsupervised training and with single- and multi-input decoding on data from two manually transcribed collections in english with diverse typefaces, genres, and time periods:

  我们试验有监督的和无监督的培训，并对两个人工转录的英语集合中的数据进行单输入和多输入解码，这些数据具有不同的字体、类型和时间段：

- for both collections, which were manually transcribed by other researchers and are in the public  domain, we aligned the one-best output of an ocr system to the manual transcripts.

  对于这两个由其他研究人员手工转录并属于公共领域的集合，我们将OCR系统的一个最佳输出与手工记录进行了比对。

### 数据

- newspaper articles from the richmond (virginia) daily dispatch (rdd) from 1860–1865

  1860年至1865年，来自里士满(弗吉尼亚)日报的文章

- books from 1500– 1800 from the text creation partnership (tcp)

  来自文本创建伙伴关系(Tcp)的1500到1800的书籍。

- for both collections, which were manually transcribed by other researchers and are in the public domain, we aligned the one-best output of an ocr system to the manual transcripts.

  对于这两个由其他研究人员手工转录并属于公共领域的集合，我们将OCR系统的一个最佳输出与手工记录进行了比对。

- we also aligned the ocr in the training and evaluation sets to other public-domain newspaper issues (from the library of congress) and books (from the internet archive) to find multiple duplicates as “witnesses”, where available, for each line.

  我们还将培训和评估集中的OCR与其他公共领域的报纸问题(来自国会图书馆)和书籍(来自互联网档案)进行调整，以便为每一行找到多份作为“见证者”的副本。

- experimental results on both datasets show that our proposed averarge attention combination mechanism is more effective than existing methods in integrating multiple inputs.

  在这两个数据集上的实验结果表明，我们提出的平均注意力组合机制比现有的多输入融合方法更有效。

- moreover, our noisy error correction model achieves comparable performance with the supervised model via multiple-input decoding on duplicated texts.

  此外，通过对重复文本进行多输入解码，我们的噪声纠错模型达到了与监督模型相当的性能。

#### 数据收集

- 从两个来源中选择一个最好的OCR输出， 基于此之上，做实验。1）美国历史新闻——200万，2）公共领域300万存档

  chronicling america and the internet archive

  记录美国和互联网档案

- **监督学习：**训练以及评估；首先进行人工匹配

- text creation partnership

  文本创建伙伴关系

- both of these manually transcribed collections, which were produced independently from the current authors, are in the public domain and in english, although both chronicling america and the internet archive also contain much non-english text.

  这两本手工抄录的藏书都是在公共领域和英文中独立制作的，尽管美国的编年史和互联网档案都包含了大量的非英语文本。

- 为了获得更多的证据来阅读OCR的第几行

- we aligned each ocr’d rdd issue to other issues of the rdd and other newspapers from chronicling america and aligned each ocr’d tcp page to other pre-1800 books in the internet archive

  我们将每一期OCR的RDD问题与RDD的其他问题和其他报纸联系起来，从美国编年史开始，并将每一OCR的TCP页面与互联网档案中其他1800前的书籍对齐。

- to perform these alignments between noisy ocr transcripts efficiently, we used methods from our earlier work on text-reuse analysis

  为了有效地在含噪声的ocr转录本之间进行这些对齐，我们使用了我们先前在文本重用分析方面的工作中的方法。

- an inverted index of hashes of word 5-grams was produced, and then all pairs from different pages in the same posting list were extracted

  生成了一个单词5-克的倒哈希索引，然后从同一发帖列表中的不同页面中提取所有对。

- pairs of pages with more than five shared hashed 5-grams were aligned with the smith-waterman algorithm with equal costs for insertion, deletion, and substitution, which returns a maximally aligned subsequence in each pair of pages

  具有超过5个共享散列5克的页面对齐使用smith-Waterman算法，其插入、删除和替换代价相等，该算法在每对页中返回最大对齐子序列。

- aligned passages that were at least five lines long in the target rdd or tcp text were output

  输出目标rdd或tcp文本中至少5行长的对齐段落。

- for each target ocr line—i.e., each line in the training or test set—there are thus, in addition to the ground-truth manual transcript, zero or more witnesses from similar texts, to use the term from textual criticism.

  因此，对于每一条目标OCR线-即训练或测试集中的每一行-除了地面-“真相手册”成绩单外，还有来自类似文本的零名或多名证人使用来自考证的术语。

#### 训练和测试

- in our experiments on ocr correction, each training and test example is a line of text following the layout of the scanned image documents

  在我们关于ocr校正的实验中，每个训练和测试示例都是按照扫描图像文档的布局排列的一行文本。

- the average number of characters per line is 42.4 for the rdd newspapers and 53.2 for the tcp books.

  RDD报纸的每行字符数平均为42.4，TCP图书为53.2。

- table 2 lists statistics for the number of ocr’d text lines with manual transcriptions and additional witnesses

  表2列出了带有人工抄写和其他证人的OCR文本行数的统计数据

  ![](https://ws1.sinaimg.cn/large/e93305edgy1fy2ttc1nl7j20h407swg3.jpg)

- in the full chronicling america data, 44% of lines align to at least one other witness.

  在完整的美国数据编年史中，44%的行与至少一位其他目击者（见证）保持一致。

- 43%的人工转录在RDD中出现，64%的则在TCP书中出现

- although not all ocr collections will have this level of repetition, it is notable that these collections, which are some of the largest public-domain digital libraries, do exhibit this kind of reprinting.

  虽然并非所有的OCR藏品都会有这种程度的重复，但值得注意的是，这些收藏是一些最大的公共领域数字图书馆，确实展示了这种重印。

- similarly, at least 25% of the pages in google’s web crawls are duplicates

  类似地，Google网页抓取中至少有25%的页面是重复的

- 


### 论文贡献

- a scalable framework needing no supervision from human annotations to train the correction model

  一个无需人工注释监督的可扩展框架来训练校正模型

- a multi-input attention mechanism incorporating aligning, correcting, and voting on multiple sequences simultaneously for consensus decoding, which is more efficient and effective than existing ensemble methods

  一种同时对多个序列进行比对、校正和投票的多输入注意机制，与现有的集成方法相比，具有更高的效率和效率。

- a method that corrects text either with or without duplicated versions, while most existing methods can only deal with one of these cases.

  一种纠正文本的方法，无论是否有重复的版本，而大多数现有方法只能处理这些情况中的一种。

- to get more evidence for the correct reading of an ocr’d line, we aligned each ocr’d rdd issue to other issues of the rdd and other newspapers from chronicling america and aligned each ocr’d tcp page to other pre-1800 books in the internet archive.

  为了获得正确阅读OCR‘d行的更多证据，我们将OCR的RDD问题与RDD的其他问题和其他报纸联系起来，从记录美国开始，并将每一个OCR’tcp页面与互联网档案中其他1800前的书籍对齐。



### 方法

#### 问题

![](https://ws1.sinaimg.cn/large/e93305edgy1fy2u61ksndj20h408xmzd.jpg)

#### 模型

- 基于注意力的端到端模型

- 先前的工作：双向RNN(1997)，将原序列转化成RNN的状态序列

- a concatenation of both forward and backward hidden states at time step

  前向和后向隐态在时间步态的级联

  ![](https://ws1.sinaimg.cn/large/e93305edgy1fy2v9wrzs9j20ej0f10w8.jpg)

  ![](https://ws1.sinaimg.cn/large/e93305edgy1fy2vawfoy9j20d00enwh7.jpg)


#### 多输入Attention

![](https://ws1.sinaimg.cn/large/e93305edgy1fy319q9mvrj20gv0cqdhy.jpg)







#### 几种非监督方法(训练设置)

训练矫正模型

##### 有监督

- seq2seq-super

  seq2seq-超级

##### 无监督

- in the absence of ground truth transcriptions, we can use different methods to generate a noisy corrected version for each ocr’d line

  在没有地面真实转录的情况下，我们可以使用不同的方法为每条OCR D线生成一个有噪声的校正版本

- 噪声训练

- 综合训练

- synthetic training with bootstrapping.

  用鞋带进行综合训练。



### 实验

#### 训练细节

- 随机把OCR分成了80%的训练，20%的测试
- 

#### 



### 结论





### 启发









### Attention机制



### 参考文献







