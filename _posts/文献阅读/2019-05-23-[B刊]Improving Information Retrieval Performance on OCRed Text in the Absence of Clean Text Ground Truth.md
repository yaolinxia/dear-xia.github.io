---
layout: post
title: "Improving Information Retrieval Performance on OCRed Text in the Absence of Clean Text Ground Truth"【B刊】
tag: 文献阅读
---

- [Ghosh K, Chakraborty A, Parui SK, Majumder P (2016) Improving Information Retrieval Performance on OCRed Text in the Absence of Clean Text Ground Truth. Inf Process Manag.
  doi:10.1016/j.ipm.2016.03.006. Artice in press 

  > 下载要钱

# **摘要**

The proposed algorithm uses context information to segregate（分离） semantically （语义）related error variants from the unrelated ones.

String similarity measures are used to join error variants with the correct query word.

The algorithm is tested on Bangla, Hindi and English datasets to show that the proposed approach is language-independent.

The Bangla and Hindi datasets have the clean, error-free versions for comparison. So, we have used the performances on the clean text versions as the performance upper-bounds. In addition, we have compared our method with an error modelling approach which, unlike our method, uses the clean version.

The English dataset is a genuine（真正） use case scenario（案例场景） for our algorithm as this dataset does not have the error-free version.

Our proposed method produces significant improvements on most of the baselines.

We have also tested our proposed algorithm on TREC 5 Confusion track dataset and showed that our proposed method is significantly better than the baselines.

所提出的算法使用上下文信息来分离语义相关的错误变体与不相关的错误变体。 •字符串相似性度量用于将错误变体与正确的查询词连接。 •该算法在Bangla，Hindi和英语数据集上进行测试，以表明所提出的方法与语言无关。 •Bangla和Hindi数据集具有干净，无错误的版本供比较。因此，我们使用干净文本版本的性能作为性能上限。此外，我们将我们的方法与错误建模方法进行了比较，与我们的方法不同，它使用了干净的版本。 •英语数据集是我们算法的真实用例场景，因为此数据集没有无错误版本。 •我们提出的方法在大多数基线上产生了显着的改进。 •我们还在TREC 5 Confusion轨道数据集上测试了我们提出的算法，并表明我们提出的方法明显优于基线。

OCR errors in text harm information retrieval performance. Much research has been reported on modelling and correction of Optical Character Recognition (OCR) errors. Most of the prior work employ language dependent resources or training texts in studying the nature of errors. However, not much research has been reported that focuses on improving retrieval performance from erroneous text in the absence of training data. We propose a novel approach for detecting OCR errors and improving retrieval performance from the erroneous corpus in a situation where training samples are not available to model errors. In this paper we propose a method that automatically identifies erroneous term variants in the noisy corpus, which are used for query expansion, in the absence of clean text. We employ an effective combination of contextual information and string matching techniques. Our proposed approach automatically identifies the erroneous variants of query terms and consequently leads to improvement in retrieval performance through query expansion. Our proposed approach does not use any training data or any language specific resources like thesaurus for identification of error variants. It also does not expend any knowledge about the language except that the word delimiter is blank space. We have tested our approach on erroneous Bangla (Bengali in English) and Hindi FIRE collections, and also on TREC Legal IIT CDIP and TREC 5 Confusion track English corpora. Our proposed approach has achieved statistically significant improvements over the state-of-the-art baselines on most of the datasets.

文本中的OCR错误危害信息检索性能。已经有关于光学字符识别（OCR）错误的建模和校正的大量研究报道。大多数先前的工作在研究错误的性质时使用依赖于语言的资源或培训文本。然而，据报道，在没有训练数据的情况下，没有太多研究关注于改善错误文本的检索性能。我们提出了一种新方法，用于在训练样本不可用于模型错误的情况下从错误的语料库中检测OCR错误并提高检索性能。在本文中，我们提出了一种方法，可以在没有干净文本的情况下自动识别嘈杂语料库中的错误术语变体，这些变体用于查询扩展。我们采用上下文信息和字符串匹配技术的有效组合。我们提出的方法自动识别查询术语的错误变体，从而通过查询扩展导致检索性能的提高。我们提出的方法不使用任何训练数据或任何语言特定资源（如同义词库）来识别错误变体。除了单词分隔符是空格外，它也不会花费任何关于语言的知识。我们已经测试了我们对错误的Bangla（英语孟加拉语）和印地语FIRE系列以及TREC Legal IIT CDIP和TREC 5 Confusion track English corpora的方法。与大多数数据集的最新基线相比，我们提出的方法取得了统计上显着的改进。