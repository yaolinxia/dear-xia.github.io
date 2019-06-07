---
layout: post
title: "Learning string distance with smoothing for OCR spelling correction【C刊】"
tag: 文献阅读
---

- [Daniel Hládek](https://dblp.uni-trier.de/pers/hd/h/Hl=aacute=dek:Daniel), [Ján Stas](https://dblp.uni-trier.de/pers/hd/s/Stas:J=aacute=n), [Stanislav Ondás](https://dblp.uni-trier.de/pers/hd/o/Ond=aacute=s:Stanislav), [Jozef Juhár](https://dblp.uni-trier.de/pers/hd/j/Juh=aacute=r:Jozef), [László Kovács](https://dblp.uni-trier.de/pers/hd/k/Kov=aacute=cs:L=aacute=szl=oacute=):
  **Learning string distance with smoothing for OCR spelling correction.** [Multimedia Tools Appl. 76(22)](https://dblp.uni-trier.de/db/journals/mta/mta76.html#HladekSOJK17): 24549-24567 (2017)

# **摘要**

- large databases of scanned documents (medical records, legal texts, historical documents) require natural language processing for retrieval and structured information extraction. errors caused by the optical character recognition (ocr) system increase ambiguity of recognized text and decrease performance of natural language processing.

  大型扫描文件数据库(医疗记录、法律文本、历史文件)需要自然语言处理才能检索和提取结构化信息。光学字符识别(OCR)系统的错误增加了识别文本的模糊性，降低了自然语言处理的性能。

- a smoothing technique is proposed to assign non-zero probability to edit operations not present in the training corpus. spelling correction accuracy is measured on database of ocr legal documents in english language.

  提出了一种平滑技术，将非零概率分配给训练语料库中不存在的操作。在OCR英文法律文件数据库中测量拼写纠正的准确性。

- language model and learning string metric with smoothing improves viterbi-based search for the best sequence of corrections and increases performance of the spelling correction system.

  语言模型和带平滑的学习字符串度量改进了基于Viterbi的搜索以获得最佳的更正顺序，并提高拼写校正系统的性能。

# **1 Introduction **

- optical character recognition (ocr) is the process of converting a digitized image into text. its main areas of application are automatic processing of hand-written business documents entries and forms, converting text from hardcopy, such as books or documents, into electronic form, and multimedia database searching for letter sequences, such as license plates in security systems.

  光学字符识别(OCR)是将数字化图像转换为文本的过程。它的主要应用领域是自动处理手写的商业文件条目和表格，将文本从硬拷贝(如书籍或文档)转换为电子形式，以及多媒体数据库搜索字母序列，如安全系统中的车牌。

- ocr assists multimedia database processing – multimedia information retrieval and extraction from video or static digitized images. additional information about improving its precision is presented in paper [12]. examples of application for ocr in multimedia databases are:

  OCR协助多媒体数据库处理-从视频或静态数字化图像中检索和提取多媒体信息。关于提高其精度的补充信息见论文[12]。多媒体数据库中OCR的应用实例如下：

- – spelling correction system in scientific databases [19]; – recognition of characters in chinese calligraphy [20]; – text extraction from video databases [36]; – preservation and processing of cultural heritage in digital humanities [28, 34]; – management of medical texts [13].

  -科学数据库中的拼写纠正系统[19]；-承认中国书法中的字符[20]；-从视频数据库中提取文本[36]；-保存和处理数字人文领域的文化遗产[28，34]；-医学文本的管理[13]。

- natural language processing of digitized texts requires error correction for improved searching in digitized document databases, document indexing, document sorting into categories, and business data acquisition. correction of ocr spelling errors improves the precision of information retrieval [8].

  数字化文本的自然语言处理需要纠正错误，以改进在数字化文档数据库中的搜索、文档索引、文档分类和业务数据获取。OCR拼写错误的纠正提高了信息检索的精度[8]。

- spelling correction recovers the original form of the word in consideration of surrounding words and typographical errors in the word without taking the original image of the page into account. ocr errors create error patterns characteristic to the recognition algorithm, print format, paper, font, and language of the document. any error correction method must be adapted to individual cases.

  拼写更正根据单词中的周围单词和排印错误恢复单词的原始形式，而不考虑页面的原始图像。OCR错误会产生特征于识别算法、打印格式、纸张、字体和文档语言的错误模式。任何纠错方法都必须适应个别情况。

- this paper proposes a correction system automatically adjustable for ocr errors. the learning algorithm adapts a parametrized string metric to specific error types manually corrected in the past.

  提出了一种自动调节OCR误差的校正系统。该学习算法将参数化的字符串度量应用于过去人工修正的特定错误类型。

- the language model improves suggestions for correction candidates taking context of the word into account.

  该语言模型改进了考虑单词上下文的更正考生的建议。

- the viterbi approach uses dynamic programming to find the best matching sequence of correct words according to the available problem information.

  Viterbi方法利用动态规划方法，根据现有的问题信息，找出正确单词的最佳匹配序列。

- the algorithm operates if it is not possible to prepare each statistical component. the correction system functions if it is not possible to train a learning string metric or to work only with the language model without losing much precision.

  如果无法准备每个统计分量，则该算法将运行。如果不可能训练一个学习字符串度量，或者只使用语言模型而不丢失很多精度，那么校正系统就能发挥作用。

- section 2 gives a brief overview of the current literature on the task after the problem statement in the introductory section. common components for better disambiguation consistent with context are language models or more advanced machine-learning techniques, such as a maximum entropy classifier or hidden markov model. custom string distance is present in some cases.

  第二节在引言部分简要概述了问题陈述之后的当前有关这一任务的文献。与上下文更好地消除歧义的常用组件是语言模型或更先进的机器学习技术，例如最大熵分类器或隐马尔可夫模型。在某些情况下，存在自定义字符串距离。

- components of the proposed system are described in section 3. the presented approach contains an error detection and spelling suggestion component, a language model, a learning string metric with parameter smoothing, and a viterbi algorithm for the best correct word sequence search. each component is described in its own subsection

  拟议系统的组成部分见第3节。该方法包括一个错误检测和拼写建议组件、一个语言模型、一个带参数平滑的学习字符串度量和一个用于最佳正确单词序列搜索的Viterbi算法。每个组件都在自己的小节中进行描述。

- section 4 describes the experimental evaluation. the proposed system is evaluated on aligned sequences from a database of ocr scanned images in the trec-5 confusion track [10]. string distance with calculated parameters is used as observation probability for the hidden markov model of the spelling correction system. the language model of a training set is used as state-transition probability. the best sequence of correction is found using the viterbi algorithm.

  第四节介绍了实验评价。在TREC-5混淆轨道[10]中，根据OCR扫描图像数据库中的对齐序列对所提出的系统进行了评估。以具有计算参数的字符串距离作为拼写校正系统隐马尔可夫模型的观测概率。训练集的语言模型作为状态转移概率.利用Viterbi算法找出了最优的校正序列。

- spelling correction has a long history of research. theoretical foundations of spelling correction were presented in [11]. spelling error types can be divided according to vocabulary into:

  拼写纠正有着悠久的研究历史。[11]介绍了拼写纠正的理论基础。拼写错误类型可根据词汇表分为：

- – non-word errors, where tokens are a sequence of characters not present in vocabulary; – real-word errors, where the error is a valid word, but not the one that was meant.

  -非单词错误，其中标记是词汇表中不存在的字符序列；-实词错误，其中错误是一个有效的单词，但不是指的错误。

- a dictionary of valid words or a formal grammar of language morphology can detect non-word errors. however, real-word errors can be detected with deeper context analysis or word sense disambiguation.

  一本有效词典或一种语言形态的形式语法可以检测非词错误.然而，通过更深入的上下文分析或词义消歧，可以发现真实的单词错误.

- spelling correction is identifying incorrect words and sorting the set of correction candidates. the best candidate for correction is selected interactively or non-interactively.

  拼写更正是指识别不正确的单词并对一组更正候选词进行排序。最好的修正选择是互动式的或非互动式的。

- common use of spelling correction is in interactive mode, where the system identifies possibly incorrect parts of text and the user selects the best matching correction.

  拼写更正的常用方式是交互模式，其中系统识别可能不正确的文本部分，用户选择最佳匹配校正。

- the paper [34] describes an interactive post-correction system for historical ocr texts. authors in [14] propose a crowd-sourcing, web based interface for spelling suggestions. the paper [28] evaluates the precision of common ocr for recognizing historical documents.

  本文[34]描述了一种交互式的OCR历史文本后校正系统。[14]作者提出了一种基于网络的集体搜索界面，用于拼写建议。本文[28]对普通OCR识别历史文献的精度进行了评价。

- **this paper is focused on non-interactive spelling correction of text produced by an ocr system, where the best correction candidates are selected automatically, taking context and string distance of the correction candidate into account. a non-interactive spelling correction system can be part of a multimedia database, a security system, or an information retrieval system.**

  本文重点研究了OCR系统产生的文本的非互动式拼写更正，其中考虑到校正候选的上下文和字符串距离，自动选择最佳的纠错候选。非交互式拼写纠正系统可以是多媒体数据库、安全系统或信息检索系统的一部分。

- one of the most recent contributions in the field of ocr spelling is correction of chinese medical records in [13]. a morphological analyzer and named entity recognizer are important parts of this system because words in chinese are not separated by spaces. presence of a word in a medical dictionary and n-gram language model is used as a feature for the maximum entropy classifier to determine the best possible correction.

  OCR拼写领域的最新贡献之一是更正[13]的中文病历。**形态学分析器和命名实体识别器是该系统的重要组成部分，因为汉语中的单词不被空格分隔。在医学词典中存在一个词，并使用n-g语言模型作为最大熵分类器的一个特征，以确定最佳可能的校正。**

- presence of a word in a medical dictionary and n-gram language model is used as a feature for the maximum entropy classifier to determine the best possible correction.

  在医学词典中存在一个词，并使用n-g语言模型作为最大熵分类器的一个特征，以确定最佳可能的校正。

- the method [8] improves information retrieval from ocr documents in indian languages with a data-driven (unsupervised) approach. this paper proposes a novel method for identifying erroneous variants of words in the absence of a clean text. **it uses string distance and contextual information to multiple variants of the same word in text distorted by ocr. identical terms are identified by classical string distance with fixed parameters and term co-occurrence in documents**

  **该方法[8]使用数据驱动(无监督)方法改进了从印度语言的OCR文档中检索信息。**本文提出了一种在没有清晰文本的情况下识别单词错误变体的新方法。它使用字符串距离和上下文信息对同一个词的多个变体在文本中被OCR扭曲。相同的术语是用具有固定参数的经典字符串距离和文档中的项共现来识别的。

- the paper [26] improves segmentation of paragraphs and words by correcting the skew angle of a document

  本文[26]通过纠正文档的斜角来改进段落和单词的切分。

- the method in [30] uses a combination of the longest common subsequences (lcs) and bayesian estimates for a string metric to automatically pick the correction candidate from google search.

  在[30]中，该方法使用最长公共子序列(LCS)和贝叶斯估计的组合，对字符串度量自动从Google搜索中选择校正候选项。

- the method in [5] uses a bigram language model and wordnet for better spelling correction. the paper [31] proposes a hidden markov model for identification and correction. the approach [7] uses morphological analysis of suffixes for candidate word proposal. the paper [21] incorporates a language model of errors into the weighted finite state transducer of ocr.

  文[5]中的方法使用bigram语言模型和Wordnet来进行更好的拼写校正。文[31]提出了一种隐马尔可夫模型进行辨识和校正。该方法[7]在候选词建议中使用后缀的形态分析。文[21]在OCR的加权有限状态换能器中引入了误差的语言模型。

# **2 State of the art of spelling correction for OCR **

- 拼写纠错有一个很长的研究历史

- 拼写纠错的理论基础见【11】，Kukich K (1992) Techniques for automatically correcting words in text. ACM Comput Surv 24(4):377–439， 其中拼写错误的种类可以根据词汇表分为

  non-word errors：不是词汇表中的一个单词

  real-word errors：是一个有效的单词，但不是真正所要表达的那个单词

  其中non-word errors,可以通过词典解决，real-word可以通过更深的上下文分析和word sense disambiguation 

- spelling correction is identifying incorrect words and sorting the set of correction candidates. the best candidate for correction is selected interactively or non-interactively

  拼写更正正在识别不正确的单词，并对一组更正候选项进行排序。最佳的校正候选被交互式地或非交互方式选择。

- 常见的拼写纠错是一个交互的模式，系统可能识别出文本中错误的部分，用户选择最好的匹配结果。

- Vobl T, Gotscharek A, Reffle U, Ringlstetter C, Schulz KU (2014) PoCoTo - an Open Source System for
  Efficient Interactive Postcorrection of OCRed Historical Texts. In: Proceedings of the First International
  Conference on Digital Access to Textual Cultural Heritage, DATeCH ’14. ACM, New York, NY, USA,
  pp 57–61. doi:10.1145/2595188.2595197 提出了一个交互式的后处理纠错系统，针对历史的OCR文本

- Springmann U, Najock D, Morgenroth H, Schmid H, Gotscharek A, Fink F (2014) OCR of Historical Printings of Latin Texts: Problems, Prospects, Progress. In: Proceedings of the First International
  Conference on Digital Access to Textual Cultural Heritage, DATeCH ’14. ACM, New York, NY, USA,
  pp 71–75. doi:10.1145/2595188.2595205 针对历史文档，对于普通OCR识别的精读做了评估

- Evershed J, Fitch K (2014) Correcting Noisy OCR: Context Beats Confusion. In: Proceedings of the
  First International Conference on Digital Access to Textual Cultural Heritage, DATeCH ’14, pp. 45–51.
  ACM, New York, NY, USA. doi:10.1145/2595188.2595200  提出了一种语言模型
- Reffle U, Ringlstetter C (2013) Unsupervised profiling of OCRed historical documents. Pattern Recog
  46(5):1346–1357. doi:10.1016/j.patcog.2012.10.002 使用字符串匹配和贝叶斯规则，确认可能的纠正

## **2.1 String distance in spelling correction **

- 在拼写纠正中，寻找两个字符串的相似性是很有必要的

- 最常用的方法是编辑距离方法

- according to edit operations, spelling errors are divided into

  根据编辑操作，拼写错误分为：

  – substitution with one letter changed into another; – deletion with one letter missing; – insertion with one letter extra.

  -用一个字母替换成另一个字母；-删除一个字母，缺失一个字母；-插入一个字母，多加一个字母。

- edit operation z is an ordered pair of symbols from source and target alphabet or the termination symbol

  编辑操作z是从源字母表和目标字母表或终止符号中的一个有序队。

