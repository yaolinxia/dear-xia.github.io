---
layout: post
title: "Correction of OCR Word Segmentation Errors in Articles from the ACL Collection through Neural Machine Translation Methods"
tag: 文献阅读
---

[Vivi Nastase](https://dblp.uni-trier.de/pers/hd/n/Nastase:Vivi), [Julian Hitschler](https://dblp.uni-trier.de/pers/hd/h/Hitschler:Julian):
Correction of OCR Word Segmentation Errors in Articles from the ACL Collection through Neural Machine Translation Methods. [LREC 2018](https://dblp.uni-trier.de/db/conf/lrec/lrec2018.html#NastaseH18) 



用神经机器翻译方法纠正文章集中的分词错误

### 方法

- depending on the quality of the original document, optical character recognition (ocr) can produce a range of errors – from erroneous letters to additional and spurious blank spaces. we applied a sequence-to-sequence machine translation system to correct word-segmentation ocr errors in scientific texts from the acl collection with an estimated precision and recall above 0.95 on test data. We present the correction process and results.

  视原始文件的质量而定，光学字符识别会产生一系列错误-从错误的字母到额外的空白。我们使用序列对序列机器翻译系统来纠正科学文本中的分词错误，并对测试数据进行精确估计和回忆。我们提出修正过程和结论

- we apply a character-level sequence-to-sequence model to learn how to segment english words in the acl collection.

  我们应用一个字符级序列到序列模型来学习如何在ACL集合中分段英文单词。

### **简介**

- The ACL anthology provides a valuable collection of scientific articles, and organizing it into a structured format could provide us with additional insight into research in this domain, help with finding related work and help with keeping up with new developments and ideas.

  ACL选集提供了一系列有价值的科学文章，并将其组织成结构化的格式为我们提供有关此主题研究的更多见解，帮助我们找到相关工作并帮助保持随着新的发展和想法。

- the acl collection contains numerous articles published before electronic submission became standard for our conferences. these older papers have been scanned and processed through ocr, resulting in texts that contain errors.

  ACL的收藏包含了在电子提交成为我们会议的标准之前发表的许多文章。这些旧的论文通过OCR进行扫描和处理，结果产生了包含错误的文本。

- 旧的文章，通过OCR已经扫描识别出来， 包含很多错误

- inspection检查， 关于这一部分一个简短的检查，展示了一个非常常见的错误，由很多假的空白的空间组成，将单词随机分成了很多较小的碎片，这个问题很普遍，进而影响了后续的处理，比如关键词的提取，

- spurious 假的，伪造的

- 基于之前的错误纠错模型，使用机器翻译模型

- 提出了一个端到端的字符级别的模型，可以学习如何切分英文单词

- written modern languages of european origin usually segment words explicitly,

  源自欧洲的现代书面语言通常明确地分割单词

- so english texts generally do not require word segmentation, but as we have seen, this problem has popped up in documents processed with ocr

  因此，英语文本一般不需要分词，但正如我们所看到的，这个问题出现在使用ocr处理的文档中。

- 这意味着我们可以简单的获取大量体积的训练数据

- 使用ACL集合中的数据，产生训练，测试，验证集， 最好的结果，达到96%的精确度

- obtained on the test data indicate that processing the part of the collection published before 2005 would solve the vast majority of the word fragmentation issues,

  从测试数据中获得的数据表明，处理2005年之前发表的部分集合将解决绝大多数的单词分割问题，

- 在这篇文章中， 我们展示了处理工具和实验，这个纠错的ACL集合，将会提供给ACL的 anthology选集编辑，从而可以适用于该社区 



### **相关工作**

- depending on their source, errors in unedited texts can fall into various categories:

  根据来源不同，未经编辑的文本中的错误可分为各种类别：

- 比如：打印错误，拼写错误，缩短/语音缩写词，错误的字符，历史文稿中的一些小错误

- 基于以上提到的一些错误，neural-based approaches originally developed for machine translation have proved to be very successful

  最初为机器翻译开发的基于神经的方法已经证明是非常成功的。

- based on these previous analyses into the kind of architectures that perform well for different types of error correction, we adopt a character-level sequence-to-sequence model for the word segmentation of English texts.

  在此基础上，对不同类型的纠错结构进行了分析，在此基础上，我们采用字符级顺序序列模型对英语文本进行分词。

### ACL collection

- ACL collection由18849个在1965-2012年间发表的科学文章组成。

- 我们观察到，最常见的OCR识别错误是不正确的单词分词错误。

- the most common ocr error we noticed was incorrect word segmentation – there are numerous spurious blank characters at random locations in the texts, as can be seen in the text fragment from paper p81-1001 displayed in figure 1 (which we reproduce as is in the file, including new lines).

  我们注意到的最常见的OCR错误是不正确的分词-文本中随机位置有许多虚假的空白字符，如图1所示的P81-1001纸中的文本片段(我们在文件中复制，包括新的行)。

- the problem is so pervasive that it influences tasks such as keyword extraction, the basis for further processing of the collection.

  这个问题是如此的普遍，以至于它影响到诸如关键字提取之类的任务，这是进一步处理集合的基础。

- this is evidenced by an inspection of the keywords produced with the saffron system (bordea et al., 2014), wherein we find keywords among the top 15 ranked for each articles that were affected by this ocr error这可以通过对藏红花系统产生的关键词(Bordea等人，2014年)进行检查来证明，在这些关键词中，我们在每一篇受OCR错误影响的文章中找到了排名前15位的关键词。

- we set to address this problem, considering that training and test data can be automatically obtained from the portion of the acl collection that consist of electronic submissions (conservatively, we choose 2005 as our lower time limit).

  我们着手解决这个问题，考虑到培训和测试数据可以自动从ACL集合中包含电子提交的部分获得(保守地，我们选择2005年作为我们的下限)。

### **基于机器翻译的矫正模型**

- Sennrich, R., Firat, O., Cho, K., Birch, A., Haddow, 

  B., Hitschler, J., Junczys-Dowmunt, M., Laubli, S., ¨ 

  Miceli Barone, A. V., Mokry, J., and Nadejde, M. 

  (2017). Nematus: a toolkit for neural machine transla

  tion. In *Proceedings of the Software Demonstrations of* 

  *the 15th Conference of the European Chapter of the As*

  *sociation for Computational Linguistics*, pages 65–68, 

  Valencia, Spain, April. Association for Computational 

  Linguistics. 

- 使用上述作者提出的传统的序列到序列的机器翻译模型。实现了基于注意力的编码-解码结构。

- for the experiments presented here we use the default cross-entropy minimization as the training objective, via (accelerated) stochastic gradient descent.

  对于这里提出的实验，我们使用默认的交叉熵最小化作为训练目标，通过(加速)随机梯度下降。

- we use this system to process sequences at the character level. the training data consists of parallel input-output sequences, with a default limit of sequence length 100. below we describe what kind of training data was provided to the system.

  我们使用这个系统在字符级别处理序列。训练数据由并行输入输出序列组成，默认限制为序列长度100.下面我们描述了向该系统提供了什么样的培训数据。

- the scanned text preserves the line breaks from the original paper, which include hyphenated words. the first processing step we apply to the entire collection is to remove the new lines if the line does not finish with a dot, question mark or colon.

  扫描后的文本保留原纸上的断线，其中包括连字符。我们应用于整个集合的第一个处理步骤是，如果该行没有以点、问号或冒号结束，则删除新行。

- hyphenated words are replaced with their non-hyphenated version if such a variant was encountered anywhere in the texts.

  如果在文本中的任何地方遇到这种变体，则将用它们的非连字版本替换连字词。

- this processing step produces texts with one paragraph per line. after this step, we separated the collection – pre-2005 (b2005) (to be conservative about the beginning of widespread use of ocr) and post-/ including 2005 (a2005).

  这个处理步骤产生每一行一个段落的文本。在这一步之后，我们分离了收集-2005年前(B 2005)(保守的开始广泛使用OCR)和后置/包括2005年(A 2005)。

- the texts with one paragraph per line are split into smaller fragments, avoiding as much as possible splitting on ”ambiguous” breaking points (i.e. spaces between text fragments which may actually be erroneous):

  每行有一个段落的文本被分割成较小的片段，尽可能避免在“模棱两可”的断点(即实际上可能是错误的文本片段之间的空格)上分割：

  - (a) split on end of sentence characters or phrase delimiting characters (.?!;: - parentheses) (b) if the fragment is longer than 50 characters, split at numbers (c) if the fragment is still longer than 50 characters, split into 50 character long sequences

    (A)在句尾分割字符或短语分隔字符(？；：-括号)

    (B)如果片段长度大于50个字符，则在数字处一分为二

    (C)如果片段仍大于50个字符，则拆分为50个字符长序列。

- produce nematus input training data by removing blank spaces from the string, and (conform nematus’ input formatting) inserting a blank space after each character

  通过从字符串中移除空格，并在每个字符后插入一个空格(符合线虫的输入格式)生成线虫输入训练数据

- produce nematus output training data by replacing blank spaces with a special sequence (##) and then inserting a blank space after each character (except the ones in the special sequence).

  通过用特殊的序列替换空格来生成线虫输出训练数据

- the parallel input-output training data is exemplified in table 1.

  平行输入输出培训数据如表1所示。

- the data prepared in this manner consists of 9,310,664 input-output parallel sequences. we selected 2,000,000 sequences for training, and 500,000 sequences for testing.

  以这种方式编制的数据由9，310，664个输入输出并行序列组成.我们选择了2，000，000序列进行训练，并选择了500，000序列进行测试。

- training was done using the system’s default settings – training is done with cross-entropy minimization with adam optimizer, encoder and decoder implement grus, learning rate 0.0001, embedding layer size 512, hidden layer size 1000, dropout for input embeddings and hidden layers 0.2.

  使用系统的默认设置完成培训–通过使用Adam优化器、编码器和解码器实现grus的交叉熵最小化进行培训，学习速率0.0001，嵌入层大小512，隐藏层大小1000，用于输入嵌入和隐藏层0.2。

- 在训练阶段构建好的模型被用来转移输入的测试数据。

- which are then compared token-by-token to the expected output test sequences.

  然后将令牌逐个与预期的输出测试序列进行比较。

### **结论和讨论**

- we have performed two evaluations: one with respect to the test data described in section 3., and one on the actually corrected data, the b2005 portion. the results of these evaluations are described in sections 5.1. and 5.2. respectively.

  我们进行了两次评估：一次是针对第3节中描述的测试数据，另一次是针对实际更正的数据，即b 2005部分。这些评价的结果见第5.1节。和5.2节。分别。

- for the acl correction, evaluation was performed on 500,000 fragments obtained as explained in section 3.. we evaluate in terms of word-level precision and recall, computing the number of correctly predicted words. 

  就ACL校正而言，对第3节所解释的500，000个碎片进行了评估。我们根据单词级别的精确性和查全率来评估，计算正确预测的单词数.

- formally, for an automatically produced sequence wa, we compute precision and recall by comparison with a gold standard sequence wgs 3 :

  形式上，对于自动生成的序列WA，我们通过与黄金标准序列WGS 3的比较来计算精度和召回：

- precision and recall on the entire test data is a microaverage of the scores for the 500,000 sequences in the test data. we obtained a precision of 0.955 and recall of 0.950 on re-segmenting into words. most of the errors we observed are caused by spaces in formulas, as in the examples in table 2.

  整个测试数据的精确性和回忆性是测试数据中500，000个序列得分的一个微观平均值。在重新分割成单词时，我们获得了0.955的精度和0.950的回忆.我们观察到的大多数错误都是由公式中的空格引起的，如表2中的例子所示。

- partially discounting this type of error (which impacts very little, if at all the text processing of the acl collection), the precision and recall become 0.979 and 0.974 respectively

  部分贴现这种类型的错误(这对acl集合的文本处理影响很小)，查准率和召回率分别达到0.979和0.974。

- for this reason we performed an additional evaluation on the a2005 test data, for tokens that do not appear in the training or development data.

  由于这个原因，我们对2005年的测试数据执行了额外的评估，用于未出现在培训或开发数据中的令牌。

- the high performance on the test data and on unknown tokens indicates that applying the model to the b2005 collection will likely solve many of the existing segmentation problems.

  测试数据和未知标记的高性能表明，将该模型应用于b 2005集合可能解决许多现有的分割问题。

#### 在已经纠正的数据上进行评估

- annotated each of their keywords as correct or incorrect with respect to word segmentation

  在分词方面将每个关键字标注为正确或不正确。

- 在2005年之前的，我们随机选取了40篇论文， 并且对单词的分词，标注为正确或者不正确

- this provided a total of 573 keywords, 38 of which were incorrect, and 535 correct.

  这共提供了573个关键词，其中38个不正确，535个正确。

- . we tested each of these keywords against the raw and corrected versions of the corresponding files, and present the summary of results in table 3.

  我们根据相应文件的原始版本和更正版本测试了这些关键字，并在表3中给出了结果摘要

- those that do not appear in the raw files seem to be caused by some preprocessing done by the keyword extractor – e.g. for the paper a94-1014, the keyword that does not appear in the raw (or corrected) file is ”computer science univ”, which seems to have been caused by collapsing the lines that contain the authors’ affiliations (”dept . of computer science [new line] univ . of central florida”)

  未出现在原始文件中的关键字似乎是由关键字提取器进行的一些预处理引起的，例如，对于论文A94-1014，未出现在原始(或更正的)文件中的关键字是“计算机科学Univ”，它似乎是由折叠包含作者的从属关系的行(“dept”)引起的。计算机科学[新线]大学在佛罗里达中部“)

- for the correct keywords, the reason why some do not appear in the raw or the corrected files is mainly the preprocessing done by the keyword extractor (e.g. lemmatization). there are two correct keywords that appear in the raw files but not in the corrected files.

  对于正确的关键字，一些在原始或更正的文件中不ap 的原因主要是关键字提取器进行的预处理（例如，外渗）。有两个正确的关键字出现在原始文件中，而不是在更正的文件中。

- information about availability will be posted on the website of the university of heidelberg’s computational linguistics institute5

  有关可用性的信息将张贴在海德堡大学计算语言学研究所的网站上。

### 实验

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/微信截图_20190305165614.png)





### 结论

- 提出了字符级别序列到序列的模型，用来纠正OCR识别中一些普遍的错误，就是残缺词的伪空白（spurious blank spaces that fragment words）

- the high results on the test portion of the data indicate that a large part of this type of errors could be corrected in the acl collection.

  数据测试部分的高结果表明，这类错误的很大一部分可以在ACL集合中纠正。

- we have applied this process and produced a cleaner version of the acl collection, which we will offer to the acl anthology editors to make it available with the raw collection to the community

  我们已经应用了这个过程，并制作了一个更干净的ACL集合版本，我们将提供给ACL选集编辑器，以便将它与原始的集合一起提供给社区。



### 启发

- 数据集，来自ACL的论文
- 针对的问题是OCR识别后的分词错误

- 该篇文章，使用传统的机器翻译的模型，针对OCR识别过程中存在的虚假空白的问题，进行OCR识别结果的一些纠正。

### 参考文献







