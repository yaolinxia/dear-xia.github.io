---
layout: post
title: "Adapting sequence models for sentence correction"
tag: 文献阅读
---

# **摘要**

In a controlled experiment of sequence-to- sequence approaches for the task of sen- tence correction, we find that character- based models are generally more effec- tive than word-based models and models that encode subword information via con- volutions, and that modeling the output data as a series of diffs improves effec- tiveness over standard approaches. Our strongest sequence-to-sequence model im- proves over our strongest phrase-based statistical machine translation model, with access to the same data, by 6 M2 (0.5 GLEU) points. Additionally, in the data environment of the standard CoNLL-2014 setup, we demonstrate that modeling (and tuning against) diffs yields similar or better M 2 scores with simpler models and/or significantly less data than previous sequence-to-sequence approaches. 

在针对句子校正的逐序列方法的受控实验中，我们发现基于字符的模型通常比基于单词的模型和通过卷积对子单词信息进行编码的模型更有效，并且 与标准方法相比，将输出数据建模为一系列差异的方法提高了效率。 我们最强大的序列到序列模型比我们最强大的基于短语的统计机器翻译模型提高了6 M2（0.5 GLEU）点，可以访问相同的数据。 此外，在标准CoNLL-2014设置的数据环境中，我们证明，与以前的逐序列方法相比，使用更简单的模型和/或更少的数据，对diff进行建模（和调整）会产生相似或更好的M 2分数。

# **简介**

## **背景和方法**

**Task** We follow recent work and treat the task of sentence correction as translation from a source sentence (the unedited sentence) into a target sen- tence (a corrected version in the same language as the source). We do not make a distinction between grammatical and stylistic corrections. 

We assume a vocabulary V of natural language word types (some of which have orthographic er- rors). Given a sentence s = [s1 · · · sI ], where si ∈ V is the i-th token of the sentence of length I, we seek to predict the corrected target sentence t = [t1 ···tJ], where tj ∈ V is the j-th token of the corrected sentence of length J. We are given both s and t for supervised training in the standard setup. At test time, we are only given access to se- quence s. We learn to predict sequence t (which is often identical to s). 

任务：我们关注最新的工作，将句子校正的任务视为从源句子（未编辑的句子）到目标句子（与源语言相同的校正版本）的翻译。 我们在语法和文体校正之间没有区别。
我们假设自然语言单词类型的词汇表V（其中一些单词具有拼写错误）。 给定一个句子s = [s1···sI]，其中si∈V是长度为I的句子的第i个标记，我们试图预测校正后的目标句子t = [t1···tJ]，其中tj ∈V是长度为J的更正句子的第j个记号。在标准设置中，我们给了s和t进行监督训练。 在测试时，我们只能访问序列s。 我们学习预测序列t（通常与s相同）。

Sequence-to-sequence We explore word and character variants of the sequence-to-sequence framework. We use a standard word-based model (WORD), similar to that of Luong et al. (2015), as well as a model that uses a convolutional neural network (CNN) and a highway network over char- acters (CHARCNN), based on the work of Kim et al. (2016), instead of word embeddings as the input to the encoder and decoder. With both of these models, predictions are made at the word level. We also consider the use of bidirectional versions of these encoders (+BI). 

Our character-based model (CHAR+BI) follows the architecture of the WORD+BI model, but the input and output consist of characters rather than words. In this case, the input and output sequences are converted to a series of characters and whites- pace delimiters. The output sequence is converted back to t prior to evaluation. 

序列到序列我们探索序列到序列框架的单词和字符变体。 我们使用标准的基于单词的模型（WORD），类似于Luong等人的模型。 （2015年），以及基于Kim等人的工作的使用卷积神经网络（CNN）和基于角色的高速公路网络（CHARCNN）的模型。 （2016），而不是将词嵌入作为编码器和解码器的输入。 通过这两种模型，可以在单词级别进行预测。 我们还考虑使用这些编码器的双向版本（+ BI）。
我们的基于字符的模型（CHAR + BI）遵循WORD + BI模型的体系结构，但是输入和输出由字符而不是单词组成。 在这种情况下，输入和输出序列将转换为一系列字符和空格定界符。 在评估之前，将输出序列转换回t。

The WORD models encode and decode over a closed vocabulary (of the 50k most frequent words); the CHARCNN models encode over an open vocabulary and decode over a closed vocab- ulary; and the CHAR models encode and decode over an open vocabulary. 

WORD模型在一个封闭的词汇表（最多50k个最常用词）中进行编码和解码； CHARCNN模型在开放的词汇表上编码，并在封闭的词汇表上解码； 而CHAR模型则通过开放的词汇表进行编码和解码。

Our contribution is to investigate the impact of sequence-to-sequence approaches (including those not considered in previous work) in a series 

of controlled experiments, holding the data con- stant. In doing so, we demonstrate that on a large, professionally annotated dataset, the most effec- tive sequence-to-sequence approach can signifi- cantly outperform a state-of-the-art SMT system without augmenting the sequence-to-sequence model with a secondary model to handle low- frequency words (Yuan and Briscoe, 2016) or an additional model to improve precision or inter- secting a large language model (Xie et al., 2016). We also demonstrate improvements over these previous sequence-to-sequence approaches on the CoNLL-2014 data and competitive results with Ji et al. (2017), despite using significantly less data. 

The work of Schmaltz et al. (2016) applies WORD and CHARCNN models to the distinct bi- nary classification task of error identification. 

我们的贡献是在一系列研究序列对序列方法（包括先前工作中未考虑的方法）的影响
控制实验，保持数据不变。 通过这样做，我们证明了在大型的，带有专业注释的数据集上，最有效的序列到序列方法可以显着优于最新的SMT系统，而无需使用辅助模型来增强序列到序列模型 处理低频单词（Yuan和Briscoe，2016）或其他模型以提高精度或与大型语言模型相交（Xie等，2016）。 我们还证明了在CoNLL-2014数据上这些以前的逐序列方法的改进以及与Ji等人的竞争结果。 （2017），尽管使用的数据少得多。
Schmaltz等人的工作。 （2016年）将WORD和CHARCNN模型应用于错误识别的不同二元分类任务。

Additional Approaches The standard formula- tion of the correction task is to model the output sequence as t above. Here, we also propose mod- eling the diffs between s and t. The diffs are pro- vided in-line within t and are described via tags marking the starts and ends of insertions and dele- tions, with replacements represented as deletion- insertion pairs, as in the following example se- lected from the training set: “Some key points are worth <del> emphasiz </del> <ins> emphasiz- ing </ins> .”. Here, “emphasiz” is replaced with “emphasizing”. The models, including the CHAR model, treat each tag as a single, atomic token. 

The diffs enable a means of tuning the model’s propensity to generate corrections by modifying the probabilities generated by the decoder for the 4 diff tags, which we examine with the CoNLL data. We include four bias parameters associated with each diff tag, and run a grid search between 0 and 1.0 to set their values based on the tuning set. 

It is possible for models with diffs to output invalid target sequences (for example, inserting a word without using a diff tag). To fix this, a deter- ministic post-processing step is performed (greed- ily from left to right) that returns to source any non-source tokens outside of insertion tags. Diffs are removed prior to evaluation. We indicate mod- els that *do not* incorporate target diff annotation tags with the designator –DIFFS. 

The AESW dataset provides the paragraph con- text and a journal domain (a classification of the document into one of nine subject categories) for each sentence.1 

其他方法校正任务的标准公式是将输出序列建模为上述t。在这里，我们还建议对s和t之间的差异建模。差异在t内以行形式提供，并通过标记插入和删除的开始和结束的标签进行描述，其中替换表示为删除插入对，如以下训练集中的示例所示： “一些关键点值得<del>强调</ del> <ins>强调</ ins>。”在这里，“强调”被替换为“强调”。这些模型（包括CHAR模型）将每个标签视为单个原子令牌。
通过修改解码器为4个diff标签生成的概率（我们使用CoNLL数据进行了检查），差异使调整模型倾向以生成校正的方法成为可能。我们包括与每个diff标签相关的四个偏差参数，并在0到1.0之间运行网格搜索以基于调整集设置其值。
带有差异的模型可能会输出无效的目标序列（例如，插入单词而不使用差异标签）。为了解决这个问题，执行了确定的后处理步骤（最好从左到右），该步骤将返回插入标签之外的所有非源标记。评估之前先清除差异。我们用-DIFFS表示未包含目标差异注释标签的模型。
AESW数据集为每个句子提供了段落上下文和日记域（将文档分类为九个主题类别之一）。



# **数据集**

Data AESW (Daudaravicius, 2016; Daudaravicius et al., 2016) consists of sentences taken from academic articles annotated with corrections by professional editors used for the AESW shared task. The training set contains 1,182,491 sentences, of which 460,901 sentences have edits. We set aside a 9,947 sentence sample from the original development set for tuning (of which 3,797 contain edits), and use the remaining 137,446 sentences as the dev set 3 (of which 53,502 contain edits). The test set contains 146,478 sentences.

数据AESW（Daudaravicius，2016; Daudaravicius et al。，2016）包括从学术文章中摘录的句子，并由用于AESW共享任务的专业编辑者进行了更正。 训练集包含1,182,491个句子，其中460,901个句子进行了编辑。 我们从原始开发集中预留了9,947个句子样本进行调整（其中3,797个包含编辑），并将其余137,446个句子用作开发集3（其中53,502个包含编辑）。 测试集包含146,478个句子。