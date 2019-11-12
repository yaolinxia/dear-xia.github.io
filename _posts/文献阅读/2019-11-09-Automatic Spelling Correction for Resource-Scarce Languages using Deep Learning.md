---
layout: post
title: "Automatic Spelling Correction for Resource-Scarce Languages using Deep Learning"
tag: 文献阅读
---

# **摘要**

Spelling correction is a well-known task in Natural Language Processing (NLP). Automatic spelling correction is important for many NLP applications like web search engines, text summarization, sentiment analysis etc. Most approaches use parallel data of noisy and correct word mappings from different sources as training data for automatic spelling correction. Indic languages are resourcescarce and do not have such parallel data due to low volume of queries and non-existence of such prior implementations. In this paper, we show how to build an automatic spelling corrector for resourcescarce languages. We propose a sequenceto-sequence deep learning model which trains end-to-end. We perform experiments on synthetic datasets created for Indic languages, Hindi and Telugu, by incorporating the spelling mistakes committed at character level. A comparative evaluation shows that our model is competitive with the existing spell checking and correction techniques for Indic languages.

拼写校正是自然语言处理（NLP）中的一项众所周知的任务。 自动拼写校正对许多NLP应用程序（例如Web搜索引擎，文本摘要，情感分析等）非常重要。大多数方法使用嘈杂的并行数据和来自不同来源的正确单词映射作为自动拼写校正的训练数据。 由于查询量少且不存在此类现有实现，因此印度语资源贫乏且没有此类并行数据。 在本文中，我们展示了如何为资源匮乏的语言构建自动拼写校正器。 我们提出了一个端到端训练的序列到序列深度学习模型。 通过结合在字符级别犯下的拼写错误，我们在为印度语（印地语和泰卢固语）创建的合成数据集上进行了实验。 一项比较评估表明，我们的模型与现有的印度语拼写检查和更正技术具有竞争力。

# **简介**

Spelling correction is important for many of the potential NLP applications such as text summarization, sentiment analysis, machine translation (Belinkov and Bisk, 2017). Automatic spelling correction is crucial in search engines as spelling mistakes are very common in user-generated text. Many websites have a feature of automatically giving correct suggestions to the misspelled user queries in the form of Did you mean? suggestions or automatic corrections.

拼写校正对于许多潜在的NLP应用程序非常重要，例如文本摘要，情感分析，机器翻译（Belinkov和Bisk，2017年）。 自动拼写更正在搜索引擎中至关重要，因为拼写错误在用户生成的文本中非常常见。 许多网站都具有自动为拼写错误的用户查询提供正确建议的功能，形式为“您是说吗？”。 建议或自动更正。

suggestions or automatic corrections. Providing suggestions makes it convenient for users to accept a proposed correction without retyping or correcting the query manually. This task is approached by collecting similar intent queries from user logs (Hasan et al., 2015; Wilbur et al., 2006; Ahmad and Kondrak, 2005). The training data is automatically extracted from event logs where users re-issue their search queries with potentially corrected spelling within the same session. Example query pairs are (house lone, house loan), (ello world, hello world), (mobile phone, mobile phone). Thus, large amounts of data is collected and models are trained using techniques like Machine Learning, Statistical Machine Translation etc.

建议或自动更正。 提供建议使用户可以方便地接受建议的更正，而无需手动重新键入或更正查询。 通过从用户日志中收集类似的意图查询来实现此任务（Hasan等人，2015； Wilbur等人，2006； Ahmad和Kondrak，2005）。 训练数据是从事件日志中自动提取的，用户可以在事件日志中使用同一会话中可能更正的拼写来重新发出搜索查询。 示例查询对是（房屋独栋，房屋贷款），（世界，问候世界），（手机，手机）。 因此，使用诸如机器学习，统计机器翻译等技术来收集大量数据并训练模型。

The task of spelling correction is challenging for resource-scarce languages. In this paper, we consider Indic languages, Hindi and Telugu, because of their resource scarcity. Due to lesser query share, we do not ﬁnd the same level of parallel alteration data from logs. We also do not have many language resources such as Parts of Speech (POS) Taggers, Parsers etc. to linguistically analyze and understand these queries. Due to lack of relevant data, we create synthetic dataset using highly probable spelling errors and real world errors in Hindi and Telugu given by language experts. Similarly, synthetic dataset can be created for any resource-scarce language incorporating the real world errors. Deep Learning techniques have shown enormous success in sequence to sequence mapping tasks (Sutskever et al., 2014). Most of the existing spell-checkers for Indic languages are implemented using rule-based techniques (Kumar et al., 2018). In this paper, we approach the spelling correction problem for Indic languages with Deep learning. This model can be employed for any resource-scarce language. We propose a character based Sequence-to-sequence text Correction Model for Indic Languages (SCMIL) which trains end-to-end.

对于资源稀缺的语言，拼写纠正的任务是具有挑战性的。在本文中，由于资源稀缺，我们考虑了印度语（印地语和泰卢固语）。由于较少的查询份额，我们无法从日志中找到相同级别的并行变更数据。我们也没有很多语言资源（例如词性（POS）标注器，解析器等）来进行语言分析和理解这些查询。由于缺少相关数据，我们使用语言专家给出的北印度文和泰卢固文中极有可能的拼写错误和现实错误来创建综合数据集。同样，可以为任何包含实际错误的资源稀缺的语言创建综合数据集。深度学习技术在序列映射任务方面显示出巨大的成功（Sutskever et al。，2014）。现有的大多数印度语拼写检查器都是使用基于规则的技术实现的（Kumar等人，2018）。在本文中，我们通过深度学习解决了印度语的拼写纠正问题。该模型可用于任何资源稀缺的语言。我们提出了一种基于字符的印度语序列到序列文本校正模型（SCMIL），该模型可以进行端到端的训练。

Our main contributions in this paper are summarized as follows:

• We propose a character based recurrent sequence-to-sequence architecture with a Long Short Term Memory (LSTM) encoder and a LSTM decoder for spelling correction of Indic languages.

• We create synthetic datasets 1 of noisy and correct word mappings for Hindi and Telugu by collecting highly probable spelling errors and inducing noise in clean corpus.

• We evaluate the performance of SCMIL by comparing with various approaches such as Statistical Machine Translation (SMT), rulebased methods, and various deep learning models, for this task.

我们在本文中的主要贡献概述如下：

•我们提出了一种基于字符的循环序列到序列结构，该结构具有长短期记忆（LSTM）编码器和LSTM解码器，用于印度语的拼写校正。

•我们通过收集极有可能的拼写错误并在干净的语料库中引起噪音，为北印度语和泰卢固语创建了嘈杂的合成数据集1和正确的单词映射。

•我们通过与各种方法（例如统计机器翻译（SMT），基于规则的方法和各种深度学习模型）进行比较来评估SCMIL的性能，以完成此任务。

# **模型**

We address the spelling correction problem for Indic languages by having a separate corrector network as an encoder and an implicit language model as a decoder in a sequence-to-sequence attention model that trains end-to-end.

我们通过在端到端训练的序列到序列注意模型中使用单独的校正器网络作为编码器和隐式语言模型作为解码器来解决印度语的拼写校正问题。

## Sequence-to-sequence Model

Sequence-to-sequence (seq2seq) models (Sutskever et al., 2014; Cho et al., 2014) have enjoyed great success in a variety of tasks such as machine translation, speech recognition, image captioning, and text summarization. A basic sequence-to-sequence model consists of two neural networks: an encoder that processes the input and a decoder that generates the output. This model has shown great potential in input-output sequence mapping tasks like machine translation. An input side encoder captures the representations in the data, while the decoder gets the representation from the encoder along with the input and outputs a corresponding mapping to the target language. Intuitively, this architectural set-up seems to naturally ﬁt the regime of mapping noisy input to de-noised output, where the corrected prediction can be treated as a different language and the task can be treated as Machine Translation.

序列到序列（seq2seq）模型（Sutskever等，2014； Cho等，2014）在各种任务（例如机器翻译，语音识别，图像字幕和文本摘要）中都取得了巨大的成功。 一个基本的序列到序列模型由两个神经网络组成：一个处理输入的编码器和一个生成输出的解码器。 该模型在输入输出序列映射任务（如机器翻译）中显示出巨大潜力。 输入端编码器捕获数据中的表示，而解码器则从编码器获取输入的表示，并输出到目标语言的对应映射。 直观上，这种体系结构设置似乎自然地适合将有噪声的输入映射为经过降噪的输出的方式，其中可以将校正后的预测视为另一种语言，而将任务视为机器翻译。

## System Architecture

SCMIL has the similar underlying architecture of sequence-to-sequence models. The encoder and decoder in SCMIL operate at character level.
Encoder: In SCMIL, the encoder is a character based LSTM. With LSTM as encoder, the input sequence is modeled as a list of vectors, where each vector represents the meaning of all characters in the sequence read so far.
Decoder: The decoder in SCMIL is a character level LSTM recurrent network with attention. The output from the encoder is the ﬁnal hidden state of the character based LSTM encoder. This becomes the input to the LSTM decoder.

SCMIL具有类似的序列到序列模型的基础架构。 SCMIL中的编码器和解码器以字符级别运行。
编码器：在SCMIL中，编码器是基于字符的LSTM。 使用LSTM作为编码器，输入序列被建模为一个向量列表，其中每个向量代表到目前为止读取的序列中所有字符的含义。
解码器：SCMIL中的解码器是字符级的LSTM循环网络，值得关注。 编码器的输出是基于字符的LSTM编码器的最终隐藏状态。 这成为LSTM解码器的输入。