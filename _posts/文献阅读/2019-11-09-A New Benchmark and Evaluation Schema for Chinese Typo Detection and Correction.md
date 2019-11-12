---
layout: post
title: "A New Benchmark and Evaluation Schema for Chinese Typo Detection and Correction"
tag: 文献阅读
---

# 摘要

Despite the vast amount of research related to Chinese typo
detection, we still lack a publicly available benchmark dataset
for evaluation. Furthermore, no precise evaluation schema for
Chinese typo detection has been defined. In response to these
problems: (1) we release a benchmark dataset to assist research on Chinese typo correction; (2) we present an evaluation schema which was adopted in our NLPTEA 2017 Shared
Task on Chinese Spelling Check; and (3) we report new improvements to our Chinese typo detection system ACT.

尽管有关中国错字的研究很多
检测，我们仍然缺乏公开可用的基准数据集
进行评估。 此外，没有精确的评估方案
中文错字检测已定义。 针对这些
问题：（1）我们发布了基准数据集，以帮助研究中文错字； （2）我们介绍了NLPTEA 2017 Shared中采用的评估架构
中文拼写检查任务； （3）我们报告了中文错字检测系统ACT的新改进。

# **简介**

Automatic typo detection is an important prerequisite step
for many Natural Language Processing (NLP) applications
to better understand the underlying semantics of sentences.
Error detection applications for Chinese are not yet well developed, unlike applications for alphabet-based languages
such as English and French. This is partly due to a lack of
benchmark data, and due to the absence of a precise evaluation schema. Our contributions described in this paper are:
• Constructed and released a benchmark dataset for Chinese
typo detection;
• Proposed a new schema for evaluating the performance of
a typo detection system;
• Improved a system for automatically detecting typos in
Chinese, which is an extension of our previous work

自动错字检测是重要的前提步骤
适用于许多自然语言处理（NLP）应用
以便更好地理解句子的基本语义。
与基于字母的语言的应用程序不同，中文的错误检测应用程序尚未开发完善
例如英语和法语。 部分原因是缺乏
基准数据，并且由于缺少精确的评估方案。 本文描述的我们的贡献是：
•构建并发布了中文基准数据集
错字检测；
•提出了一种新的方案来评估
错字检测系统；
•改进了自动检测打字错误的系统
中文，这是我们以前工作的延伸



# Benchmark Dataset

The Hong Kong Applied Science and Technology Research
Institute first collected more than 5,000 writings by Hong
Kong primary students. We then invited researchers from the
Department of Chinese Language and Literature at CUHK
to help us mark and annotate these writings. Next, we selected a total of 6,890 sentences of a reasonable length (50–
150 characters, including punctuation) which contain at least
one error. The average number of errors in a sentence is 2.7,
and the maximum is 5. Since our benchmark dataset also requires positive examples, we manually added 3,110 entirely
correct sentences for a round total of 10,000.

香港应用科技研究
研究所首先收集了洪的五千多篇著作
孔小学生。 然后，我们邀请了来自
香港中文大学中文系
帮助我们标记和注释这些著作。 接下来，我们总共选择了6,890个合理长度的句子（50-
150个字符（包括标点符号），至少包含
一个错误。 句子中的平均错误数为2.7，
最大值为5。由于我们的基准数据集也需要正面示例，因此我们手动总共添加了3110
正确句子的总数约为10,000。

Our benchmark dataset contains the following types of errors: (1) Typo – Similar shape (e.g., in the word 辨論, 辨 is a typo and should be written as 辯. 辨 and 辯 have similar shapes); (2) Typo – Similar pronunciation (e.g., in the word 合諧, 合 is a typo and should be replaced by 和. 合 and 和 have similar pronunciations in Cantonese); (3) Colloquialism – Incorrect character (e.g., the character 佢 is colloquial and should be changed into 他); (4) Colloquialism – Incorrect phrase (e.g., the word 撞返 is colloquial even though the characters 撞 and 返 both are not. Here, 撞返 should be replaced by 碰見); (5) Incorrect word ordering (no characters or phrases are colloquial, but the ordering of some characters or words results in colloquial language. E.g., in the sentence “我走先了” the word 走先 is colloquial and should be written as 先走); (6) Mixing Simplified Chinese and Traditional Chinese (e.g., for the word 词語, 词 is simplified Chinese and should be replaced by its traditional counterpart 詞); (7) Errors in poems and idioms (e.g., in the idiom 天生我才必有 用, 才 should be replaced by 材).

我们的基准数据集包含以下类型的错误：（1）错字–形状相似（例如，在辨别词中，辨别是
错字，应写为辩词。辨和辩具有相似的
形状）； （2）错别字–类似的发音（例如，单词
合谐，合是一个错字，应替换为和。合和和
在粤语中有相似的发音）; （3）口语–不正确的字符（例如，排水渠字符是口语
并应更改为他）； （4）口语-不正确的用语（例如，即使
字符撞和返都不是。在这里，撞返应该用碰见代替）； （5）单词顺序错误（无字符）
或短语是口语的，但某些字符的顺序
或单词会导致口语化。例如，在句子中
“我走先了”这个词是口语，应写为先走）； （6）简体中文和繁体中文的混合
中文（例如，对于单词而言，词为简体中文
并应以其传统的对应词代替）； （7）
诗词和成语中的错误（例如，成语中的天生我才必有
用，才应用材代替）。

Note that it is possible to have any mixture of the above cases. For example, consider the sentence 大家討論緊這件 事. In this context the character 緊 is a colloquial word which means 正在 (error type 3). Yet simply replacing 緊 with 正在 is still wrong since it then triggers error type 4. Instead, the correction should be 大家正在討論這件事. To the best of our knowledge, there is no publicly available benchmark dataset that takes into account error types 3 to 7. We are the first to release such dataset, and it can be obtained from the CUHK MOE lab website.1 For the NLPTEA 2017 Shared Task on Chinese Spelling Check (Fung et al. 2017), we assembled two sets of 1,000 randomly selected sentences from the benchmark dataset. Each of these two sets then had a corresponding gold standard: the best solution that any spell checking system can possibly give. The gold standard includes as many valid corrections as possible for each error. To the best of our knowledge, these are the first datasets with this property.

请注意，可以将以上各项混合使用
案件。例如，考虑句子大家讨论紧这件
事。在这种情况下，紧紧是一个口语单词，
表示正在（错误类型3）。只是简单地用紧代替
仍然是错误的，因为它随后会触发错误类型4。相反，
纠正应该是大家正在讨论这件事。
据我们所知，没有公开可用的基准数据集考虑了错误类型3
到7。我们是第一个发布此类数据集的人，它可以是
可从香港中文大学教育部实验室网站获得1。
对于NLPTEA 2017中文拼写共享任务
Check（Fung等人，2017年），我们组装了两组1,000
从基准数据集中随机选择的句子。
然后，这两套工具中的每套都有相应的黄金标准：任何拼写检查系统都可以提供的最佳解决方案
可能给。黄金标准包括针对每个错误的尽可能多的有效更正。据我们所知，这些是具有此属性的第一个数据集。

