---
layout: post
title: "Overview of SIGHAN 2014 Bake-off for Chinese Spelling Check"
tag: 文献阅读
---

# Overview of SIGHAN 2014 Bake-off for Chinese Spelling Check



Chinese spelling checkers are relatively difficult 

to develop, partly because no word delimiters 

exist among Chinese words and a Chinese word 

can contain only a single character or multiple 

characters. Furthermore, there are more than 13 

thousand Chinese characters, instead of only 26 

letters in English, and each with its own context 

to constitute a meaningful Chinese word. All 

these make Chinese spell checking a challengea

ble task. 

An empirical analysis indicated that Chi

nese spelling errors frequently arise from confu

sion among multiple-character words, which are 

phonologically and visually similar, but semanti

cally distinct (Liu et al., 2011). The automatic 

spelling checker should have both capabilities of 

identifying the spelling errors and suggesting the 

correct characters of erroneous usages. The 

SIGHAN 2013 Bake-off for Chinese Spelling 

Check was the first campaign to provide data sets 

as benchmarks for the performance evaluation of 

Chinese spelling checkers (Wu et al., 2013). The 

data in SIGHAN 2013 originated from the essays 

written by native Chinese speakers. Following 

the experience of the first evaluation, the second 

bake-off was held in CIPS-SIGHAN Joint CLP-2014 conference, which focuses on the essays 

written by learners of Chinese as a Foreign Lan

guage (CFL) (Yu et al., 2014). 

Due to the greater challenge in detecting and 

correcting spelling errors in CFL leaners’ written 

essays, SIGHAN 2015 Bake-off, again features a 

Chinese Spelling Check task, providing an eval

uation platform for the development and imple

mentation of automatic Chinese spelling check

ers. Given a passage composed of several sen

tences, the checker is expected to identify all 

possible spelling errors, highlight their locations,

and suggest possible corrections. 

The rest of this article is organized as follows. 

Section 2 provides an overview of the SIGHAN 

2015 Bake-off for Chinese Spelling Check. Sec

tion 3 introduces the developed data sets. Section 

4 proposes the evaluation metrics. Section 5 

compares results from the various contestants. 

Finally, we conclude this paper with findings and 

offer future research directions in Section 6.

中文拼写检查器相对困难
之所以发展，部分是因为没有单词定界符
在汉语单词和汉语单词之间存在
只能包含一个或多个字符
字符。此外，还有13个以上
千个汉字，而不是只有26个
英文字母，每个字母都有自己的上下文
构成一个有意义的中文单词。所有
这些使中文拼写检查成为一项艰巨的任务。
实证分析表明，中文拼写错误经常是由多个字符单词之间的混淆引起的。
在语音和视觉上相似，但在语义上截然不同（Liu等，2011）。自动
拼写检查器应具有以下两项功能：
识别拼写错误并建议
错误用法的正确字符。的
SIGHAN 2013中国拼写大放异彩
Check是第一个提供数据集的活动
作为绩效评估的基准
中文拼写检查（Wu等，2013）。的
SIGHAN 2013中的数据源自论文
由说中文的人撰写。以下
第一次评估的经验，第二次
在CIPS-SIGHAN联合CLP中举行了审核

第二

在CIPS-SIGHAN CLP-2014联合会议上进行了讨论，重点是论文

由汉语学习者撰写的《外国语》

（CFL）（Yu et al。，2014）。

由于在检测和

纠正CFL学习者书面中的拼写错误

SIGHAN 2015 Bake-off的论文，再次以

中文拼写检查任务，提供评估

ui平台的开发与实现

自动中文拼写检查

ers。给定一段由几句组成的段落

倾向于，检查程序应识别所有

可能的拼写错误，突出显示其位置，

并提出可能的更正。

本文的其余部分安排如下。

第2部分概述了SIGHAN

2015年，中国拼写检查正式开始。秒

第3部分介绍了开发的数据集。部分

图4提出了评估指标。第5节

比较各个参赛者的结果。

最后，我们以结论总结了本文

在第6节中提供未来的研究方向。







收集了我们任务中使用的学习者语料库
从基于计算机的论文部分
对外汉语考试（TOCFL）
在台湾管理。 拼写错误是
由训练有素的中国人手动注释
发言者，他们还会根据每个错误提供更正。 当时的论文
分为三组，如下





lit into three sets as follows

(1) Training Set: this set included 970 se

lected essays with a total of 3,143 spelling errors. 

Each essay is represented in SGML format 

shown in Fig. 1. The title attribute is used to de

scribe the essay topic. Each passage is composed 

of several sentences, and each passage contains 

at least one spelling error, and the data indicates 

both the error’s location and corresponding cor

rection. All essays in this set are used to train the 

developed spelling checker.

(2) Dryrun Set: a total of 39 passages were 

given to participants to familiarize themselves 

with the final testing process. Each participant 

can submit several runs generated using differ

ent models with different parameter settings of 

their checkers. In addition to make sure that the 

submitted results can be correctly evaluated, 

participants can fine-tune their developed mod

els in the dryrun phase. The purpose of dryrun is 

to validate the submitted output format only, and 

no dryrun outcomes were considered in the offi

cial evaluation

(3) Test Set: this set consists of 1,100 testing 

passages. Half of these passages contained no 

spelling errors, while the other half included at 

least one spelling error.  The evaluation was conducted as an open test. In addition to the data sets 

provided, registered participant teams were al

lowed to employ any linguistic and computa

tional resources to detect and correct spelling 

errors. Besides, passages written by CFL learners 

may yield grammatical errors, missing or redun

dant words, poor word selection, or word order

ing problems. The task in question focuses ex

clusively on spelling error correction.



分为三组

（1）训练套：这套包括970 se

精选论文，总共3,143个拼写错误。

每篇论文均以SGML格式表示

如图1所示。title属性用于定义

写下论文主题。每个段落组成

几句话，每个段落包含

至少一个拼写错误，并且数据表明

错误的位置和相应的错误

纠正。这套文章中的所有文章都用来训练

开发了拼写检查器。

（2）Dryrun Set：总共39个段落

给予参与者熟悉自己的知识

最后的测试过程。每个参与者

可以提交使用不同生成的多个运行

具有不同参数设置的实体模型

他们的跳棋。除了确保

提交的结果可以正确评估，

参与者可以微调他们开发的mod

处于干运行阶段。空运行的目的是

仅验证提交的输出格式，并且

官方没有考虑空转结果

社会评价

（3）测试集：此测试集包含1,100个测试

段落。这些段落中有一半没有

拼写错误，而另一半包含

至少一个拼写错误。评估以公开测试的形式进行。除了数据集

提供，已注册的参与者团队

不得使用任何语言和计算机

检测和纠正拼写的国家资源

错误。此外，CFL学习者撰写的文章

可能会产生语法错误，丢失或重新出现

随意的单词，错误的单词选择或单词顺序

问题。有问题的任务集中在

仅在拼写错误更正上。





# 