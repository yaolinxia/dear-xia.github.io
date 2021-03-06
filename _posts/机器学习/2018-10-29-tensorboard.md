---
layout: post
title: "Tensorboard"
tag: 机器学习
---

## 一、简介

​	TensorFlow 可用于训练大规模深度神经网络所需的计算，使用该工具涉及的计算往往复杂而深奥。为了更方便 TensorFlow 程序的理解、调试与优化，我们发布了一套名为 TensorBoard 的可视化工具。您可以用 TensorBoard 来展现 TensorFlow 图，绘制图像生成的定量指标图以及显示附加数据（如其中传递的图像）。



## 二、目的

​	可以显现tensorflow的图表结构，计算结构，便于找出错误。



## 三、使用

![](https://ws1.sinaimg.cn/large/e93305edgy1fwoxl8rnb9j20su0h9wlr.jpg)

![1540783988459](C:\Users\yao\AppData\Roaming\Typora\typora-user-images\1540783988459.png)

![](https://ws1.sinaimg.cn/large/e93305edgy1fwoxssz880j20xw0f1qic.jpg)

![](https://ws1.sinaimg.cn/large/e93305edgy1fwoxtfm878j210w0ch7fm.jpg)

不同数值的分布情况

![](https://ws1.sinaimg.cn/large/e93305edgy1fwoxujj3bzj20ho057ac1.jpg)

![](https://ws1.sinaimg.cn/large/e93305edgy1fwoxv362fpj20za09iqap.jpg)

![](https://ws1.sinaimg.cn/large/e93305edgy1fwoxvie10wj20oe0e4jw2.jpg)

- 协议缓冲区，然后把缓冲区递给FileWriter, 将它们写入硬盘

![](https://ws1.sinaimg.cn/large/e93305edgy1fwoxxszjdyj20hq0am44v.jpg)

### 项目案例

~~~
# 定义存放目录的位置
writer = tf,summary.FileWriter("tb“)
~~~



## **四、Docker中使用tensorboard**

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/2019-04-17 22-01-28屏幕截图.png)

**需要完成Docker到宿主机的映射**

**参考网址：**

- <https://www.cnblogs.com/dyufei/p/8094507.html>
- <http://data4q.com/2017/07/22/%E8%AE%BF%E9%97%AEdocker%E5%AE%B9%E5%99%A8%E4%B8%ADtensorboard/>
- <https://blog.csdn.net/qq_20909377/article/details/79103113>

**该方法有问题**

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/2019-04-18 10-02-09屏幕截图.png)

> 不需要再重新创建镜像



