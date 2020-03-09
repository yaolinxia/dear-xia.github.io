---
layout: post
title: "tesseract使用"
tag: OCR
---

# **登录服务器**

~~~
ssh root@192.168.68.38
~~~

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/2019-03-06  15-25-13屏幕截图.png)

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/2019-03-06 15-24-17屏幕截图.png)

![ ](https://raw.githubusercontent.com/yaolinxia/img_resource/master/multi-input attention/2019-03-06 15-24-52屏幕截图.png)

# **显示出所有的容器**

~~~
docker container ls -a
~~~

- tesseract4re 就是我们所要安装的容器



# **启动所要用的Docker**

~~~
docker exec -it 1bc8****  /bin/bash
~~~

- 执行bash 命令





# 指定语言模型

**基础用法**

tesseract <imagename> <outputbase>

```text
#解析test4.png图片，生成的文字会放入gemfield.txt文件中
root@gemfield:# tesseract test4.png gemfield
Tesseract Open Source OCR Engine v4.0.0-beta.1-218-g2645 with Leptonica

#将解析结果打印到屏幕上
root@gemfield:# tesseract test4.png stdout

#指定语言包模型
root@gemfield:# tesseract test4.png stdout -l chi_sim
```

# **ubuntu 安装中文输入法**

~~~
sudo apt-get install ibus-pinyin
~~~

- 要先安装拼音

- <https://blog.csdn.net/zhangchao19890805/article/details/52743380>

~~~
打开/etc/environment
在下面添加如下两行
LANG=”zh_CN.UTF-8″
LANGUAGE=”zh_CN:zh:en_US:en”

打开 /var/lib/locales/supported.d/local
添加zh_CN.GB2312字符集，如下：
en_US.UTF-8 UTF-8
zh_CN.UTF-8 UTF-8
zh_CN.GBK GBK
zh_CN GB2312
保存后，执行命令：
sudo locale-gen

打开/etc/default/locale
修改为：
LANG=”zh_CN.UTF-8″
LANGUAGE=”zh_CN:zh:en_US:en”
~~~

- 上述方法，试了一下，行不通

# **参考网址**

- <https://zhuanlan.zhihu.com/p/36397839>