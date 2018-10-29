---
layout: post
title: "CNN单字模型预测"
tag: OCR
---

## 一、 模型构建

### 1.1 相关函数介绍

####  x = tf.layers.conv2d()     2D 卷积层的函数接口

这个层创建了一个卷积核，将输入进行卷积来输出一个 tensor。如果 `use_bias` 是 `True`（且提供了 `bias_initializer`），则一个偏差向量会被加到输出中。最后，如果 `activation` 不是 `None`，激活函数也会被应用到输出中。

~~~
 conv2d(inputs, filters, kernel_size, 
    strides=(1, 1), 
    padding='valid', 
    data_format='channels_last', 
    dilation_rate=(1, 1),
    activation=None, 
    use_bias=True, 
    kernel_initializer=None,
    bias_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x000002596A1FD898>, 
    kernel_regularizer=None,
    bias_regularizer=None, 
    activity_regularizer=None, 
    kernel_constraint=None, 
    bias_constraint=None, 
    trainable=True, 
    name=None,
    reuse=None)

~~~

- 参数
  inputs：Tensor 输入

  filters：整数，表示输出空间的维数（即卷积过滤器的数量）

  kernel_size：一个整数，或者包含了两个整数的元组/队列，表示卷积窗的高和宽。如果是一个整数，则宽高相等。

  **strides：**一个整数，或者包含了两个整数的元组/队列，表示卷积的纵向和横向的步长。如果是一个整数，则横纵步长相等。另外， strides 不等于1 和 dilation_rate 不等于1 这两种情况不能同时存在。

  **padding：**"valid" 或者 "same"（不区分大小写）。"valid" 表示不够卷积核大小的块就丢弃，"same"表示不够卷积核大小的块就补0。 
  "valid" 的输出形状为 
  ![middle](https://ws1.sinaimg.cn/large/e93305edgy1fwo0shvyvsj207001wmwz.jpg)

  "valid" 的输出形状为 
  ![](https://ws1.sinaimg.cn/large/e93305edgy1fwo0t42d7ej204n0203yb.jpg)
  dilation_rate：一个整数，或者包含了两个整数的元组/队列，表示使用扩张卷积时的扩张率。如果是一个整数，则所有方向的扩张率相等。另外， strides 不等于1 和 dilation_rate 不等于1 这两种情况不能同时存在。

  activation：激活函数。如果是None则为线性函数。

  use_bias：Boolean类型，表示是否使用偏差向量。

  **kernel_initializer：**卷积核的初始化。

  bias_initializer：偏差向量的初始化。如果是None，则使用默认的初始值。

  kernel_regularizer：卷积核的正则项

  bias_regularizer：偏差向量的正则项

  activity_regularizer：输出的正则函数

  kernel_constraint：映射函数，当核被Optimizer更新后应用到核上。Optimizer 用来实现对权重矩阵的范数约束或者值约束。映射函数必须将未被影射的变量作为输入，且一定输出映射后的变量（有相同的大小）。做异步的分布式训练时，使用约束可能是不安全的。

  bias_constraint：映射函数，当偏差向量被Optimizer更新后应用到偏差向量上。

  trainable：Boolean类型。

  name：字符串，层的名字。

  reuse：Boolean类型，表示是否可以重复使用具有相同名字的前一层的权重。

- 返回值
  输出 Tensor

- 异常抛出

  ValueError：if eager execution is enabled.

#### tf.layers.batch_normalization()

​	作用：**归一化**

​	目的：**提高速度**

​	详见：<https://blog.csdn.net/NNNNNNNNNNNNY/article/details/70331796>

#### **tf.layers.max_pooling2d**()

~~~python
max_pooling2d(
    inputs,  #进行池化的数据
    pool_size,	#池化的核大小
    strides,	#池化的滑动步长
    padding='valid',	
    data_format='channels_last',
    name=None	#层的名字
)
~~~

#### x = tf.layers.flatten()

​	Flatten层，即把一个 Tensor 展平

#### tf.layers.dense()

~~~python
dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
~~~

- **inputs**: 输入数据，2维tensor.
- **units**: 该层的神经单元结点数。
- **activation**: 激活函数.
- **use_bias**: Boolean型，是否使用偏置项.
- **kernel_initializer**: 卷积核的初始化器.
- **bias_initializer**: 偏置项的初始化器，默认初始化为0.
- **kernel_regularizer**: 卷积核化的正则化，可选.
- **bias_regularizer**: 偏置项的正则化，可选.
- **activity_regularizer**: 输出的正则化函数.
- **trainable**: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中 `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
- **name**: 层的名字.
- **reuse**: Boolean型, 是否重复使用参数.

#### tf.argmax()

​	tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。

​	tf.argmax(input, axis=None, name=None, dimension=None)，此函数是对矩阵按行或列计算最大值。

- **input:** 输入Tensor
- **axis:**0表示列，1表示行
- **name:**名称
- **demension:**默认axis取值优先，新加的字段

#### tf.get_collection(）

​	从一个结合中取出全部变量，是一个列表。tf.get_collection(key, scope=None) 用来获取一个名称是‘key’的集合中的所有元素，返回的是一个列表，列表的顺序是按照变量放入集合中的先后;   scope参数可选，表示的是名称空间（名称域），如果指定，就返回名称域中所有放入‘key’的变量的列表，不指定则返回所有变量。

#### tf.control_dependencies（）

​	用来控制计算流图

### 1.2 主体部分

~~~python
class Model:

    def __init__(self, input_width, input_height, num_class, mode):
        self.input_width = input_width
        self.input_height = input_height
        self.num_class = num_class
        self.training = mode.lower() in ('train',)

        self.images = tf.placeholder(tf.float32, [None, input_height, input_width, 1], name='input_img_batch')
        self.labels = tf.placeholder(tf.int32, [None], name='input_lbl_batch')

        # define op
        self.step = None
        self.loss = None
        self.classes = None
        self.train_op = None
        self.val_acc = self.val_acc_update_op = None

    def feed(self, images, labels):
        return {
            self.images: images,
            self.labels: labels
        }
    
    def build(self):
        images = self.images
        labels = self.labels

        input_layer = tf.reshape(images, [-1, self.input_height, self.input_width, 1])

        # cnn block 1
        x = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2)

        # cnn block 2
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 3
        x = tf.layers.conv2d(
            inputs=x,
            filters=128,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 4
        x = tf.layers.conv2d(
            inputs=x,
            filters=256,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=2
        )

        # dense
        x = tf.layers.flatten(
            inputs=x
        )

        x = tf.layers.dense(
            inputs=x,
            units=8192
        )

        logits = tf.layers.dense(
            inputs=x,
            units=self.num_class
        )
        
        # probabilities = tf.nn.softmax(logits, name='P')
        self.classes = tf.argmax(input=logits, axis=1, name='class')
        self.step = tf.train.get_or_create_global_step()

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(
                loss=self.loss,
                global_step=self.step
            )

        self.val_acc, self.val_acc_update_op = tf.metrics.accuracy(labels, self.classes)

        return self
~~~



### 1.3 参考网址

- <https://blog.csdn.net/HappyRocking/article/details/80243790>

- <https://cuiqingcai.com/5715.html>
- 

## 二、 模型训练

![](https://ws1.sinaimg.cn/large/e93305edgy1fwp34osf60j21hc0u0x6r.jpg)

