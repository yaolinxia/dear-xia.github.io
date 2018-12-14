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

#### tf.train.get_or_create_global_step( )

global_step经常在滑动平均，学习速率变化的时候需要用到，这个参数在tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)里面有，系统会自动更新这个参数的值，从1开始。

#### tf.train.AdamOptimizer()

优化器



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

### 2.1 相关函数介绍

#### tf.ConfigProto()

~~~
config = tf.ConfigProto(allow_soft_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
sess = tf.Session(config=config)
~~~

- **tf.ConfigProto(log_device_placement=True)**

  设置tf.ConfigProto()中参数log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的。

- **tf.ConfigProto(allow_soft_placement=True)**

  在tf中，通过命令 "with tf.device('/cpu:0'):",允许手动设置操作运行的设备。如果手动设置的设备不存在或者不可用，就会导致tf程序等待或异常，为了防止这种情况，可以设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。



  为了加快运行效率，TensorFlow在初始化时会尝试分配所有可用的GPU显存资源给自己，这在多人使用的服务器上工作就会导致GPU占用，别人无法使用GPU工作的情况。

  tf提供了两种控制GPU资源使用的方法，一是让TensorFlow在运行过程中动态申请显存，需要多少就申请多少;第二种方式就是限制GPU的使用率。

  **a. 动态申请显存**

~~~python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
~~~

​	**b. 限制GPU使用率**

~~~python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
session = tf.Session(config=config)
~~~

### 2.2 主体部分

~~~python
import os
import tensorflow as tf

from args import args
from data import SingleCharData as Data
# from data import RotationData as Data
from models.single_char_model import Model

class Main:
    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = None

    def run(self, mode):
        self.sess.run(tf.global_variables_initializer())
        if mode in ('train',):
            self.train()
        elif mode in ('infer', 'pred'):
            self.infer()
        else:
            print('%s ??' % mode)

    def train(self):

        model = Model(args['input_width'], args['input_height'], args['num_class'], 'train')
        model.build()
        self.sess.run(tf.global_variables_initializer())
# 获取数据
# dir_val：验证集目录
#
        val_data = Data(args['input_height'], args['input_width'], args['num_class'])\
            .read(args['dir_val'], size=args['val_size'], make_char_map=True)\
            .dump_char_map('label_maps/single_char.json')
        train_data = Data(args['input_height'], args['input_width'], args['num_class'])\
            .load_char_map('label_maps/single_char.json')\
            .read(args['dir_train'], size=args['train_size'], make_char_map=False)\
            .shuffle_indices()
        print('start training')

        if args['restore']:
            self.restore()

        # init tensorboard
        writer = tf.summary.FileWriter("tb")

        # start training
        step = 0
        cost_between_val = 0
        samples_between_val = 0
        batch_size = args['batch_size']
        for itr in range(args['num_epochs']):
            train_data.shuffle_indices()
            train_batch = train_data.next_batch(batch_size)
            while train_batch is not None:
                images, labels = train_batch
                feed_dict = model.feed(images, labels)
                step, loss, _ = self.sess.run([model.step, model.loss, model.train_op],
                                              feed_dict=feed_dict)
                train_batch = train_data.next_batch(batch_size)
                cost_between_val += loss
                samples_between_val += batch_size

                if step % args['save_interval'] == 1:
                    self.save(step)

                if step % args['val_interval'] == 0:
                    print("#%d[%d]\t\t" % (step, itr), end='')

                    val_data.init_indices()
                    val_batch = val_data.next_batch(batch_size)

                    self.sess.run(tf.local_variables_initializer())
                    acc = 0.0
                    val_cost = val_samples = 0
                    while val_batch is not None:
                        val_image, val_labels = val_batch
                        val_feed_dict = model.feed(val_image, val_labels)
                        loss, _acc, acc = self.sess.run(
                            [model.loss, model.val_acc_update_op, model.val_acc],
                            feed_dict=val_feed_dict)
                        val_cost += loss
                        val_samples += batch_size
                        val_batch = val_data.next_batch(batch_size)
                    loss = val_cost / val_samples
                    custom_sm = tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy", simple_value=acc)
                    ])
                    writer.add_summary(custom_sm, step)
                    print("#validation: accuracy=%.6f,\t average_batch_loss:%.4f" % (acc, loss))
                    cost_between_val = samples_between_val = 0
        self.save(step)

    def infer(self):
        model = Model(args['input_width'], args['input_height'], args['num_class'], 'infer')
        model.build()
        self.restore()
        print("start inferring")
        batch_size = args['batch_size']
        infer_data = Data(args['input_height'], args['input_width'], args['num_class'])
        infer_data.read(args['dir_infer'])
        infer_data.init_indices()
        infer_batch = infer_data.next_batch(batch_size)
        self.sess.run(tf.local_variables_initializer())
        while infer_batch is not None:
            infer_images, infer_labels = infer_batch
            infer_feed_dict = model.feed(infer_images, infer_labels)
            classes = self.sess.run([model.classes],
                                    feed_dict=infer_feed_dict)

    def restore(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(args['ckpt'])
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            print('successfully restored from %s' % args['ckpt'])
        else:
            print('cannot restore from %s' % args['ckpt'])

    def save(self, step):
        if self.saver is None:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.saver.save(self.sess, os.path.join(args['ckpt'], '%s_model' % str(args['name'])), global_step=step)
        print('ckpt saved')

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))

            tf.summary.scalar('histogram', var)


def main(_):
    print('using tensorflow', tf.__version__)
    m = Main()
    if args['gpu'] == -1:
        dev = '/cpu:0'
    else:
        dev = '/gpu:%d' % args['gpu']

    # with tf.device(dev):
    #     print('---')
    #     print(dev)
    #     m.run('train')
    m.run('train')


if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    tf.app.run()
~~~

## 三、数据部分

### 3.1 相关函数介绍

#### json.load ()

读写JSON数据

~~~python
import json

data = {
    'name' : 'ACME',
    'shares' : 100,
    'price' : 542.23
}

json_str = json.dumps(data)
~~~

将一个JSON编码的字符串转换回一个Python数据结构：

~~~python
data = json.loads(json_str)
~~~

如果你要处理的是文件而不是字符串，你可以使用 `json.dump()` 和 `json.load()` 来编码和解码JSON数据。例如：

~~~python
# Writing JSON data
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading data back
with open('data.json', 'r') as f:
    data = json.load(f)
~~~

### 3.2 主体部分

~~~python
import os
import re
import json
import cv2 as cv
import numpy as np
from progressbar import ProgressBar


class AbstractData:
    def __init__(self, height, width, num_class):
        self.height, self.width = height, width
        self.images = None
        self.labels = None
        self.indices = None
        self.predictions = None
        self.num_class = num_class

        self.label_map = {}
        self.label_map_reverse = {}

        self.batch_ptr = 0

    def load_char_map(self, file_path):
        print('Loading char map from `%s` ...\t' % file_path, end='')
        with open(file_path, encoding='utf-8') as f:
            self.label_map = json.load(f)
        for k, v in self.label_map.items():
            self.label_map_reverse[v] = k
        print('[done]')
        return self

    def dump_char_map(self, file_path):
        print('Generating char map to `%s` ...\t' % file_path, end='')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, ensure_ascii=False)
        print('[done]')
        return self

    def clear_char_map(self):
        self.label_map_reverse = {}
        self.label_map = {}
        return self

    def read(self, src_root, size=None, make_char_map=False):
        print('loading data...%s' % '' if size is None else ("[%d]" % size))
        images = []
        labels = []
        with ProgressBar(max_value=size) as bar:
            for parent_dir, _, filenames in os.walk(src_root):
                for filename in filenames:
                    lbl = self.filename2label(filename)
                    if make_char_map and lbl not in self.label_map:
                        next_idx = len(self.label_map)
                        self.label_map[lbl] = next_idx
                        self.label_map_reverse[next_idx] = lbl
                    labels.append(self.label_map[lbl])
                    images.append(
                        cv.imdecode(np.fromfile(os.path.join(parent_dir, filename)), 0)
                        .astype(np.float32)
                        .reshape((self.height, self.width, 1)) / 255.
                    )
                    bar.update(bar.value+1)
        self.images = np.array(images)
        self.labels = np.array(labels)
        return self

    def filename2label(self, filename: str):
        # return filename.split('_')[-1].split('.')[0]
        raise Exception('filename2label not implement')

    def shuffle_indices(self):
        samples = self.size()
        self.indices = np.random.permutation(samples)
        self.batch_ptr = 0
        return self

    def init_indices(self):
        samples = self.size()
        self.indices = np.arange(0, samples, dtype=np.int32)
        self.batch_ptr = 0
        return self

    def next_batch(self, batch_size):
        start, end = self.batch_ptr, self.batch_ptr + batch_size
        end = end if end <= len(self.indices) else len(self.indices)
        if start >= self.size():
            return None
        else:
            indices = [self.indices[i] for i in range(start, end)]
            self.batch_ptr = end
            return self.images[indices], self.labels[indices]

    def get(self):
        return self.images, self.labels

    def size(self):
        return self.images.shape[0]

    def get_imgs(self):
        return self.images

    def init_pred_buff(self):
        self.predictions = []

    def buff_pred(self, pred):
        self.predictions += pred


class RotationData(AbstractData):
    def filename2label(self, filename: str):
        _ = filename.split('.')[:-1]
        basename = '.'.join(_)
        angle = round(float(basename.split('_')[1]))
        return angle + self.num_class // 2


class SingleCharData(AbstractData):
    ptn = re.compile("\d+_(\w+)\.(?:jpg|png|jpeg)")

    def filename2label(self, filename: str):
        return SingleCharData.ptn.search(filename).group(1)

~~~




## 参考网址：

- <https://cuiqingcai.com/5715.html>  (各个参数讲的很详细)
- <https://blog.csdn.net/dcrmg/article/details/79091941>(tf.ConfigProto())

- <https://python3-cookbook.readthedocs.io/zh_CN/latest/c06/p02_read-write_json_data.html>(json)