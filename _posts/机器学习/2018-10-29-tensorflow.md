---
layout: post
title: "Tensorflow"
tag: 机器学习
---

## 一、 参数介绍

### 1. 1 epoch，batch size，iteration

梯度下降法是迭代的，也就是说我们需要多次计算结果，最终求得最优解。梯度下降的迭代质量有助于使输出结果尽可能拟合训练数据。在训练模型时，如果训练数据过多，无法一次性将所有数据送入计算，那么我们就会遇到epoch，batchsize，iterations这些概念。为了克服数据量多的问题，我们会选择将数据分成几个部分，即**batch**，进行训练，从而使得每个批次的数据量是可以负载的。将这些batch的数据逐一送入计算训练，更新神经网络的权值，使得网络收敛。

- Epoch

  一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程。由于一个epoch常常太大，计算机无法负荷，我们会将它分成几个较小的batches。

  在训练时，将所有数据迭代训练一次是不够的，需要反复多次才能拟合收敛。在实际训练时，我们将所有数据分成几个batch，每次送入一部分数据，梯度下降本身就是一个迭代过程，所以单个epoch更新权重是不够的。

- Batch Size

  所谓Batch就是每次送入网络中训练的一部分数据，而Batch Size就是每个batch中训练样本的数量.

- Iterations

  所谓iterations就是完成一次epoch所需的batch个数。batch numbers就是iterations。

  简单一句话说就是，我们有2000个数据，分成4个batch，那么batch size就是500。运行所有的数据进行训练，完成1个epoch，需要进行4次iterations。

### 1.2 Tensor

TensorFlow中数据的中心单元是**张量**。 张量由一组原始值组成，这些原始值被组织成形状为任意维度的数组。 张量的**秩**是其维数。 以下是一些张量的例子：

~~~python
# 一个秩为0的张量；这是一个形状为[]的标量
[1., 2., 3.] # 一个秩为1的张量；这是一个形状为[3]的向量
[[1., 2., 3.], [4., 5., 6.]] # 一个秩为2的张量；这是一个形状为[2, 3]的矩阵
[[[1., 2., 3.]], [[7., 8., 9.]]] # 一个秩为3、形状为[2, 1, 3]的张量
~~~

### 1.3 计算图

排列成**节点**的一系列Tensorflow操作。

每个**节点**将零个或多个张量作为输入，并生成张量作为输出。

- 构建计算图
- 运行计算图

~~~python
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # 隐式地表示同样是tf.float32
print(node1, node2)
=====
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
~~~

为了在Python中进行有效的数值计算，我们通常使用像[NumPy](http://www.numpy.org/)这样的库，这些库执行昂贵的操作，例如Python之外的矩阵乘法，使用以另一种语言实现的高效代码。 不幸的是，每次操作都会返回Python，但仍然会有很多开销。 如果您想要在GPU上运行计算或以分布式方式运行计算，则这种开销尤其糟糕，因为在这种情况下，传输数据的成本很高。

TensorFlow也在Python之外进行繁重的工作，但为了避免这种开销，它需要更进一步。 TensorFlow不是只独立于Python运行一个单一的昂贵操作，而是让我们描述完全一个完全在Python之外运行的交互操作的图。 这种方法类似于Theano或Torch中使用的方法。

因此，Python代码的作用是构建这个外部计算图，并指定应该运行计算图的哪些部分。 有关详细信息，请参阅[TensorFlow 入门](https://yiyibooks.cn/__trs__/yiyi/tensorflow_13/get_started/get_started.md)中的[计算图](https://yiyibooks.cn/__trs__/yiyi/tensorflow_13/get_started/get_started.md#the_computational_graph)部分。

## 二、 相关函数

### 2.1 tf.placeholder()

~~~python
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_class])
~~~

请注意，打印节点不会像你预期的那样输出值`3.0`和`4.0`。 相反，它们是在求值时分别产生3.0和4.0的节点。 为了实际评估节点，我们必须在**会话**内运行计算图。 会话封装了TensorFlow运行时的控制和状态。

以下代码创建一个`会话`对象，然后调用其`run`方法运行足够的计算图来评估`node1`和`node2`。 通过在会话中运行计算图如下：

~~~python
sess = tf.Session()
print(sess.run([node1, node2]))
~~~

~~~python 
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
======================================
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
~~~

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_add.png)

就像这样，这个图并不是特别引人注目，因为它总是产生一个恒定的结果。 可以将图形参数化为接受外部输入，称为**placeholder**。 **占位符**是稍后提供值的承诺。

~~~
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + 提供tf.add(a, b)的一个快捷方式
~~~

前面的三行有点像函数或lambda，我们在其中定义两个输入参数（a和b），然后对它们进行操作。 我们可以通过使用[run 方法](https://www.tensorflow.org/api_docs/python/tf/Session#run)的feed_dict参数为占位符提供具体值，从而对多个输入进行求值。

~~~python
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
7.5
[ 3.  7.]
~~~

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_adder.png)

~~~python
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
====
22.5
~~~

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_triple.png)

在机器学习中，我们通常需要一个可以进行任意输入的模型，比如上面的模型。 为了使模型可训练，我们需要修改图形以获得具有相同输入的新输出。 **Variables**允许我们向图中添加可训练的参数。 它们由一个类型和初始值构建而成：

```
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

当你调用`tf.constant`时，常量被初始化，并且它们的值永远不会改变。 相比之下，当您调用`tf.Variable`时，变量不会被初始化。 要初始化TensorFlow程序中的所有变量，您必须显式调用特殊操作，如下所示：

```
init = tf.global_variables_initializer()
sess.run(init)
```

重要的是要注意到`init`是TensorFlow子图的句柄，它初始化所有全局变量。 在我们调用`sess.run`之前，变量未初始化。

由于`x`是一个placeholder，我们可以对`x`的几个值同时计算`linear_model`，如下所示：

```
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
```

产生输出

```
[ 0.          0.30000001  0.60000002  0.90000004]
```

我们已经创建了一个模型，但我们不知道它有多好。 为了评估训练数据模型，我们需要一个`y`占位符来提供所需的值，我们需要编写一个损失函数。

损失函数用于衡量当前模型距离提供的数据有多远。 我们将使用线性回归的标准损失模型，它将当前模型与提供的数据之间的增量的平方相加。 `linear_model - y` 创建一个向量，其中每个元素都是对应的样本的错误差值。 我们调用`tf.square`来平方该误差。 然后，我们使用`tf.reduce_sum`求和所有平方误差来创建一个单一的标量，它抽取出所有样本的误差：

```
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```

产生损失值

```
23.66
```

### 2.2 tf.square()

```
tf.square( x,
    name=None)
# 计算 x 元素的平方。
```

### 2.3 tf.reduce_sum()

压缩求和，一般用于降维

```python
x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x)  # 6求和
tf.reduce_sum(x, 0)  # [2, 2, 2] 按列求和
tf.reduce_sum(x, 1)  # [3, 3] 按照行求和
tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]] 按照行的维度求和
tf.reduce_sum(x, [0, 1])  # 6 行列求和
```

### 2.4 tf.train API

机器学习的完整讨论超出了本教程的范围。 但是，TensorFlow提供了**优化器**，可以缓慢更改每个变量以最大限度地减少损失函数。 最简单的优化器是**渐变下降**。 它根据相对于该变量的损失导数的大小来修改每个变量。 一般来说，手动计算符号派生是繁琐且容易出错的。 因此，TensorFlow可以使用函数`tf.gradients`自动生成衍生物，只给出模型的描述。 为了简单起见，优化程序通常会为您执行此操作。 例如，

```
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # 重置为不正确的默认值。
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
```

得到最终的模型参数：

```
[array([-0.9999969], dtype=float32), array([ 0.99999082],
 dtype=float32)]
```

现在我们完成了实际的机器学习！ 尽管做这种简单的线性回归并不需要太多的TensorFlow核心代码，但更复杂的模型和方法将数据提供给模型需要更多的代码。 因此TensorFlow为常见的模式，结构和功能提供更高层次的抽象。 我们将在下一节中学习如何使用这些抽象。

### 2.5 完整的程序

完整的可训练线性回归模型如下所示：

```
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

运行时，它会产生

```
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

请注意，损失是非常小的数字（非常接近零）。 如果你运行这个程序，你的损失可能不完全相同，因为模型是用伪随机值初始化的。

这个更复杂的程序仍然可以在TensorBoard中可视化 ![TensorBoard final model visualization](https://www.tensorflow.org/images/getting_started_final.png)

### 2.6 tf.estimator()

`tf.estimator` is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:

- 运行训练循环
- 运行评估循环
- 管理数据集

tf.estimator定义了许多通用模型。

**可以用于简化线性回归程序**

~~~python
import tensorflow as tf
# NumPy 经常用于加载、操作和预处理数据。
import numpy as np

# 声明特征列表。 我们只有一个数字特征。 有许多
# 其它类型的列，它们更复杂而且更有用。
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). 有许多预定义类型，如线性回归、
# 线性分类以及许多神经网络分类器和回归器。
# 下面的代码给出一个estimator，它实现线性回归。
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow 提供许多辅助方法来读取和建立数据集。
# 这里我们使用两个数据集：一个用于训练，一个用于评估。
# 我们必须告诉函数
# 我们想要数据的多少个batch(num_epochs)以及每个batch的大小。
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# 这里我们评估我们的模型表现如何
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
运行时，它会产生

train metrics: {'loss': 1.2712867e-09, 'global_step': 1000}
eval metrics: {'loss': 0.0025279333, 'global_step': 1000}
请注意我们的评估数据的损失有点高，但仍然接近于零。 这意味着我们正在正确地学习。

定制模型
tf.estimator不会将您锁定到其预定义的模型中。 假设我们想创建一个没有内置到TensorFlow中的自定义模型。 我们仍然可以维持对数据集、feeding、训练等的高层抽象。 tf.estimator。 为了说明，我们将使用我们对较低级别TensorFlow API的了解，展示如何实现我们自己的与LinearRegressor等效模型。

To define a custom model that works with tf.estimator, we need to use tf.estimator.Estimator. tf.estimator.LinearRegressor is actually a sub-class of tf.estimator.Estimator. 不用子类化Estimator，我们仅仅提供Estimator一个函数model_fn，该函数告诉tf.estimator如何评估预测、训练步骤和损失。 代码如下：

import numpy as np
import tensorflow as tf

# 声明特征列表，我们只有一个实数特征
def model_fn(features, labels, mode):
  # 构建一个线性模型并预测值
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
~~~

```
import tensorflow as tf
# NumPy 经常用于加载、操作和预处理数据。
import numpy as np

# 声明特征列表。 我们只有一个数字特征。 有许多
# 其它类型的列，它们更复杂而且更有用。
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). 有许多预定义类型，如线性回归、
# 线性分类以及许多神经网络分类器和回归器。
# 下面的代码给出一个estimator，它实现线性回归。
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow 提供许多辅助方法来读取和建立数据集。
# 这里我们使用两个数据集：一个用于训练，一个用于评估。
# 我们必须告诉函数
# 我们想要数据的多少个batch(num_epochs)以及每个batch的大小。
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# 这里我们评估我们的模型表现如何
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```

运行时，它会产生

```
train metrics: {'loss': 1.2712867e-09, 'global_step': 1000}
eval metrics: {'loss': 0.0025279333, 'global_step': 1000}
```

请注意我们的评估数据的损失有点高，但仍然接近于零。 这意味着我们正在正确地学习。

### 2.7 定制模型

`tf.estimator`不会将您锁定到其预定义的模型中。 假设我们想创建一个没有内置到TensorFlow中的自定义模型。 我们仍然可以维持对数据集、feeding、训练等的高层抽象。 `tf.estimator`。 为了说明，我们将使用我们对较低级别TensorFlow API的了解，展示如何实现我们自己的与`LinearRegressor`等效模型。

To define a custom model that works with `tf.estimator`, we need to use `tf.estimator.Estimator`.`tf.estimator.LinearRegressor` is actually a sub-class of `tf.estimator.Estimator`. 不用子类化`Estimator`，我们仅仅提供`Estimator`一个函数`model_fn`，该函数告诉`tf.estimator`如何评估预测、训练步骤和损失。 代码如下：

```python
import numpy as np
import tensorflow as tf

# 声明特征列表，我们只有一个实数特征
def model_fn(features, labels, mode):
  # 构建一个线性模型并预测值
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```

运行时，它会产生

```
train metrics: {'loss': 1.227995e-11, 'global_step': 1000}
eval metrics: {'loss': 0.01010036, 'global_step': 1000}
```

请注意，自定义`model_fn()`函数的内容与我们采用底层API的手动模型训练循环非常相似。

### 2.8 tf.train.Saver()

管理模型中的所有变量。可以将变量保存到检查点文件中。

~~~python
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
~~~

### 2.9 tf.logging.set_verbosity日志消息

在tensorflow中有函数可以直接log打印，这个跟ROS系统中打印函数差不多。

TensorFlow使用五个不同级别的日志消息。 按照上升的顺序，它们是DEBUG，INFO，WARN，ERROR和FATAL。 当您在任何这些级别配置日志记录时，TensorFlow将输出与该级别相对应的所有日志消息以及所有级别的严重级别。 例如，如果设置了ERROR的日志记录级别，则会收到包含ERROR和FATAL消息的日志输出，如果设置了一个DEBUG级别，则会从所有五个级别获取日志消息。    # 默认情况下，TENSFlow在WARN的日志记录级别进行配置，但是在跟踪模型训练时，您需要将级别调整为INFO，这将提供适合操作正在进行的其他反馈。

### 2.10 tf.app.run()

理flag解析，然后执行main函数，那么flag解析是什么意思呢？诸如这样的：

~~~python
tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS
~~~

- 如果你的代码中的入口函数不叫main()，而是一个其他名字的函数，如test()，则你应该这样写入口`tf.app.run(test())`
- 如果你的代码中的入口函数叫main()，则你就可以把入口写成`tf.app.run()`

### 2.11 tf.device(dev)

使用 **tf.device()** 指定模型运行的具体设备，可以指定运行在GPU还是CUP上，以及哪块GPU上。

**设置使用GPU**

使用 tf.device('/gpu:1') 指定Session在第二块GPU上运行

## 三、模型的训练

### 3.1 定义指标

- 定义指标，衡量模型是好的还是坏的
- 指标称为成本（成本）或者损失（loss）
- 然后需要最小化这个指标

### 3.2 交叉熵

- 一个非常漂亮的成本函数

- 产生于信息论里面的信息压缩编码技术，后来演变称为机器学习中的重要技术手段

  ![1545220617372](C:\Users\yao\AppData\Roaming\Typora\typora-user-images\1545220617372.png)

  > y：预测的概率分布
  >
  > y'：实际的分布（eg：one-hot）

- 为了计算交叉熵，需要添加一个新的占位符

  ~~~
  y_ = tf.placeholder("float", [None,10])
  ~~~

- 计算交叉熵

  ```python
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  ```

  首先，用 `tf.log` 计算 `y` 的每个元素的对数。接下来，我们把 `y_` 的每一个元素和 `tf.log(y_)` 的对应元素相乘。最后，用 `tf.reduce_sum` 计算张量的所有元素的总和。（注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。

### 3.3 自动使用反向传播

- 确定你的变量是如何影响你想要最小化的那个成本值

- TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。

  ~~~
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  ~~~

  在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。当然TensorFlow也提供了[其他许多优化算法](http://www.tensorfly.cn/tfdoc/api_docs/python/train.html#optimizers)：只要简单地调整一行代码就可以使用其他的算法。

### 3.4 初始化变量

现在，我们已经设置好了我们的模型。在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：

~~~python
init = tf.initialize_all_variables()
~~~

在一个`Session`里面启动我们的模型，并且初始化变量：

~~~python
sess = tf.Session()
sess.run(init)
~~~

### 3.5 设置训练次数

- eg：然后开始训练模型，这里我们让模型循环训练1000次！

~~~python
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
~~~

- 该循环的每个步骤，会随机抓取训练数据中的100个批处理数据点。
- 然后用这个数据点，作为参数替换之前的占位符来运行`train_step`
- 使用一小部分的随机数据来进行训练被称为**随机训练**（stochastic training）- 在这里更确切的说是随机梯度下降训练。
- 在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。
- 所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。

## 四、模型评估

### 4.1 找出预测正确的标签

- `tf.argmax` 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。

- 由于标签向量是由0,1组成，因此**最大值1**所在的索引位置就是类别标签，比如`tf.argmax(y,1)`返回的是模型对于任一输入x预测到的标签值，而 `tf.argmax(y_,1)` 代表正确的标签，我们可以用 `tf.equal`来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

  ~~~python
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  ~~~

- 上述代码，输出一组布尔值，为了确定正确预测项的比例，可以把布尔值转换成浮点数，然后再去平均值。

- 比如，`[True, False, True, True]` 会变成 `[1,0,1,1]` ，取平均值后得到 `0.75`.

  ~~~python
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  ~~~

- 将学习到的模型在测试数据集上看下正确率

  ~~~python
  print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  ~~~

## 参考网址

- <https://www.jianshu.com/p/e5076a56946c>
- <https://yiyibooks.cn/yiyi/tensorflow_13/get_started/get_started.html>