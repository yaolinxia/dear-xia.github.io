---
layout: post
title: "pycorrector"
tag: 	文本纠错

---

# **seq2seq_attention**

## **config.py**

- 设置训练数据所在路径

- 设置结果存放路径
- 数据处理后训练，测试集路径
- 模型存储位置

~~~python
pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# chinese corpus
raw_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

result_path = os.path.join(output_dir, "result.txt")
# seq2seq_attn_train config
save_vocab_path = os.path.join(output_dir, 'vocab.txt')
attn_model_path = os.path.join(output_dir, 'attn_model.weight')
~~~

- 训练相关参数

~~~python
batch_size = 64
epochs = 40
rnn_hidden_dim = 128
maxlen = 400
dropout = 0.0
use_gpu = False
~~~

## **corpus_reader**

- 读取并且处理原始数据，存进train.txt，test.txt
- train.py中调用此模块

## **train**.py

~~~python
class Seq2seqAttnModel(object):
    def __init__(self, chars, hidden_dim=128, attn_model_path=None, use_gpu=False, dropout=0.2):
        self.chars = chars
        self.hidden_dim = hidden_dim
        self.model_path = attn_model_path
        self.use_gpu = use_gpu
        self.dropout = dropout

    def build_model(self):
        # 搭建seq2seq模型
        x_in = Input(shape=(None,))
        y_in = Input(shape=(None,))
        print("x:",x_in)
        print("===============================")
        print("y:", y_in)
        x = x_in
        y = y_in
        # 为了训练的时候，方便一次性都放进去进行训练，提高效率
        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)
        print("x_mask:", x_mask)
        print("y_mask:", y_mask)

        x_one_hot = Lambda(self._one_hot)([x, x_mask])
        x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（target的字词很可能在input出现过）

        # embedding
        embedding = Embedding(len(self.chars), self.hidden_dim) # 其中一个是指input_dim, 另一个参数是指output_dim
        x = embedding(x)
        y = embedding(y)

        # encoder，双层双向GRU; decoder，双层单向GRU
        if self.use_gpu:
            # encoder
            x = Bidirectional(CuDNNGRU(int(self.hidden_dim / 2), return_sequences=True))(x)
            x = Bidirectional(CuDNNGRU(int(self.hidden_dim / 2), return_sequences=True))(x)
            # decoder
            y = CuDNNGRU(self.hidden_dim, return_sequences=True)(y)
            y = CuDNNGRU(self.hidden_dim, return_sequences=True)(y)
        else:
            # encoder
            x = Bidirectional(GRU(int(self.hidden_dim / 2), return_sequences=True, dropout=self.dropout))(x)
            x = Bidirectional(GRU(int(self.hidden_dim / 2), return_sequences=True, dropout=self.dropout))(x)
            # decoder
            y = GRU(self.hidden_dim, return_sequences=True, dropout=self.dropout)(y)
            y = GRU(self.hidden_dim, return_sequences=True, dropout=self.dropout)(y)

        xy = Interact()([y, x, x_mask])
        xy = Dense(512, activation='relu')(xy)
        xy = Dense(len(self.chars))(xy)
        xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
        xy = Activation('softmax')(xy)

        # 交叉熵作为loss，但mask掉padding部分
        cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
        loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

        model = Model([x_in, y_in], xy)
        model.add_loss(loss)
        model.compile(optimizer=Adam(1e-3))
        if os.path.exists(self.model_path):
            model.load_weights(self.model_path)
        return model

    def _one_hot(self, x):
        """
        输出 one hot 向量
        :param x:
        :return:
        """
        x, x_mask = x
        x = K.cast(x, 'int32')
        x = K.one_hot(x, len(self.chars))
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')
        return x

~~~

## **infer.py**







# **出现问题**

##  **运行train.py时**

~~~
Using TensorFlow backend.
Read data, path:/Users/stone/PycharmProjects/pycorrector/pycorrector/seq2seq_attention/output/train.txt
Read data, path:/Users/stone/PycharmProjects/pycorrector/pycorrector/seq2seq_attention/output/test.txt
WARNING:tensorflow:From /Users/stone/anaconda3/envs/tensorflow_36/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /Users/stone/anaconda3/envs/tensorflow_36/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /Users/stone/anaconda3/envs/tensorflow_36/lib/python3.6/site-packages/tensorflow/python/ops/math_grad.py:102: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
Epoch 1/40
2019-11-01 21:45:20.999811: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-11-01 21:45:21.000020: I tensorflow/core/common_runtime/process_util.cc:71] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
~~~

### 解决方法

- <https://github.com/dmlc/xgboost/issues/1715>: 此方法未能解决问题

- **解决：**<https://qiita.com/161abcd/items/6ddf76366bc30c79522f>

  ~~~
  import os
  os.environ['KMP_DUPLICATE_LIB_OK']='True'
  ~~~

  
