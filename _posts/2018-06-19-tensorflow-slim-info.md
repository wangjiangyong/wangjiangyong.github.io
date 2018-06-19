---
layout: post
title: "Tensorflow Slim 使用"
category: 深度学习
tags: [DeepLearning,Tensorflow]
---

#### TensorFlow-Slim
TF-Slim是TensorFlow中定义、训练和验证复杂模型的轻量级库。tf-slim组件可以自由地与原生TensorFlow混用，当然也包括其他框架，比如tf.contrib.learn。

##### 用法
import tensorflow.contrib.slim as slim

##### Why选用TF-Slim
+ 减少样板代码，使定义模型更加简单，紧凑。这主要是通过argument scoping和大量上层定义的层和变量实现。提高了代码可读性和可维护性，降低了拷贝超参数带来错误的可能性，并简化了超参数调整。
+ 提供常用的正则项，简化模型的开发。
+ 在Slim中几种广泛使用的CV模型（例如，VGG，AlexNet）已经开发出来，用户可以直接使用。这些模型可以作为黑盒使用，也可以以各种方式扩展。
+ Slim使扩展复杂模型变得很容易，可以使用预先存在的模型检查点(pre-existing model checkpoints)来热启动训练算法。

##### TF-Slim的各种组件
TF-Slim被设计为独立存在的几部分组成。包括如下主要部分：
arg_scope data evaluation layers learning losses metrics nets queues regularizers variables


#### 定义模型
使用TF-Slim的variables, layers and scopes可以方便地定义模型。

##### Variables
原生TensorFlow中创建Variables需要预先定义值或者初始化机制（比如高斯随机采样）。为了减少创建变量的代码量，TF-Slim提供了一系列简单封装函数。
例如，创建weights变量，使用截断正态分布初始化，使用l2_loss正则化并放置在CPU上，只用使用如下语句：
```python
weights = slim.variable('weights',
                             shape=[10, 10, 3 , 3],
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             regularizer=slim.l2_regularizer(0.05),
                             device='/CPU:0')
```

原生TF中有两类变量：普通变量(regular variables)和局部变量(local (transient) variables)。大部分变量都是普通变量，一旦创建就可以使用saver将其保存在硬盘上。
局部变量只存在于会话期间并且不保存在硬盘上。
TF-Slim进一步定义model variables，此变量表示模型参数，训练和导入的模型参数。Non-model variables指在训练和验证中所有其他变量，他们在实际执行推断时不再需要。
比如global_step在训练和验证时候使用，但它实际不是模型中的一部分。

```python
# Model Variables
weights = slim.model_variable('weights',
                              shape=[10, 10, 3 , 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
model_variables = slim.get_model_variables()

# Regular variables
my_var = slim.variable('my_var',
                       shape=[20, 1],
                       initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()
```

当使用TF-Slim层(TF-Slim's layers)或slim.model_variable创建model variable时，TF-Slim将会把变量加入到tf.GraphKeys.MODEL_VARIABLES集合(collection)。当使用自己定义的层
或者变量创建的方法，并使用TF-Slim来管理你的模型变量时，TF-Slim提供了将模型变量加入集合的便捷方法：
```python
my_model_variable = CreateViaCustomCode()

# Letting TF-Slim know about the additional variable.
slim.add_model_variable(my_model_variable)
```

##### Layers
虽然TF有大量的运算操作集，但神经网络开发者通常根据"layers", "losses", "metrics",和"networks"高层次的概念来考虑模型。比如卷积层、全连接层或者BatchNorm层都是比TF运算操作更抽象。
而且不像更原生的操作，网络层一般还包含相关的变量。比如卷积层由如下一系列低级操作组成：
1. 创建weight和bias变量
2. 使用上层网络的输出作为输入进行卷积运算
3. 卷积结果加上偏移值
4. 使用激活函数

使用原生TF代码，这将比较繁琐：
```python
input = ...
with tf.name_scope('conv1_1') as scope:
  kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(bias, name=scope)
```

为了减少这些重复的代码，在神经网络层的更抽象层次上TF-Slim提供了许多方便的操作。比如如上代码可以简化为：
```python
input = ...
net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')
```

TF-Slim为构建神经网络的众多组件提供了标准实现。包括：

Layer | TF-Slim
------- | --------
BiasAdd  | [slim.bias_add](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
BatchNorm  | [slim.batch_norm](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Conv2d | [slim.conv2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Conv2dInPlane | [slim.conv2d_in_plane](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Conv2dTranspose (Deconv) | [slim.conv2d_transpose](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
FullyConnected | [slim.fully_connected](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
AvgPool2D | [slim.avg_pool2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Dropout| [slim.dropout](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
Flatten | [slim.flatten](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
MaxPool2D | [slim.max_pool2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
OneHotEncoding | [slim.one_hot_encoding](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
SeparableConv2 | [slim.separable_conv2d](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)
UnitNorm | [slim.unit_norm](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py)

TF-Slim也提供了repeat和stack两种元操作，允许用户重复执行相同的操作。例如，VGG网络中的以下片段：
```python
net = ...
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```

一种方法是使用循环减少代码冗余：
```python
net = ...
for i in range(3):
  net = slim.conv2d(net, 256, [3, 3], scope='conv3_%d' % (i+1))
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```
当然也可以使用TF-Slim的repeat操作
```python
net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```

注意slim.repeat不仅适用于相同参数，它还可以合理地为调用的slim.conv2d设置scopes。
更具体地说，上面的例子中的scope将被命名为'conv3/conv3_1'，'conv3/conv3_2'和'conv3/conv3_3'。

此外，TF-Slim的slim.stack操作允许调用者重复使用不同参数的相同操作来创建stack or tower of layers。slim.stack为新创建的操作也创建新的tf.variable_scope。
比如创建简单的多层感知器：
```python
# Verbose way:
x = slim.fully_connected(x, 32, scope='fc/fc_1')
x = slim.fully_connected(x, 64, scope='fc/fc_2')
x = slim.fully_connected(x, 128, scope='fc/fc_3')

# Equivalent, TF-Slim way using slim.stack:
slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')
```

更多的例子如下(Similarly, one can use stack to simplify a tower of multiple convolutions)：
```python
# Verbose way:
x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')

# Using stack:
slim.stack(x, slim.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])], scope='core')
```

##### Scopes
除了TF中作用域机制类型(name_scope, variable_scope)外，TF-Slim还添加了一个名为arg_scope的新作用域机制。
这个新的作用域允许用户指定一个或多个操作和一系列参数，这些参数将被传递给arg_scope中定义的每个操作。考虑下面的代码片段：
```python
net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
net = slim.conv2d(net, 256, [11, 11], padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')
```

由上可知三个卷积层共享了需用相同的参数。两个卷积层拥有相同padding，所有三个卷积层拥有相同weights_initializer和weight_regularizer。
以上代码很难阅读，并且包含很多重复的值，这些值应该被分解出来。一种解决方案是使用变量指定默认值：
```python
padding = 'SAME'
initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = slim.l2_regularizer(0.0005)
net = slim.conv2d(inputs, 64, [11, 11], 4,
                  padding=padding,
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv1')
net = slim.conv2d(net, 128, [11, 11],
                  padding='VALID',
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv2')
net = slim.conv2d(net, 256, [11, 11],
                  padding=padding,
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv3')
```

该解决方案确保所有三个卷积共享完全相同的参数值，但这不会完全消除代码混乱。
通过使用arg_scope，我们可以确保每个图层使用相同的值并简化代码：
```python
  with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
    net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
    net = slim.conv2d(net, 256, [11, 11], scope='conv3')
```
注意以上代码中，参数值被arg_scope设置，但是可以被局部修改。padding已被设置为SAME，第二个卷积层将其设置为VALID。
arg_scopes也可以被嵌套，在同一scope中使用多操作，例如：
```python
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
  with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
    net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = slim.conv2d(net, 256, [5, 5],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                      scope='conv2')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')
```


##### 例子：VGG16图层
结合TF-Slim Variables, Operations and scopes，可以使用很少的代码行编写一个常见的复杂网络。例如，整个VGG可以通过以下代码片段来定义：
```python
def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
  return net
```























