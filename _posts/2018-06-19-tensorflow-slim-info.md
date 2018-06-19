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


