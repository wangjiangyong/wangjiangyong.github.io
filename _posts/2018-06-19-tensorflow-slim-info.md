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

#### 训练模型
训练TF模型需要模型、loss函数、梯度计算，以及迭代地计算loss相关的模型权重梯度并相应地更新梯度的训练过程。
TF-Slim提供了常见的loss函数和运行训练、验证过程中的一系列辅助函数。

##### Losses
loss函数定义了期望优化的数量。对于分类问题，常用的就是类别间实际分布和预测概率分布之间的交叉熵。对于回归问题，通常使用预测值和真实值的误差平方和。      
一些多任务学习模型，需要同时使用多loss函数。换句话说，最小化的loss函数是多个loss函数之和。比如，不仅预测图片中场景类型，还有每个像素的相机深度。
这个模型的loss函数将会是分类loss和深度预测loss之和。      
TF-Slim提供了通过losses模块简单地定义和追踪loss函数的机制。考虑如下训练VGG网络的例子：
```python
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg

# Load the images and labels.
images, labels = ...

# Create the model.
predictions, _ = vgg.vgg_16(images)

# Define the loss functions and get the total loss.
loss = slim.losses.softmax_cross_entropy(predictions, labels)
```
以上例子中，使用了TF-Slim实现的VGG来创建模型，并加入了标准的分类损失。      
下面的例子是多任务模型的情形：
```python
# Load the images and labels.
images, scene_labels, depth_labels = ...

# Create the model.
scene_predictions, depth_predictions = CreateMultiTaskModel(images)

# Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)

# The following two lines have the same effect:
total_loss = classification_loss + sum_of_squares_loss
total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
```
以上列子中，使用了slim.losses.softmax_cross_entropy和slim.losses.sum_of_squares两种loss。并调用slim.losses.get_total_loss()来得到总的loss。
当使用TF-Slim创建loss函数，TF-Slim将loss添加到特定的TF loss函数集合，这样可以使用户既可以人工管理总loss，也可以让TF-Slim来辅助管理。        

当希望使用TF-Slim来管理定制的loss函数，loss_ops.py提供函数来添加loss到TF-Slim集合，如下例子：
```python
# Load the images and labels.
images, scene_labels, depth_labels, pose_labels = ...

# Create the model.
scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)

# Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
slim.losses.add_loss(pose_loss) # Letting TF-Slim know about the additional loss.

# The following two ways to compute the total loss are equivalent:
regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss

# (Regularization Loss is included in the total loss by default).
total_loss2 = slim.losses.get_total_loss()
```

##### Training Loop
TF-Slim在learning.py中提供了训练模型的简单强大的工具集。其中包括Train函数，它可重复地测量loss，计算梯度和保存模型到硬盘以及一些便捷操作梯度的函数。比如，一旦指定了模型
、loss函数和优化模式，就可以调用slim.learning.create_train_op和slim.learning.train来实现优化过程：
```python
g = tf.Graph()

# Create the model and specify the losses...
...

total_loss = slim.losses.get_total_loss()
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# create_train_op ensures that each time we ask for the loss, the update_ops
# are run and the gradients being computed are applied too.
train_op = slim.learning.create_train_op(total_loss, optimizer)
logdir = ... # Where checkpoints are stored.

slim.learning.train(
    train_op,
    logdir, # specifies the directory where the checkpoints and event files are stored.
    number_of_steps=1000, #  limit the number of gradient steps taken to 1000
    save_summaries_secs=300, # indicates that compute summaries every 5 minutes
    save_interval_secs=600): # indicates that save a model checkpoint every 10 minutes
```

##### Working Example: Training the VGG16 Model
```python
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg

...

train_log_dir = ...
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
  # Set up the data loading:
  images, labels = ...

  # Define the model:
  predictions = vgg.vgg_16(images, is_training=True)

  # Specify the loss function:
  slim.losses.softmax_cross_entropy(predictions, labels)

  total_loss = slim.losses.get_total_loss()
  tf.summary.scalar('losses/total_loss', total_loss)

  # Specify the optimization scheme:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

  # create_train_op that ensures that when we evaluate it to get the loss,
  # the update_ops are done and the gradient updates are computed.
  train_tensor = slim.learning.create_train_op(total_loss, optimizer)

  # Actually runs training.
  slim.learning.train(train_tensor, train_log_dir)
```

#### 微调已有模型
##### Brief Recap on Restoring Variables from a Checkpoint
模型被训练后，可以调用tf.train.Saver()恢复模型。tf.train.Saver()将从给定的检查点恢复Variables。很多情况下，tf.train.Saver()提供了恢复所有或部分变量的简单机制。
```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to restore all the variables.
restorer = tf.train.Saver()

# Add ops to restore some variables.
restorer = tf.train.Saver([v1, v2])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
```

##### Partially Restoring Models
很多情况下，需要在新数据集上微调预训练模型。这种情形下，可以使用TF-Slim的helper函数来选择需要恢复的变量。
```python
# Create some variables.
v1 = slim.variable(name="v1", ...)
v2 = slim.variable(name="nested/v2", ...)
...

# Get list of variables to restore (which contains only 'v2'). These are all
# equivalent methods:
variables_to_restore = slim.get_variables_by_name("v2")
# or
variables_to_restore = slim.get_variables_by_suffix("2")
# or
variables_to_restore = slim.get_variables(scope="nested")
# or
variables_to_restore = slim.get_variables_to_restore(include=["nested"])
# or
variables_to_restore = slim.get_variables_to_restore(exclude=["v1"])

# Create the saver which will be used to restore the variables.
restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
```


##### Restoring models with different variable names
从检查点恢复变量时，Saver在检查文件中定位变量名同时映射它们到当前计算图中的变量。      
当检查点文件中的变量名与计算图中的匹配，这种情况可以正常运行。然而，不匹配时，需要给Saver提供映射检查点文件变量名与计算图中变量关系的字典。如下例子：
```python
# Assuming than 'conv1/weights' should be restored from 'vgg16/conv1/weights'
def name_in_checkpoint(var):
  return 'vgg16/' + var.op.name

# Assuming than 'conv1/weights' and 'conv1/bias' should be restored from 'conv1/params1' and 'conv1/params2'
def name_in_checkpoint(var):
  if "weights" in var.op.name:
    return var.op.name.replace("weights", "params1")
  if "bias" in var.op.name:
    return var.op.name.replace("bias", "params2")

variables_to_restore = slim.get_model_variables()
variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}
restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
```


##### Fine-Tuning a Model on a different task
如下情形，当已经拥有了预训练的VGG16模型。模型在拥有1000类别的ImageNet数据集上训练得到。现在需要将其用在只有20类别的Pascal VOC数据上。     
为了实现以上需求，首先使用预训练模型的值初始化新模型，包括最后一层的初始化：
```python
# Load the Pascal VOC data
image, label = MyPascalVocDataLoader(...)
images, labels = tf.train.batch([image, label], batch_size=32)

# Create the model
predictions = vgg.vgg_16(images)

train_op = slim.learning.create_train_op(...)

# Specify where the Model, trained on ImageNet, was saved.
model_path = '/path/to/pre_trained_on_imagenet.checkpoint'

# Specify where the new model will live:
log_dir = '/path/to/my_pascal_model_dir/'

# Restore only the convolutional layers:
variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)

# Start training.
slim.learning.train(train_op, log_dir, init_fn=init_fn)
```

#### 评估模型
已经训练好模型（甚至模型训练时）都希望看到模型实际的效果如何。可以使一系列评估指标来给模型打分。评估代码主要执行加载数据、完成推测、对比实际结果和记录评估分数。
这些步骤可以执行一次或者周期性执行。
##### Metrics
定义指标来衡量性能，但它不是loss函数（loss函数在训练过程中直接被优化）。但是在评估模型中关注的是什么呢。比如，我们希望最小化log loss，但是关注的是使用F1分数（测试准确度），或者是Intersection Over Union score？         
TF-Slim提供了一系列使评估模型简单的指标操作。计算指标值可以分为以下三步：
+ 初始化：初始化用来计算指标的变量
+ 聚合：实现用来计算指标的操作
+ 最后：（可选）完成任何最后操作来计算指标值。比如，计算平局值、最小/最大值等等

比如，计算mean_absolute_error，两个变量count和total初始化为零。在聚合过程中，得到预测值和标签值，用来计算差值的绝对值之和，并赋值给total。每次得到不同count值。最后，使用total除以count来得到平均值。          
如下示例代码展示申明指标的API：（指标是在测试集上进行评估）
```python
images, labels = LoadTestData(...)
predictions = MyModel(images)

mae_value_op, mae_update_op = slim.metrics.streaming_mean_absolute_error(predictions, labels)
mre_value_op, mre_update_op = slim.metrics.streaming_mean_relative_error(predictions, labels)
pl_value_op, pl_update_op = slim.metrics.percentage_less(mean_relative_errors, 0.3)
```
以上示例中，创建指标后返回两个值：value_op 和 update_op。value_op是idempotent?操作，返回当前指标值。update_op操作返回聚合步骤所提到的指标值。            
追踪每步的value_op 和 update_op较为繁杂，为了解决此问题，TF-Slim提供了两个便捷的函数：
```python
# Aggregates the value and update ops in two lists:
value_ops, update_ops = slim.metrics.aggregate_metrics(
    slim.metrics.streaming_mean_absolute_error(predictions, labels),
    slim.metrics.streaming_mean_squared_error(predictions, labels))

# Aggregates the value and update ops in two dictionaries:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
    "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
})
```

##### Working example: Tracking Multiple Metrics
```python
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg


# Load the data
images, labels = load_data(...)

# Define the network
predictions = vgg.vgg_16(images)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
    "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
})

# Evaluate the model using 1000 batches of data:
num_batches = 1000

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  for batch_id in range(num_batches):
    sess.run(names_to_updates.values())

  metric_values = sess.run(names_to_values.values())
  for metric, value in zip(names_to_values.keys(), metric_values):
    print('Metric %s has value: %f' % (metric, value))
```

##### Evaluation Loop
TF-Slim提供了评估模块（evaluation.py），包括了使用metric_ops.py模块中的指标来编写评估脚本的helper函数。这些包括了周期运行评估、评估批数据指标、打印和概述指标结果。      
示例如下：
```python
import tensorflow as tf

slim = tf.contrib.slim

# Load the data
images, labels = load_data(...)

# Define the network
predictions = MyModel(images)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    'accuracy': slim.metrics.accuracy(predictions, labels),
    'precision': slim.metrics.precision(predictions, labels),
    'recall': slim.metrics.recall(mean_relative_errors, 0.3),
})

# Create the summary ops such that they also print out to std output:
summary_ops = []
for metric_name, metric_value in names_to_values.iteritems():
  op = tf.summary.scalar(metric_name, metric_value)
  op = tf.Print(op, [metric_value], metric_name)
  summary_ops.append(op)

num_examples = 10000
batch_size = 32
num_batches = math.ceil(num_examples / float(batch_size))

# Setup the global step.
slim.get_or_create_global_step()

output_dir = ... # Where the summaries are stored.
eval_interval_secs = ... # How often to run the evaluation.
slim.evaluation.evaluation_loop(
    'local',
    checkpoint_dir,
    log_dir,
    num_evals=num_batches,
    eval_op=names_to_updates.values(),
    summary_op=tf.summary.merge(summary_ops),
    eval_interval_secs=eval_interval_secs)
```


原文：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/README.md




















