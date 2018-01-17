---
layout: post
title: "cs231n-lecture11图像检测和分割课程笔记"
category: 深度学习
tags: [Image, DeepLearning]
---

#### 语义分割 
**语义分割（semantic segmentation）**

输入：图像

处理：需对图片的每个像素进行分类。不区别同类目标，关注像素点。这也是它的不足。

输出：
<img src="https://raw.githubusercontent.com/wangjiangyong/wangjiangyong.github.io/master/assets/images/semanticSegmentation.png
" width="550" height="256" />

可行方法：
1. 滑动窗口Sliding Window，将原始图片分成局部小块，进行图像分类，并将小图片的中心像素进行类别标记。由此循环，计算复杂度极高。因为卷积过程中有很多重复计算，两块小图片很可能一半是重叠的。
2. 全连接卷积网络。
<img src="https://raw.githubusercontent.com/wangjiangyong/wangjiangyong.github.io/master/assets/images/fullconn.png"          width="550" height="256" />


#### 分类定位 
**分类和定位（Classification + Localization）**

与目标检测不同，**定位前提是知道某物体是要找的**，或者不止一个。先对图片进行物体分类，再画上边界（bounding box）告知物体位置。

<img src="https://raw.githubusercontent.com/wangjiangyong/wangjiangyong.github.io/master/assets/images/localization.jpg" width="550" height="256" />

多任务loss（multi-task loss）一般处理，参数加权求和。

**人体姿势估计（Human Pose Estimation）**

图像中固定几个点的思路：除了分类和定位，还可以应用到人体姿势估计（Human Pose Estimation）上
输入：人像图片
输出：人体关节的点位（假定14个关节点位），网络预测人体姿势

<img src="https://raw.githubusercontent.com/wangjiangyong/wangjiangyong.github.io/master/assets/images/humanposeestimation.jpg" width="550" height="256" />

回归损失：L2欧几里得损失，（平滑）L1损失
分类问题，考虑交叉熵损失，softmax损失或SVM边界类型损失。
