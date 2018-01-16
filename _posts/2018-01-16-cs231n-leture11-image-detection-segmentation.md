---
layout: post
title: "cs231n-lecture11图像检测和分割课程笔记"
category: 深度学习
tags: [Image, DeepLearning]
---

#### 语义分割 
语义分割（semantic segmentation）

输入：图像

处理：需对图片的每个像素进行分类。不区别同类目标，关注像素点。这也是它的不足。

输出：


可行方法：
1. 滑动窗口Sliding Window，将原始图片分成局部小块，进行图像分类，并将小图片的中心像素进行类别标记。由此循环，计算复杂度极高。因为卷积过程中有很多重复计算，两块小图片很可能一半是重叠的。
2. 全连接卷积网络。
