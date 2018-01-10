---
layout: post
title: "使用Python读写rtmp视频流处理方案"
category: 视频
tags: [Python, FFmpeg, Video]
---

#### 介绍

使用[SRS][1]搭建了流服务器（Ubuntu系统），支持rtmp协议的推流和拉流。使用OBS工具将笔记本摄像头的数据向服务器进行推流。

根据原型系统的需求，
+ 需要使用Python3从服务器读取rtmp视频流；
+ 将拆分的图片帧转换为Numpy.array提供给模型处理；
+ 将处理后的图片帧序列推流到服务器。

#### 相关第三方库

#### 可行方案

关键点是在Python3代码中，通过**Pipe(管道)来使外部ffmpeg程序读写视频帧**。


参考：http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/









[1]:https://github.com/ossrs/srs
