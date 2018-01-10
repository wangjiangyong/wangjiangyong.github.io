---
layout: post
title: "使用Python读写rtmp视频流处理方案"
category: 视频
tags: [Python, FFmpeg, Video]
---

#### 介绍

使用[SRS][1]搭建了流服务器（Ubuntu系统），支持rtmp协议的推流和拉流。使用OBS工具将笔记本摄像头的数据向服务器进行推流。

根据原型系统的需求，需要使用Python3从服务器读取rtmp视频流，并进行拆分图片帧











[1]:https://github.com/ossrs/srs
