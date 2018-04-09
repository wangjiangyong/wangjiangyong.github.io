---
layout: post
title: "在Ubuntu 16.04安装Caffe及Linux库设置"
category: Ubuntu
tags: [DeepLearing,Linux]
---

#### 介绍

```shell
ldd xxxx.so 查看相关的库的链接情况  
locate xxx.so 查找系统中的相关库目录  
ls -l xxx.so 查看库的链接  
sudo ln -s 建立软连接  
sudo ldconfig 使链接生效  
```

详情：
https://stackoverflow.com/questions/4754633/linux-program-cant-find-shared-library-at-run-time
https://www.cyberciti.biz/tips/linux-shared-library-management.html
