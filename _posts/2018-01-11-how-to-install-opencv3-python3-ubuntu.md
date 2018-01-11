---
layout: post
title: "在Ubuntu16.04安装Python3.x版本OpenCV3.x组件"
category: Ubuntu
tags: [Ubuntu, Python, OpenCV]
---

#### 介绍

在Linux上安装基于Python3的OpenCV没有那么便捷，本文记录了安装的步以供参考。使用的系统是64位的Ubuntu16.04，和Python3.5（适用于Ubuntu12.04++和Python3.x）。

#### 安装过程

```bash
# Update the repository before installing the necessary packages
sudo apt-get update

sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python3.5-dev

# If you using Python 3.2  or below, you can skip to the next step
# As the pycofig.h header file is not in the typical place, we would have to copy the file to the expected location.
# Run the following code below:
python3.5-config --includes

# The output would look something similar below:
# -I/usr/include/python3.5m -I/usr/include/x86_64-linux-gnu/python3.5m
# 如果输出前后部分一致，可以忽略下面的说明。
# The first part of the output is the expected location & the second part shows the current location of the config file. 
# To solve this problem, we’ll copy the file from the current location to the expected location.
# sudo cp /usr/include/x86_64-linux-gnu/python3.5m/pyconfig.h /usr/include/python3.5m/

# Downloading the OpenCV Source Code
git clone https://github.com/opencv/opencv.git
cd opencv
# 切换到需要的版本分支
git checkout 3.2.0

mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

# make -j $(nproc --all)
# $(nproc --all)输出所有可用处理器数目，如果是2就是如下命令：
make -j2

sudo make install

```

#### 验证安装

```bash
python3
>>> import cv2
>>> cv2.__version__
'3.2.0'

# 直接安装可能需要的包
sudo apt-get install python3-numpy
sudo apt-get install python3-matplotlib
```

参考：
+ http://cyaninfinite.com/tutorials/installing-opencv-in-ubuntu-for-python-3/
+ https://docs.opencv.org/trunk/d2/de6/tutorial_py_setup_in_ubuntu.html
+ https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/




