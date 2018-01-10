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

```Python
import subprocess as sp
import numpy as np

FFMPEG_BIN = "ffmpeg"

command2 = [ FFMPEG_BIN,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',     #-f fmt 强迫采用格式fmt
        '-vcodec','rawvideo',
        '-s', '960x540', # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '24', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        #'-vcodec', 'libx264',
        'output_video.flv' ]

proc2stream = sp.Popen( command2, stdin=sp.PIPE,bufsize=10**8)

command1 = [FFMPEG_BIN,
           '-i', 'rtmp://192.168.xxx.xxx:1935/live/livestream1' , 
           '-f', 'image2pipe', 
           '-pix_fmt', 'rgb24',
           '-vcodec', 'rawvideo', '-']

proc2pics = sp.Popen(command1,stdout=sp.PIPE,bufsize=10**8)

while True:
    raw_image = proc2pics.stdout.read(960*540*3)
    image = np.fromstring(raw_image,dtype='uint8')
    #print(type(image))
    #image = image.reshape((540,960,3))
    #print(image.shape)
    proc2stream.stdin.write(image.tostring())
```
以上代码可实现从数据流提取数据帧，并将提取的帧写入output_video.flv文件。使用一下命令将文件进行推流：

ffmpeg -re -i output_video.flv -f flv rtmp://192.168.xxx.xxx/live/livestream2 


参考：
1. http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/     
2. https://ffmpeg.org/ffmpeg-protocols.html#pipe （管道）
3. https://github.com/Zulko/moviepy/

#### ffmpeg常用参数

ffmpeg [global_options] **{[input_file_options] -i input_url} ... {[output_file_options] output_url}** ...

**-f fmt (input/output)**

Force input or output file format. The format is normally auto detected for input files and guessed from the file extension for output files, so this option is not needed in most cases.

**-vcodec codec (output)**

Set the video codec. This is an alias for "-codec:v".

**-s[:stream_specifier] size (input/output,per-stream)**

Set frame size.

**-pix_fmts**

Show available pixel formats.


**-pix_fmt[:stream_specifier] format (input/output,per-stream)**

Set pixel format. Use "-pix_fmts" to show all the supported pixel formats.  If the selected pixel format can not be selected, 
ffmpeg will print a warning and select the best pixel format supported by the encoder.  If pix_fmt is prefixed by a "+", 
ffmpeg will exit with an error if the requested pixel format can not be selected, and automatic conversions inside filtergraphs are disabled.If pix_fmt is a single "+", ffmpeg selects the same pixel format as the input (or graph output) and automatic conversions are disabled.

**-r[:stream_specifier] fps (input/output,per-stream)**

Set frame rate (Hz value, fraction or abbreviation).

**-i url (input)**

input file url


**-an (output)**

Disable audio recording.








[1]:https://github.com/ossrs/srs
