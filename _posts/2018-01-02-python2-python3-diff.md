---
layout: post
title: "python2与python3区别汇总"
category: Program
tags: [Python]
---
Python 的设计哲学 Explicit is better than implicit. 明着优于暗着。

Python为了确保你能顺利过渡到新版本，特别提供了__future__模块，让你在旧的版本中试验新版本的一些特性。

Python 2.x与3.x的差别
--------------------
+ print()。
+ 除法 / 这种除法叫“地板除”， 2.7结果为整数，3.x为浮点数。Python 3.x中，所有的除法都是精确除法，地板除用//表示。
+ 2.x里的字符串用'xxx'表示str，Unicode字符串用u'xxx'表示unicode。而在3.x中，所有字符串都被视为unicode，因此，写u'xxx'和'xxx'是完全一致的。
+ Python2.x默认编码是ascii，3.x默认是UTF-8，不需要继续在开头写coding:utf-8
+ 2.x中，字符串有两个类型，unicode和str，前者是文本字符串，后者字节序列。但是二者没有明显界限。
  3.x中严格区分，str表示字符串，byte表示字节序列。任何文件读写，网络传输走字节序列。

+ True 和 False 在 Python2 中是两个全局变量，可以被修改。而在 Python3 中则是关键字，不能被修改。


持续更新...
