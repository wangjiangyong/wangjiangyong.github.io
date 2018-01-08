---
layout: post
title: "Federa系统驱动概述"
category: Linux 
tags: [Driver, Fedora]
---

#### How Hardware Drivers Work on Linux

大部分驱动是开源的或者已经整合到Linux上了。驱动程序一般是内核的一部分，即使一些图形驱动是Xorg (the graphics system)的一部分。打印机驱动是包含在CUPS(the print system). 也就是说大部分驱动是包含在内核，图形服务器和打印服务器。这些驱动一般是开源爱好者，或者企业把源码直接贡献到内核或者其他项目。 再换句话说就是很多驱动程序是开箱即用的，不需要去找厂商提供的驱动并安装，系统会自己检测硬件设备并加载合适驱动。

#### How to Install Proprietary Drivers

一些厂商提供闭源驱动，这些驱动厂商自己编写并维护。大部分厂商无法将其打包并自动使用这些驱动。 通常这些主要包括NVIDIA和AMD的专有图形驱动，它将给游戏提供更好图形性能，虽然开源驱动可以使显卡工作起来，但是很难提供同水平的3D游戏性能。 一些无线网卡驱动也是专有驱动，需要查找和手动安装驱动程序。

目前安装专有驱动是和Linux发行版有关，Ubuntu和Ubuntu衍生系统，"Additional Drivers"工具，它会根据硬件检测适合的驱动并安装它。Linux Mint的“Driver Manager”工具也有同样的功能。 Fedora是反对专有驱动（详情见链接），所以安装起来会更复杂。不同的发行版处理专有驱动是不同的方法。 链接:https://fedoraproject.org/wiki/Forbidden_items?rd=ForbiddenItems#Proprietary_Kernel_Drivers

#### How to Install Printer Drivers

打印机驱动可能是Linux最头痛的事，它很有可能工作不起，无论你做什么。比较好的方式是选购Linux支持的打印机。


#### How to Make Other Hardware Work

一般来说，需要安装的专有驱动，发行版中并不提供。比如NVIDIA和AMD提供可以使用的driver-installer packages。但是需要的是发行版将专有驱动制作好的特定包。 -- they’ll work best. 如果Linux开机后设备无法使用，或者是安装了发行版提供的专有驱动也无法使用，那么它真得没法使用了。 如果使用的是老版本的发行版，升级到更新的版本将会获得最新的硬件支持和改善。但是，如果还是没法工作，那么硬件驱动安装将没那么简单。

查找配置特定硬件在特定Linux发行版可以工作的指南，可能会有帮助。这样的指南会指导你查找厂商提供的驱动，并使用终端命令去安装它。 老的专有驱动无法保证在新的发行版系统中正常工作，最好的方案还是厂商将驱动贡献到内核中。

如果不得不查找厂商专有驱动，并过分配置（超出发行版提供提供的指南）安装使用它，很有可能它无法使发行版中最新的软件正常工作。

链接：http://www.howtogeek.com/213488/how-to-install-hardware-drivers-on-linux/

