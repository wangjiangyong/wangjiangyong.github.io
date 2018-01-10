---
layout: post
title: "Fedora大战Ubuntu"
category: Linux
tags: [Linux, Fedora, Ubuntu]
---
<img src="https://raw.githubusercontent.com/wangjiangyong/wangjiangyong.github.io/master/assets/images/fudiff.jpg" width="250" height="250" />

### 介绍

新的Linux发行版一直在出现，对于一些用户来说，这些发版本都尝试下可能会变得很乏味。 你可能听到有人问，“这些发行版的关键是什么？”。也许你也被问到某两个**Linux发行版之间的区别**这样的问题。 这些问题可能起初看起很奇怪，但它们是有意义的，特别是一个在坚持学习Linux的爱好者。Fedora和Ubuntu都不是新的发行版，但是最近都有新版本。

#### 历史&研发

Ubuntu第一个版本是基于Debian的不稳定分支，并于2004年10月发行。Fedora老些，第一个版本发布于2003年11月，这后面的故事有些复杂。

Fedora第一个版本称为Fedora Core 1，它基于Red Hat Linux 9。Fedora被设想为面向社区以替代Red Hat，它拥有两个主要的仓库：1) Core，由Red Hat开发者维护； 2) Extras，由社区维护。 然而在2003年底，Red Hat Linux与Fedora合并成为一个社区发行版，而Red Hat Enterprise Linux被创建为支持其商业版。 直到2007年Fedora保留“Core”作为其名称的一部分，但随着Fedora 7的发布，Core和Extra仓库被合入。由此发行版被称为Fedora。

> 说到这里，它们二者最大的区别是：原来的Red Hat Linux本质上分为Fedora和Red Hat Enterprise Linux，而Debian仍然是一个完整的，独立于Ubuntu的实体，Ubuntu是从Debian的一个分支导入软件包。

许多人认为Fedora是直接基于Red Hat Enterprise Linux（RHEL），这样理解并不完全正确。恰恰相反，RHEL新版本是Fedora的分支，它已经过质量和稳定性测试。 例如，RHEL 7是基于Fedora 19和20的仓库。Fedora社区也为RHEL提供存储在Extra Packages for Enterprise Linux（EPEL）仓库的软件包。

这两个发行版背后的研发组织结构类似。Fedora项目（成立于2003年）负责协调Fedora的开发，它由Red Hat赞助。 Fedora委员会管理这个倡议，并且主席（Fedora项目负责人）被红帽选择和雇用。这还有其他管理团体，如Fedora工程指导委员会和Fedora大使指导委员会，它们的成员由社群选出。

另一方面，Ubuntu由Canonical直接资助和管理。 Ubuntu社区由几个较大的团体构成，主要的是社区理事会和技术委员会，它们的成员由Canonical创始人Mark Shuttleworth提名。 其他团体包括论坛理事会，IRC理事会和开发者会员理事会。用户可以申请Ubuntu成员资格，并在各种社区组织的团队中成为贡献者。


#### 发布周期&支持

Ubuntu每六个月发布一个新版本，一般在四月和十月发布。 每四个版本被认为是长期支持（LTS）版本，这意味着LTS版本每两年出版一次。自2012年以来，每个LTS版本都接受未来五年的官方支持和更新。其他，常规版本过去是支持18个月，但是在2013年将其缩短为9个月。

Fedora没有严格的计划，但是新版本通常每六个月发布一次。但是，它们支持13个月，这比Ubuntu的常规版本的支持期更长。**Fedora的没有LTS版本**。

#### 名字里包含什么

两个发行版都在其名称中都包含了版本号。Ubuntu发行版名字中，**第一个数字表示版本发布的年份和第二个是月份**。这实际上是有帮助的，可以一眼就知道发布版的年纪。例如，Ubuntu 13.04于2013年4月发布。Fedora比较简单，使用整数。从第一个版本开始为1，现在是25。

对于Ubuntu，codename包含两个以相同字母开头的单词。第一个是形容词，第二是动物，通常不寻常或罕见。任何人都可以为即将到来的Ubuntu版本建议名称，但最终决定由Mark Shuttleworth发布，以及关于名称的解释或轶事。

2013年发布的Fedora 20 Heisenbug 是最后一个版本的codename，所有后续的版本只被称为“Fedora X”，其中X代表前一版本之后的任何一个数字。在此之前，社群中的任何人都可以建议一个名字，但它必须遵循一系列规则，才有资格获得理事会成员的批准。发布名称应该共享有联系的事物，最好是一个不寻常或新颖的联系，不应该是活人的名字或商标的术语。 Fedora X和Fedora X+1的名称之间的关系应该匹配“is-a”公式，因此下面是真实的：X是Y，X是1。 为了说明，Fedora 14被称为Laughlin和Fedora 15 Lovelock。 Lovelock和Laughlin都是内华达州的城市。 但是，Fedora X和Fedora X+2的关系应该不一样！

这名字真蛋疼，好吧，这也许是开发人员决定放弃它的原因之一。

#### 版本 & 桌面环境

Fedora有三个主要版本：云，服务器和工作站。前两个是很直观，工作站实际上是大多数人使用的版本，针对台式计算机和笔记本电脑（32位或64位）。Fedora社区也为基于ARM的设备提供三个版本的单独映像。 还有Fedora Rawhide，它是Fedora不断更新的开发版本，包含所有Fedora软件包的最新版本。Rawhide是新软件包的测试基地，所以它不是100％稳定，但你仍然可以使用它作为滚动发布版本。

**Ubuntu在版本方面比Fedora强，至少在数量上**。 除了标准的桌面版本，Ubuntu提供了单独的产品，比如云，服务器，核心（物联网设备）和针对移动设备的Ubuntu Touch。 桌面版本支持32位和64位系统，并且服务器映像可用于不同的架构（ARM，LinuxONE，POWER8）。还有Ubuntu Kylin，一个为中国用户提供的Ubuntu特殊版本，它于2010年首次发布为“Ubuntu中文版”，并在2013年被重新命名为官方子项目。

至于桌面环境，主要的Fedora版本使用Gnome 3和Gnome Shell。Ubuntu的默认DE是Unity，其他选项通过“Ubuntu flavor”的方式提供，它是Ubuntu不同的桌面环境。有Kubuntu（KDE），Ubuntu GNOME，Ubuntu MATE，Xubuntu（Xfce），Lubuntu（LXDE）和一个新的变种名为Budgie Remix，它希望成为官方Ubuntu flavor。

Fedora和Ubuntu flavor类似的是Spins，或“替代桌面”。有KDE，Xfce，LXDE，MATE和Cinnamon桌面环境，和一个特殊地称为Sugar，它是简单的学习环境。 这个项目专为儿童和学校，特别是在发展中国家。

Fedora还有Labs或“功能软件包”。它们是可以安装在现有Fedora系统上或作为独立Linux发行版的专用软件集合。 可用实验室包括设计套件，游戏，机器人套件，安全实验室和科学。 Ubuntu提供类似于Edubuntu，Mythbuntu和Ubuntu Studio - 子项目的形式，分别具有用于教育，家庭娱乐系统和多媒体制作的专用应用程序。


#### 包 & 仓库

> 这个方面是Ubuntu和Fedora之间最明显的区别。二者都使用包管理系统，Fedora使用RPM和包的后缀是.rpm，而Ubuntu使用DPKG和包的后缀是.deb。这意味着默认情况下Ubuntu的软件包与Fedora不兼容，除非使用类似Alien工具的转换来安装它。Ubuntu还提供了Snappy包，它应该比.deb包更安全和更容易维护，但是它们尚未在开发人员中广泛使用。

**除了一些二进制固件，Fedora不在其官方仓库中包含任何专有软件**。这包括**图形驱动程序，编解码器以及受专利和法律问题限制的任何其他软件**。这样的话，Ubuntu仓库中比Fedora有更多的软件包。

Fedora面向的主要目标之一是只提供免费和开源软件，社区鼓励用户为其非免费应用程序找到替代方案。如果你想听MP3音乐或在Fedora上播放DVD，你将不会在官方仓库中找到相应的支持。但是，**有第三方仓库，如RPMFusion**。它包含大量的免费和非免费的软件，可以安装在Fedora上直接安装。

Ubuntu旨在遵守Debian的自由软件准则，但它仍然做出了很多让步。与Fedora不同，**Ubuntu在其官方仓库的受限分支中包含了专有驱动程序**。**还有Partner合作伙伴仓库，其中包含Canonical合作供应商的专有软件** - 例如Skype和Adobe Flash Player。可以从Ubuntu Software Center购买商业应用程序，并从仓库中安装软件包（ubuntu-restricted-extras），即可得到DVD，MP3和其他流行的编解码器的支持。

Fedora's Copr是一个类似于Ubuntu's Personal Packages（PPA）的平台 - 它允许任何人上传包和创建自己的仓库。 这里的区别与软件许可的一般方法相同 -- 不应该上传包含非自由组件的软件包，或者任何其他明显被Fedora Project Board禁止的软件包。

#### 目标群体 & 愿景

从开始，Fedora一直专注于三件事：创新，社区和自由。它提供和促进免费和开源软件，并强调每个社区成员的重要性。它是由社区开发的，积极鼓励用户参与项目，不仅作为开发人员，而且作为作家，翻译，设计师和公众演讲者（Fedora大使）。有一个特别的项目，帮助想要贡献社区的女性，目标是在科技和FOSS圈子打击基于性别的偏见和隔离。

此外，Fedora通常是第一个或首批发布版本中，使用和展示新技术和应用程序的发行商。它是SELinux首批发行版本其中之一，也包括Gnome 3桌面，使用Plymouth作为bootsplash应用程序，采用systemd作为默认初始化系统，并使用Wayland而不是Xorg作为默认显示服务器。

Fedora的开发人员通常与其他发行商和上游项目合作，并与其他Linux生态系统共享他们的升级和贡献。由于这种不断的实验和创新，Fedora经常被贴标签为bleeding-edge，不稳定的版本，不适合初学者和日常使用。这是Fedora最广泛的神话之一，Fedora社区正在努力改变这种看法。**虽然热衷于最新功能的开发人员和高级用户是Fedora主要的目标受众用户**，但是**Fedora依然可以被任何人使用，就像Ubuntu一样**。

对于Ubuntu，它的目标与Fedora有所重叠。Ubuntu也努力创新，但他们选择了一种更加面向消费者的方法。通过为移动设备提供操作系统，Ubuntu正在试图在市场上创建自己的立足之地，同时推动其主要项目-融合。

社区似乎不太参与关键问题的决策，这方面主要反映在用户反对过去的Ubuntu版本的变化。Ubuntu卷入一些争议，最突出的是Ubuntu 12.10中Unity购物镜头的隐私问题。 尽管如此，**Ubuntu通常被宣布为最受欢迎的Linux发行版，这归功于它的用户友好，面向初学者和Windows用户这样的战略**。

不过，**Fedora有杰出的领袖 -- Linus Torvalds**，Linux的创造者，他的电脑上使用的是Fedora。


链接：http://beebom.com/ubuntu-vs-fedora/
