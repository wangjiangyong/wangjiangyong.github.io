---
layout: post
title: "C Program GObject简介 "
category: Program
tags: [C,OO]
---

C Programming/GObject
---

因为C语言并不是为面向对象程序设计而生，它没有明确的支持类，继承，多态和其他面向对象概念。它同样没有自己的虚表，虚表多见于C++/Java/C#面向对象语言。因此使用C语言特性和标准库来实现面向对象的编程范例可能不那么容易。然而，可以使用包含函数指针和数据的结构体，或者通过使用第三方库来完成。有许多第三方库被设计来支持C语言实现面向对象程序。其中最通用和广泛使用的是GObject系统，它是Glib的一部分。GObject系统有自己的虚表。使用GObject系统来创建C语言实现的对象，它必须从GObject结构体进行子类化。

##### 声明类
（两个结构体需要被声明，*instance*对象成员 和 *class*类成员）

```C
/*in myobject.h*/
typedef struct _MyObject MyObject;
typedef struct _MyObjectClass MyObjectClass;

struct _MyObject
{
  GObject parent_instance;

  /* instance members */
};

struct _MyObjectClass
{
  GObjectClass parent_class;

  /* class members */
};
```
##### 样板代码
因为GObject系统仅仅是第三方库，因此不能对C语言本身做什么修改，所以创建新的对象（类）需要大量的模版代码。

样板代码如下：
```C
/* in myobject.h */
#define MY_TYPE_OBJECT                   (my_object_get_type ())
#define MY_OBJECT(obj)                   (G_TYPE_CHECK_INSTANCE_CAST ((obj), MY_TYPE_OBJECT, MyObject))
#define MY_IS_OBJECT(obj)                (G_TYPE_CHECK_INSTANCE_TYPE ((obj), MY_TYPE_OBJECT))
#define MY_OBJECT_CLASS(_class)          (G_TYPE_CHECK_CLASS_CAST ((_class), MY_TYPE_OBJECT, MyObjectClass))
#define MY_IS_OBJECT_CLASS(_class)       (G_TYPE_CHECK_CLASS_TYPE ((_class), MY_TYPE_OBJECT))
#define MY_OBJECT_GET_CLASS(obj)         (G_TYPE_INSTANCE_GET_CLASS ((obj), MY_TYPE_OBJECT, MyObjectClass))
```

##### 定义类
使用之前，新创建的类需要被定义。
```C
/* in myobject.c */
G_DEFINE_TYPE(MyObject, my_object, G_TYPE_OBJECT);
```



##### 静态方法
有一些静态方法可定义，可不定义。主要依据于设计的类，对于最小的类以下的函数是必须的：
```C
/* in myobject.c */
static void my_object_class_init(MyObjectClass *_class)
{
     /* code */
}

static void my_object_init(MyObject* obj)
{
     /* code */
}
```

##### 构造函数
对于C语言实现的对象没有内部的方法分配内存，因此对于新类需要明确的构造函数。
```C
/* in myobject.c */

GObject* my_object_new(void)
{
     return g_object_new(MY_TYPE_OBJECT , 0);
}
```

##### 类的使用
虽然在创建对象时，使用自己指针类型是很对的，但是推荐使用继承关系顶层的对象的指针类型，即最远的基类。新创建的类可以被如下使用：
```C
/* in main.c */

/* Note: GObject is at the top of the hierarchy. */

/* declaration and construction */
GObject* myobj = my_object_new();

/* destruction */
g_object_unref(myobj);   //析构函数？
```

##### 继承实现
在GObject系统，继承可以通过子类化GObject实现。因为C语言不提供与继承相关的关键字和操作符，通过分别在继承instance和继承class结构体中声明基instance和基class为成员，来实现继承。C代码如下：
```C
/* derived object instance */
struct DerivedObject
{
     /* the base instance is a member of the derived instance */
     BaseObject parent_instance;
};

/* derived object class */
struct DerivedObjectClass
{
    /* the base class is a member of the derived class */
    BaseObjectClass parent_class;
};
```

###### 参考：
https://en.wikibooks.org/wiki/C_Programming/GObject
https://www.cs.rit.edu/~ats/books/ooc.pdf
