---
layout: post
title: "数据处理 brief"
category: 数据处理
tags: [Data, Python, NumPy]
---

数据处理过程
-----------------
![Alt text](https://raw.githubusercontent.com/wangjiangyong/wangjiangyong.github.io/master/images/dp1.png)

+ Question Phase 提出想要解决和回答的问题
+ Wrangle Phase 数据加工，包括数据采集和清洗
+ Explore Phase 数据探索，熟悉数据并找出模式
+ Draw Conclusions Phase (or make predictions) 总结或者预测数据，需要统计或者机器学习等
+ Communicate Phase 交流研究成果，博客，论文邮件... 数据可视化是比较有效的方法


一维数据np.array
-------------
+ NumPy Index Arrays

```Python
a=np.array([1,2,3,4,5])
b=np.array([False,False,True,True,True])  #index array
a[b]
a[a>2]
```
+ In-Place VS Not In-Place

```python
import numpy as np

a = np.array([1,2,3,4])
b=a
a += np.array([1,1,1,1])  #operate in-place  也就是将新值存在原来位置，大多不是operate in-place
print b     #array([2,3,4,5])

a = np.array([1,2,3,4])
b=a
a = a + np.array([1,1,1,1]) #创建新的数组保存数值
print b     #array([1,2,3,4])

b = [1,2,3,4,5]
slice1 = b[:3]
slice1[0] = 100
print b  #[1,2,3,4,5]  由于slice1是新开的数组，所以b保持原样

a = np.array([1,2,3,4,5])
slice = a[:3]
slice[0] = 100
print a #array([100,2,3,4,5])  np.array 不同于python list 它是直接在原数组操作，提高效率。切片时候多留意

```

一维数据pd.series
-----------
+ Pandas Series它与NumPy数组类似，功能更强大，它基于np.array。
+ Pandas数组可以设置index。NumPy array是增强型的Python List。Pandas Series是list和字典的杂交。
+ Pandas Series向量化运算需要index一直才行。如果没有内置函数处理series，可使用apply函数简化数据处理。

```Python
import pandas as pd
life_expectancy = pd.Series([66.0, 40.4, 45., 100.2],
                            index=['Albania',
                                   'Algeria',
                                   'Andorra',
                                   'Angola']

)

life_expectancy.loc['Angola']
life_expectancy.iloc[0]  # 注意位置和索引的区别。

```



二维数据
--------------
###### Python: List of lists
###### NumPy: 2D array
+ 数组每个元素都是相同的数据类型，在处理CSV文件时候不方便。
+ axis=0时候，按照列来计算; axis=1时候，按照行来计算值.

###### Pandas: DataFrame
+ 优于NumPy，每列可以保存不同数据类型。可以拥有列名。即可以使用位置定位元素，也可以使用index索引和column列名来定位元素

```python
import pandas as pd

# Subway ridership for 5 stations on 10 different days
ridership_df = pd.DataFrame(
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)

```

+ Pandas Axis Names:
Instead of axis=0 or axis=1, use axis='index' or axis='columns'

+ Non-Built-In Functions for DataFrame: 针对于f:elemet -> element，使用applymap函数。针对于f:列->列，使用apply函数。类似于series使用apply处理元素。使用apply函数，也可以f:列-> 单值。

```python
def clean_state(s):
     ...

df.applymap(clean_state)

```

+ DataFrame + Series

```python
import pandas as pd

s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    0: [10, 20, 30, 40],
    1: [50, 60, 70, 80],
    2: [90, 100, 110, 120],
    3: [130, 140, 150, 160]
})

print df + s    #s与df每列相加，即1与第一列每个元素相加，2与第二列每个元素相加，以此类推。
print df.add(s, axis="columns")  #df+s 可指定轴参数

s1 = pd.Series([1, 2, 3, 4], index=('b','c','d','e'))
df1 = pd.DataFrame({
    a: [10, 20, 30, 40],
    b: [50, 60, 70, 80],
    c: [90, 100, 110, 120],
    d: [130, 140, 150, 160]
})

print df1 + s1     #s1与df1每列相加，series索引值需与dataFrame列名匹配。

```

+ Working with Multiple DataFrames：使用merge函数，类似于SQL的连接
