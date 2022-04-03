# Pandas学习笔记
## 1. 数据结构
---
Pands的主要数据结构是 Series （一维数）与 DataFrame （二维数据）。

### 1.1 Series
`Series`是带标签的一维数组，可存储整数、浮点数、字符串、Python 对象等类型的数据。轴标签统称为索引。调用 `pd.Series` 函数即可创建`Series`，`Series`由索引（`index`）和列组成，函数如下：
```python
pd.Series(data, index, dtype, name, copy)
```
参数说明：
>`data`：一组数据(`ndarray`类型)
>`index`：数据索引标签，如果不指定，默认从 0 开始
>`dtype`：数据类型，默认会自己判断
>`name`：设置名称
>`copy`：拷贝数据，默认为`False`

一个简单的实例：
```python
import pandas as pd

a = [1, 2, 3]
myvar = pd.Series(a)
print(myvar)
```
得到输出结果
```
0    1
1    2
2    3
dtype: int64
```
从上图可知，如果没有指定索引，索引值就从 0 开始，我们可以根据索引值读取数据：
```python
import pandas as pd

a = [1, 2, 3]
myvar = pd.Series(a)
print(myvar[1])
```
输出结果如下：
```
2
```
__可以指定索引值并根据索引读取数据__：
```python
import pandas as pd

a = ["Google", "Runoob", "Wiki"]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar["y"])
```
得到输出结果：
```
Runoob
```
__我们也可以使用 key/value 对象，类似字典来创建 Series__：
```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}

myvar = pd.Series(sites)

print(myvar)
```
输出结果：
```
1    Google
2    Runoob
3      Wiki
dtype: object
```
或者使用`name`给`Series`设置名称，或者使用`rename`更改名称

### 1.2 DataFrame
可以看作二维数组，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。  
DataFrame 构造方法如下：

```python
pandas.DataFrame( data, index, columns, dtype, copy)
```

参数说明：
>`data`：一组数据(`ndarray`、`series`, `map`, `lists`, `dict` 等类型)
>`index`：索引值，或者可以称为行标签
>`columns`：列标签，默认为 `RangeIndex (0, 1, 2, …, n)`
>`dtype`：数据类型
>`copy`：拷贝数据，默认为`False`  

使用列表创建,实例：
```python
import pandas as pd

data = [['Google',10],['Runoob',12],['Wiki',13]]

df = pd.DataFrame(data,columns=['Site','Age'],dtype=float)

print(df)
```

输出结果

```
     Site   Age
0  Google  10.0
1  Runoob  12.0
2    Wiki  13.0
```

使用字典（key/value）创建，其中字典的key为列名：
```python
import pandas as pd

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

df = pd.DataFrame(data)

print (df)
```
输出结果为：
```
   a   b     c
0  1   2   NaN
1  5  10  20.0
```
_使用`ndarray`时创建时，`ndarray`的长度必须相同__

__Pandas 可以使用 loc 属性返回指定行的数据，如果没有设置索引，第一行索引为 0，第二行索引为 1，以此类推：__
```python
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)

# 返回第一行
print(df.loc[0])
# 返回第二行
print(df.loc[1])
```
输出结果如下：
```
calories    420
duration     50
Name: 0, dtype: int64
calories    380
duration     40
Name: 1, dtype: int64
```

### 1.3 CSV文件
CSV（Comma-Separated Values，逗号分隔值，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本）。

Pandas中相关的函数如下：
>`read_csv`: 读取逗号分割的cvs文件为DataFrame格式

`to_string()` 用于返回 DataFrame 类型的数据，如果不使用该函数，则输出结果为数据的前面 5 行和末尾 5 行，中间部分以 `...` 代替。

我们也可以使用`to_csv()`方法将`DataFrame`存储为`csv`文件：