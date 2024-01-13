# Numpy

入门教程：[https://www.numpy.org.cn](https://www.numpy.org.cn) 

详细应用和底层实现： [https://www.labri.fr/perso/nrougier/from-python-to-numpy/](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)

CuPy - Numpy计算加速包：[https://cupy.dev/](https://cupy.dev/)

NumCpp - 基于Numpy的C++ 库： [https://dpilger26.github.io/NumCpp/doxygen/html/index.html](https://dpilger26.github.io/NumCpp/doxygen/html/index.html)

官方文档：[https://numpy.org/](https://numpy.org/)

## 初始化数组

创建一维数组：

```python
import numpy as np 

# ---- 1D Array
# 为了创建一个2D（二维）数组，我们传递一个列表给array()函数。
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))

# 或者使用默认函数arange, linspace
c = np.arange(5) # 输出自然数组成的数量为定义指的数组
d = np.arange(0, 1, 0.2) # 等间隔 （创建一个从0开始间隔为0.2的数组，直到不超过1）
e = np.linspace(0, 2*np.pi, 5)# 单位间隔 （创建一个从0到1等间隔的5个数的数组）
```

创建二维数组：

```python
# ---- 2D Array
# 为了创建一个2D（二维）数组，我们传递一个列表的列表（或者是一个序列的序列）给array()函数。
a_2d = np.array([[11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25],
                 [26, 27, 28 ,29, 30],
                 [31, 32, 33, 34, 35]])

# ---- MD Array
# 创建一个3D（三维）数组，我们就要传递一个列表的列表的列表，
# 创建一个4D（四维）数组，那就是列表的列表的列表的列表，以此类推至多维数组。
# Slice Viewing: You can view individual 2D slices of the 3D array. Each slice represents a 2D matrix. (简化对多维数组的观察，或者是使用jupter notebook实时查看)
```

创建特殊的数组：

```python
# ---- Special Array
zero_matrix = np.zeros((2,3)) # 零矩阵
zeros_like_matrix = np.zeros_like(<matrix>) # 同尺寸零矩阵

ones_matrix = np.ones((2,4)) # 一矩阵
ones_like_matrix = np.ones_like(<matrix>) # 同尺一矩阵

identity_matrix = np.eye(5) # 单位矩阵

random_matrix = np.random.random((2,5)) # 随机矩阵
rand_normal_matrix = np.random.randn(2,5)

diagonal_matrix = np.diag([1, 2, 3, 4]) # 对角矩阵

```

访问与切片：

```python
# ---- Indexing and Simple Slicing
print(a[0, 1:4]) # 提取第一行中第二到第四列的元素
print(a[1:4, 0]) # 提取第一列中第二到第四行的元素
print(a[::2, ::2]) # 以步长为2提取行和列
print(a[:, 1]) # 提取第二列的所有元素
''' 注意：
				1. 当你从多维数组中提取出一个列或一行时，结果通常是一个一维数组，
					 不论是从原数组中提取出的行还是列。
				2. 0是第一位, -1是最后一位, 数字加1是真实索引数。
'''
```

数组属性：

```python
# ---- Array property
# 返回 a 的类型
print(type(a)) # >>><class 'numpy.ndarray'>

# 返回数组中元素的数据类型
print(a.dtype) # >>>int64

# 返回数组中元素的总数
print(a.size) # >>>25

# 返回数组的形状（尺寸）
print(a.shape) # >>>(5, 5)

# 返回数组的维度数，这里是 2，说明 a 是一个二维数组。
print(a.ndim) # >>>2

# 返回数组中每个元素的大小（以字节为单位），这里是 8，表示每个 int64 类型的元素占用8字节的内存。
print(a.itemsize) # >>>8

# 返回数组中所有元素占用的总字节数，这里是 200，计算方式是 元素个数(size) * 每个元素的大小(itemsize)，即 25 * 8 = 200 字节。
print(a.nbytes) # >>>200
```

## 使用数组

基础操作及运算：

```python
# ---- Basic operator
# NumPy默认使用行优先顺序，这意味着数组会先填满一行再移到下一行。
a = np.arange(25).reshape((5, 5))
b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78]).reshape((5,5))

# ---- 基本运算
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
print(a < b)
print(a > b)

# ---- dot product
print(a.dot(b)) # >> scalar number

# ---- 矩阵运算
matrix_product = a @ b # >> matrix
matrix_transpose = a.T # a.T

# ---- 特殊操作
print(a.sum()) 
print(a.min())
print(a.max())
```

常见内置函数：

```python
import numpy as np

# -----三角函数
np.sin(x)       # 正弦函数
np.cos(x)       # 余弦函数
np.tan(x)       # 正切函数
np.arcsin(x)    # 反正弦函数
np.arccos(x)    # 反余弦函数
np.arctan(x)    # 反正切函数
np.hypot(x, y)  # 计算直角三角形的斜边长度

# ----指数和对数函数
np.exp(x)       # 计算指数 e^x
np.exp2(x)      # 计算指数 2^x
np.log(x)       # 自然对数
np.log2(x)      # 以 2 为底的对数
np.log10(x)     # 以 10 为底的对数
np.log1p(x)     # 计算1+x的自然对数，用于小数值的精确计算

# ----算术运算
np.add(x, y)    # 加法
np.subtract(x, y) # 减法
np.multiply(x, y) # 乘法
np.divide(x, y)   # 除法
np.power(x, y)    # 幂运算 x^y
np.sqrt(x)        # 平方根
np.cbrt(x)        # 立方根

# ----统计函数
np.mean(array)   # 平均值
np.median(array) # 中位数
np.std(array)    # 标准差
np.var(array)    # 方差
np.min(array)    # 最小值
np.max(array)    # 最大值
np.sum(array)    # 总和
np.prod(array)   # 乘积

# ----复数函数
np.real(complex_array) # 返回复数的实部
np.imag(complex_array) # 返回复数的虚部
np.conj(complex_array) # 返回复数的共轭

# ----其他数学函数
np.abs(x)       # 绝对值
np.floor(x)     # 向下取整
np.ceil(x)      # 向上取整
np.round(x)     # 四舍五入

# ----线性代数函数
np.dot(x, y)          # 点积
np.cross(x, y)        # 叉积
np.linalg.inv(matrix) # 矩阵求逆
np.linalg.det(matrix) # 计算矩阵行列式
np.linalg.eig(matrix) # 计算方阵的特征值和右特征向量
```

## 索引进阶

花式索引，使用整数数组作为索引直接访问数组中的元素。

```python
# ---- fancy indexing
a = np.arange(0, 100, 10)
indices = [1, 5, -1] # 获取第2，6以及最后1个元素的索引
b = a[indices] # >>>[ 0 10 20 30 40 50 60 70 80 90]
print(b) # >>>[10 50 90]
```

布尔屏蔽，使用布尔数组进行索引，选择数组中特定的项。

```python
# ---- Boolean masking
'''布尔屏蔽是一个有用的功能，它允许我们根据我们指定的条件检索数组中的元素。'''
a = np.arange(10)
mask = (a % 2 == 0)
even_numbers = a[mask] # 获取数组内元素是否满足条件的bool集合
print(even_numbers) # >>>[0 2 4 6 8]
```

缺省索引， 使用省略号（...）来代表多个冒号表示的切片。

```python
# ---- Incomplete Indexing
'''不完全索引是从多维数组的第一个维度获取索引或切片的一种方便方法。
		例如，如果数组a=[1，2，3，4，5]，[6，7，8，9，10]，
		那么[3]将在数组的第一个维度中给出索引为3的元素，这里是值4。'''
a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(a[1, ...]) # 等同于 a[1,:,:] 或 a[1]
# >>>[[ 7  8  9]
#     [10 11 12]]
```

Where 函数， `np.where` 条件查找，可以找到数组中符合特定条件的元素的索引。

```python
# ---- where function
'''where() 函数是另外一个根据条件返回数组中的值的有效方法。
	 只需要把条件传递给它，它就会返回一个使得条件为真的元素的列表。'''
a = np.array([1, 2, 3, 4, 5])
# 找到数组中大于3的元素的索引
indices = np.where(a > 3)
print(indices) # >>>(array([3, 4]),)
print(a[indices]) # >>>[4 5]
```

## `Ndarray` 与内置的 `list` 交互

将 Python 列表转换为 NumPy 数组：

```python
# 使用 np.array() 函数将 Python 列表转换为 NumPy 数组：
# Python 列表
py_list = [1, 2, 3, 4, 5]

# 转换为 NumPy 数组
np_array = np.array(py_list)
```

将 NumPy 数组转换为 Python 列表：

```python
# 使用 NumPy 数组的 tolist() 方法，可以将数组转换回 Python 列表：
back_to_list = np_array.tolist()
```

在 NumPy 数组和 Python 列表之间进行迭代：

```python
# 在 Python 列表上进行迭代，就像在任何 Python 迭代对象上一样。对于 NumPy 数组也是一样：
# 迭代 NumPy 数组
for element in np_array:
    print(element)

# 迭代 Python 列表
for element in py_list:
    print(element)
```

在 NumPy 数组中使用 Python 列表：

```python
# 可以在 NumPy 的一些操作中直接使用 Python 列表，NumPy 会自动将其转换为 NumPy 数组：
# 使用 Python 列表创建 NumPy 数组
np_array = np.array([1, 2, 3, 4, 5])

# 使用 Python 列表作为索引
print(np_array[[0, 2, 4]])
```

在最后这个例子中，我们使用一个 Python 列表 `[0, 2, 4]` 作为索引来选择 `np_array` 中的元素。

### 注意事项

- 当在 NumPy 和 Python 列表之间转换时，要注意数据类型和复制的行为。NumPy 数组通常是同质的（即所有元素都有相同的数据类型），而 Python 列表可以是异质的。
- 另外，使用 `np.array()` 创建的 NumPy 数组是原始列表的一个拷贝，而不是引用。这意味着，原始列表的后续更改不会影响 NumPy 数组，反之亦然。