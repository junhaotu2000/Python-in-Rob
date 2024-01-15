# Numpy

[NumPy](https://numpy.org/)

Numpy 作为python的通用计算包，功能强大又足够简单, 因此往往被用来做基础运算；但Numpy不支持自动微分，也不能调用GPU加速运算。相关需求可参见其他相关计算包，如`cupy`, `pytorch`, `jax`, `tensorflow`, `PaddlePaddle`.

入门教程：[https://www.numpy.org.cn](https://www.numpy.org.cn) 

官方文档：[https://numpy.org/](https://numpy.org/)

详细应用和底层实现： [https://www.labri.fr/perso/nrougier/from-python-to-numpy/](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)

NumCpp - 基于Numpy的C++ 库： [https://dpilger26.github.io/NumCpp/doxygen/html/index.html](https://dpilger26.github.io/NumCpp/doxygen/html/index.html)

```python
import numpy as np    # 导入numpy
print(np.__version__) # 检查numpy版本
np.show_config()      # 显示底层配置信息 - 使用函数库与运行指令集
```

## 数组基础

数组创建 

```python
# ---- ND Array (输入为list)
# 为了创建ndarray，我们传递一个list给array()函数。
a = np.array([0, 1, 2, 3, 4])

# 为了创建一个2D（二维）数组，我们传递一个列表的列表给array()函数。
a_2d = np.array([[11, 12, 13, 14, 15],
                 [31, 32, 33, 34, 35]])

# 创建一个3D数组，我们就要传递一个列表的列表的列表,
a_3d = np.array([[[11, 12, 13, 14, 15],       
                  [31, 32, 33, 34, 35]]])
'''
1. 创建一个4D数组，那就是列表的列表的列表的列表，以此类推至多维数组。
2. 多维数组常常难以观察位置，可使用interactive windows简化对多维数组的观察，或者是使用jupter notebook variable实时查看。
'''

####################################################################################
# ---- Standard and Special Array 
b = np.arange(5)               # 输出自然数组成的数量为定义指的数组
c = np.arange(0, 1, 0.2)       # 等间隔 （创建一个从0开始等间隔为0.2的数组，直到不超过1）
d = np.linspace(0, 1, 5)       # 单位间隔 （创建一个从0到1含5个等间隔数的数组）

zeros_matrix = np.zeros((2,3)) # 零矩阵， 常用来初始化已知尺寸的矩阵
ones_matrix = np.ones((2,4))                    # 一矩阵
zeros_like_matrix = np.zeros_like(<matrix>)     # 基于输入矩阵创建同尺寸零矩阵
ones_like_matrix = np.ones_like(<matrix>)       # 基于输入矩阵创建同尺寸一矩阵
full_matrix = np.full((3,4), np.pi)             # 全尺寸相同元素矩阵 
empty_matrix = np.empty((3,3))                  # 空矩阵

identity_matrix = np.eye(5)                     # 单位矩阵
diagonal_matrix = np.diag([1, 2, 3, 4], k = -1) # 对角矩阵, k = ‘到对角线的距离’

####################################################################################
# ---- Common Random Array
# 矩阵中的每个元素都是从 [0, 1) 范围内均匀分布的随机浮点数
random_matrix = np.random.random((rows, cols)) 
rand_matrix = np.random.rand(rows, cols) 

# 矩阵中的每个元素都是从平均值为 0，标准差为 1 的正态分布中随机抽取的
randn_matrix = np.random.randn(rows, cols) 

# 矩阵中的每个元素都是从 low（含）到 high（不含）的整数中随机选择的
randint_matrix = np.random.randint(low, high, (rows, cols))  

# 这个数是从 low（含）到 high（不含）的范围内均匀分布的, 这个函数只返回一个单独的数值
rand_uniform = np.random.uniform(low, high) 
```

数组属性

```python
# ---- Array Properties and operations
a = np.random.random((5,5))

####################################################################################
# 返回 a 的类型
print(type(a)) # >>> <class 'numpy.ndarray'>

# 设置专门数据类型
color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)])

####################################################################################
# 返回数组中元素的数据类型
print(a.dtype) # >>> float64

# 将old_array的数据类型转换为new_type
new_array = old_array.astype(new_type)
'''
1. Numpy 不可与[]混用
2. 当在 NumPy 和 Python 列表之间转换时，要注意数据类型和复制的行为。NumPy 数组通常是同质的（即所有元素都有相同的数据类型），而 Python 列表可以是异质的。
'''

# 展示了不同 NumPy 数据类型的最小值、最大值和机器精度（epsilon）
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min) # 整数类型最小值， 通过2^(n-1)计算
    print(np.iinfo(dtype).max) # 整数类型最大值， 通过2^(n-1)计算
    
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min) # 浮点数类型最小值， 无法手算
    print(np.finfo(dtype).max) # 浮点数类型最小值， 无法手算
    print(np.finfo(dtype).eps) # 机器精度 - 机器精度是能够被浮点数系统区分的最小的正数ε，使得 1.0+ε不等于1.0。
'''
1. 使用 numpy.array() 创建数组而不指定数据类型时, 整数默认为 'np.int64'，浮点数默认为 'np.float64'
2. 索引使用的是 Python 的内置 int 类型，而不是 NumPy 的特定整数类型。Python 的 int 类型通常与机器的原生整数类型对齐，这意味着在32位系统上它可能是32位，而在64位系统上则是64位。

3. 在涉及浮点运算的函数中，如果未指定数据类型，NumPy 通常会使用 'np.float64'
4. NumPy 中的一些数值常量（例如 np.pi 和 np.e）通常是 np.float64 类型
5. 在执行操作时，如果涉及多种不同的数据类型，NumPy 会按照一定的规则提升（转换）数据类型以保持精度。这通常意味着向更精确的类型转换（例如，从 np.int32 到 np.float64）。
'''

####################################################################################
# 返回数组的形状（尺寸）
print(a.shape) # >>> (5, 5) 
print(a.shape[0]) # rows size
print(a.shape[1]) # cols size
'''
一维数组返回（len, ）, 无确定形态（即不为column vector也不为row vector）
可用reshape(-1,1)切换为（rows, 1）
'''
# 返回数组的维度数，即秩(rank)
print(a.ndim) # >>> 2

# 返回数组中元素的总数
print(a.size) # >>> 25

# 返回数组中每个元素的大小（以字节为单位），这里是 8，表示每个 int64 类型的元素占用8字节的内存。
print(a.itemsize) # >>> 8

# 返回数组中所有元素占用的总字节数，这里是 200，计算方式是 元素个数(size) * 每个元素的大小(itemsize)，即 25 * 8 = 200 字节。
print(a.nbytes) # >>> 200
```

Numpy 与 List 

```python
py_list = [1, 2, 3, 4, 5]

# 使用 np.array() 函数将 Python 列表转换为 NumPy 数组
np_array = np.array(py_list)

# 使用 NumPy 数组的 tolist() 方法，可以将数组转换回 Python 列表：
back_to_list = np_array.tolist()

 可以在 NumPy 的一些操作中直接使用 Python 列表，NumPy 会自动将其转换为 NumPy 数组：
# 使用 Python 列表创建 NumPy 数组
np_array = np.array([1, 2, 3, 4, 5])

# 使用 Python 列表作为索引
print(np_array[[0, 2, 4]])

'''
1. 另外，使用 np.array() 创建的 NumPy 数组是原始列表的一个拷贝，而不是引用。
这意味着，原始列表的后续更改不会影响 NumPy 数组，反之亦然。

2. 在迭代中添加元素，一般使用list.append(), 然后在loop结束后np.array(list);
 因为list是动态内存而numpy不是。

3. Numpy 和 List 都是可迭代容器。

'''
```

## 常用操作

复制与视图

```python
# 创建array的一个新视图，可能使用不同的数据类型dtype，但视图与原始数组共享相同的数据。
# dtype is an optional input
array_view = array.view(dtype)
'''
修改array_view中的一个元素也改变了original_array中相应的元素。这展示了NumPy视图的一个重要特性：
它们与原始数组共享数据。因此，任何对视图所做的修改都会直接影响到原始数组。
这种行为在处理大型数据集时特别有用，因为它避免了不必要的数据复制，节省内存和计算资源。

'''

# 在 NumPy 中用于创建数组的一个深拷贝(整体)。这意味着它创建了原始数组的一个完全独立的副本，包括数据。
array_copy = np.copy(array)
'''
注意，np.copy 是浅层拷贝，不会拷贝数组中的对象元素。这对于包含 Python 对象的数组来说非常重要。
新数组将包含相同的对象，如果该对象可以修改（可变），则可能导致意外。

'''

# 要确保对象数组中的所有元素都被深拷贝(逐个元素)，请使用 copy.deepcopy：
import copy
c = copy.deepcopy(a)
```

形状与维度改变

```python
# 将array沿各个维度重复(rows, cols)次以构造新数组。
tiled_array = np.tile(array, (rows, cols))

# 将array1, array2等沿指定轴axis连接起来, 0 - row direction, 1 - col direction
concatenated_array = np.concatenate((array1, array2), axis = 0/1)

# 将array1, array2等数组垂直堆叠（沿轴0）
stacked_array = np.vstack((array1, array2, ...))

# 将array1, array2等数组水平堆叠（沿轴1）
stacked_array = np.hstack((array1, array2, ...))

# ary: 要分割的数组。
# indices_or_sections: 如果是一个整数，那么数组将被分割成等长的子数组，前提是它们可以等分。
# 如果是一个整数数组，这些整数指示在哪里分割数组。
# axis: 按照哪个轴分割数组，默认为0，即沿着第一个轴分割。
new_arrays = numpy.split(ary, indices_or_sections, axis=0)
np.split(x, 3)  # 将数组分割成3个等长的子数组
np.split(y, [1, 4])  # 在索引1和4的位置分割数组

# change the shape of the original matrix 
new_matrix = matrix.reshape(3,2) # change to specific shape
np.reshape(-1, 1)                # column array
np.reshape(-1)                   # row array
np.reshape(-1,N)                 # -1 means 'whatever it takes' to flatten.

np.flatten()              # flatten the ndarray into a array
array.ravel()             # 与flatten类似，但返回的是数组的视图(反转数组)
np.expand_dims(a, axis=0) # 在指定轴上增加一个维度
np.squeeze()              # 移除数组中的单维度条目
array[::-1]               # reverse the order of a array
```

特殊操作

```python
np.sort(array)                 # sort the input arrat in increasing order 
indics = np.argsort(array)     # 返回排序后的索引
max_indics = np.argmax()       # find and return max indics
min_indics = np.argmin()       # find and return min indics

np.save()    # np.save('data.npy',array), .npy is not human friendly
np.load()    # np.load('data.npy'), load .npy file
np.savetxt() # np.savetxt('data.csv',array) .csv can be opened and human readable
np.loadtxt   # np.loadtxt('path_to_file.csv'), high reso load

# explictly define a numpy function to perform some operation on a array
# 定义一个普通的 Python 函数,
def square(x):
    return x * x
# 使用 np.vectorize() 创建一个向量化函数
vectorized_square = np.vectorize(square) 
# 使用向量化函数
result = vectorized_square(array)
'''虽然 np.vectorize() 可以使代码更加简洁，但它并不总是提高性能。
实际上，对于大型数组，使用 NumPy 自身的向量化操作通常会更有效率。'''
```

## 数组运算

基础运算与内置函数

```python
# ----数值处理
np.abs(x)       # 绝对值
np.floor(x)     # 向下取整
np.ceil(x)      # 向上取整
np.round(x)     # 四舍五入
np.trunc()      # 向零截断

# ----线性代数
a @ b                  # 矩阵乘法
a.T                    # 转置
np.dot(x, y)           # 点积
np.cross(x, y)         # 叉积
np.linalg.inv(matrix)  # 矩阵求逆
np.linalg.det(matrix)  # 计算矩阵行列式
np.linalg.eig(matrix)  # 计算方阵的特征值和右特征向量
np.linalg.norm(matrix) # normalize

# ----算术运算
np.add(x, y)    # 加法
np.subtract(x, y) # 减法
np.multiply(x, y) # 乘法
np.divide(x, y)   # 除法
np.power(x, y)    # 幂运算 x^y
np.sqrt(x)        # 平方根
np.cbrt(x)        # 立方根

# ----统计
np.mean(array)   # 平均值 . mean(
np.median(array) # 中位数
np.std(array)    # 标准差
np.var(array)    # 方差
np.min(array)    # 最小值 .min()
np.max(array)    # 最大值 .max()
np.sum(array)    # 总和  .sum()
```

```python
# ---- 常数
np.pi, np.e
np.nan  # 用于表示不可表示的数值结果

# ---- 基本运算
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b)
print(a % b)
print(a ** 2)

# -----三角函数
np.sin(x)       # 正弦函数
np.cos(x)       # 余弦函数
np.tan(x)       # 正切函数
np.arcsin(x)    # 反正弦函数
np.arccos(x)    # 反余弦函数
np.arctan(x)    # 反正切函数

# ----指数和对数函数
np.exp(x)       # 计算指数 e^x
np.exp2(x)      # 计算指数 2^x
np.log(x)       # 自然对数
np.log2(x)      # 以 2 为底的对数
np.log10(x)     # 以 10 为底的对数

# ----复数
np.real(complex_array) # 返回复数的实部
np.imag(complex_array) # 返回复数的虚部
np.conj(complex_array) # 返回复数的共轭
```

广播 (broadcasting)

```python
##### Refer to the pytorch, they are using same broadcasting system ##### 
```

## 索引及切片

Numpy ndarray can be **sliced** using the syntax `start:stop` or `start:stop:step`. The `stop` index is always non-inclusive: it is the first element not to be included in the slice.

```python
# ---- 索引及切片
print(a[1,1])      # 单步索引， 首先 a[1] 选中 a 的第二行，然后 [1] 从这个选中行中选择第二个
print(a[1][1])     # 两步索引，在 NumPy 中更为常见，因为它更直接且更适用于多维数组
print(a[0, 1:4])   # 提取第一行中第二到第四列的元素
print(a[1:4, 0])   # 提取第一列中第二到第四行的元素
print(a[::2, ::2]) # 以步长为2提取行和列
print(a[:, 1])     # 提取第二列的所有元素
''' 注意：
				1. 当你从多维数组中提取出一个列或一行时，结果通常是一个一维数组，
					 不论是从原数组中提取出的行还是列。
				2. 0是第一位, -1是最后一位, 数字加1是真实索引数。
'''
# ---- 花式索引, 使用整数数组作为索引直接访问数组中的元素
a = np.arange(0, 100, 10)
indices = [1, 5, -1]    # 获取第2，6以及最后1个元素的索引
b = a[indices]          # >>>[ 0 10 20 30 40 50 60 70 80 90]
print(b)                # >>>[10 50 90]

# ---- 布尔屏蔽, 使用布尔数组进行索引，选择数组中特定的项
'''布尔屏蔽是一个有用的功能，它允许我们根据我们指定的条件检索数组中的元素。'''
a = np.arange(10)
mask = (a % 2 == 0)
even_numbers = a[mask]   # 获取数组内元素是否满足条件的bool集合
print(even_numbers)      # >>>[0 2 4 6 8]

# ---- 缺省索引， 使用省略号（...）来代表多个冒号表示的切片
'''不完全索引是从多维数组的第一个维度获取索引或切片的一种方便方法。
		例如，如果数组a=[1，2，3，4，5]，[6，7，8，9，10]，
		那么[3]将在数组的第一个维度中给出索引为3的元素，这里是值4。'''
a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(a[1, ...])    # 等同于 a[1,:,:] 或 a[1]
# >>>[[ 7  8  9]
#     [10 11 12]]

# ---- 条件索引，可以找到数组中符合特定条件的元素的索引
'''where() 函数是另外一个根据条件返回数组中的值的有效方法。
	 只需要把条件传递给它，它就会返回一个使得条件为真的元素的列表。'''
a = np.array([1, 2, 3, 4, 5])

# 找到数组中大于3的元素的索引
indices = np.where(a > 3)
print(indices)    # >>>(array([3, 4]),)
print(a[indices]) # >>>[4 5]
```