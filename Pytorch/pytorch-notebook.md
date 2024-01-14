# pytorch

https://github.com/pytorch/pytorch

[PyTorch](https://pytorch.org/)

```python
import torch 
```

## Basics

```python
# ---------------------torch constructor---------------------
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2],[3, 4],[5, 6]])
x = torch.eye(3)
x = torch.zeros(2,3)
x = torch.ones(2,3)
x = torch.rand(4,5) #  random numbers from a uniform distribution on the interval [0,1)
x = torch.randn(4,5) # returns a normal dis with mean 0 and var 1
x = torch.full((2, 3), 3.141592)
x = torch.zeros_like(x0) # with the same shape and type as a given tensor
x = torch.ones_like(x0)
x = x0.new_zeros(4, 5) # that create tensors the same type but possibly different shapes
x = torch.full((4,5),2.3).to(x0) #can take a tensor as an argument, in which case it casts to the datatype of the argument.
x = torch.arange(1,10,2) # return list in [1, 10) with step 2 

x = torch.empty(2, 3) # initial a empty tensor
```

## Data type

Even though PyTorch provides a large number of numeric datatypes, the most commonly used datatypes are:

- `torch.float32`: Standard floating-point type; used to store learnable parameters, network activations, etc. Nearly all arithmetic is done using this type.
- `torch.float16`: Used for mixed-precision arithmetic, usually on NVIDIA GPUs.
- `torch.int64`: Typically used to store indices.
- `torch.bool`: Stores boolean values: 0 is false and 1 is true.

```python
# ---------------------data type---------------------
data_type = x.dtype

# Let torch choose the datatype
x = torch.tensor([1, 2]) # integers
x = torch.tensor([1.0, 2.0]) # floats
x = torch.tensor([1.0, 2]) # mixed list

# tensor have specific data type
x = torch.tensor([1, 2], dtype=torch.float32) # specific data type 
x = torch.ones(1, 2, dtype=torch.uint8) # 8-bit (unsigned) integer

# Cast to 32-bit float
x = x0.float() 
x = x0.to(torch.float32) # to function

# Cast to 64-bit float
x = x0.double() 
x = x0.to(torch.float64) 

# Cast to 64-bit integer
x = x0.long()
```

## **Slice and indexing**

Similar to Python lists and numpy arrays, PyTorch tensors can be **sliced** using the syntax `start:stop` or `start:stop:step`. The `stop` index is always non-inclusive: it is the first element not to be included in the slice.

```python
# --------------------- access torch and indexing ---------------------
a = torch.tensor([0, 11, 22, 33, 44, 55, 66])
print(0, a)         # (0) Original tensor
print(1, a[2:5])    # (1) Elements between index 2 and 5
print(2, a[2:])     # (2) Elements after index 2
print(3, a[:5])     # (3) Elements before index 5
print(4, a[:])      # (4) All elements
print(5, a[1:5:2])  # (5) Every second element between indices 1 and 5
print(6, a[:-1])    # (6) All but the last element
print(7, a[-4::2])  # (7) Every second element, starting from the fourth-last
```

There are two common ways to access a single row or column of a tensor: using an integer will reduce the rank by one, and using a length-one slice will keep the same rank. 

`start:stop` → return same dimension as original matrix

`start` → return single dimension row or column

```python
# --------------------- two different indexing ---------------------
a = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print('\nTwo ways of accessing a single row:')
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

col_r1 = a[:, 1]    # Rank 1 view of the second col of a
col_r2 = a[:, 1:2]  # Rank 2 view of the second col of a
print('\nTwo ways of accessing a single column:')
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)
```

```python
Two ways of accessing a single row:
tensor([5, 6, 7, 8]) torch.Size([4])
tensor([[5, 6, 7, 8]]) torch.Size([1, 4])

Two ways of accessing a single column:
tensor([ 2,  6, 10]) torch.Size([3])
tensor([[ 2],
        [ 6],
        [10]]) torch.Size([3, 1])
```

Slicing a tensor returns a **view** into the same data, so modifying it will also modify the original tensor. To avoid this, you can use the `clone()` method to make a copy of a tensor. The behavior you're observing in the Python code with the PyTorch tensors `a`, `b`, and `c` is due to how PyTorch handles tensor slicing and cloning, and it's rooted in memory management and optimization.

```python
a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
b = a[0, 1:]
c = a[0, 1:].clone() # for numpy, it use copy
print('Before mutating:')
print(a)
print(b)
print(c)

a[0, 1] = 20   # a[0, 1] and b[0] point to the same element
b[1] = 30     # b[1] and a[0, 2] point to the same element
c[2] = 40     # c is a clone, so it has its own data
print('\nAfter mutating:')
print(a)
print(b)
print(c)

print(a.storage().data_ptr() == c.storage().data_ptr())
```

```python
Before mutating:
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])
tensor([2, 3, 4])
tensor([2, 3, 4])

After mutating:
tensor([[ 1, 20, 30,  4],
        [ 5,  6,  7,  8]])
tensor([20, 30,  4])
tensor([ 2,  3, 40])
False
```

**Tensor Slicing and Memory Sharing:**

- When you create `b` as a slice of `a` (i.e., `b = a[0, 1:]`), `b` does not allocate new memory for its elements. Instead, `b` becomes a view of `a`. This means `b` references the same memory location as `a`. This is an optimization technique to save memory and computation time, especially for large tensors.
- Any changes made to the elements of `b` will reflect in the corresponding elements of `a`, and vice versa, because they are essentially accessing the same data in memory.

**Cloning and Independent Memory Allocation:**

- When you create `c` as a clone of a slice of `a` (i.e., `c = a[0, 1:].clone()`), `c` gets its own separate memory allocation.
- Cloning creates a deep copy of the tensor's data. Therefore, `c` is a completely separate tensor from `a`, even though it was initially created with the same values.
- Any changes made to `c` will not affect `a` or `b`, and vice versa, as they are now stored in different memory locations.

**Memory Pointer Check:**

- The check `a.storage().data_ptr() == c.storage().data_ptr()` is used to verify whether `a` and `c` point to the same memory location.
- Since `c` is a clone and thus has its own memory, this check returns `False`, indicating different memory locations for `a` and `c`.

This approach of memory sharing for slices and independent memory for clones is common in many programming libraries that handle large data structures, as it provides a balance between performance (avoiding unnecessary data copying) and flexibility (allowing for independent manipulation when needed). 

[numpy.copy()，pytorch.clone()， copy.deepcopy(list)](https://www.notion.so/numpy-copy-pytorch-clone-copy-deepcopy-list-2a608a15c5dd4268b4f4befea66ec7c2?pvs=21)

```python
# --------------------------- modify subtensors --------------------------- 
a = torch.zeros(2, 4, dtype=torch.int64)
a[:, :2] = 1 # constant
a[:, 2:] = torch.tensor([[2, 3], [4, 5]]) # torch.tensor

```

Rearange a tensor with a `list` or a `torch.tensor()` 

```python

# --------------------------- Integer tensor indexing --------------------------- 
a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('Original tensor:')
print(a)

idx = [0, 0, 2, 1, 1]  # index arrays can be Python lists of integers
print('\nReordered rows:')
print(a[idx])

idx = torch.tensor([3, 2, 1, 0])  # index arrays can be int64 torch tensors
print('\nReordered columns:')
print(a[:, idx])
```

```python
Original tensor:
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])

Reordered rows:
tensor([[ 1,  2,  3,  4],
        [ 1,  2,  3,  4],
        [ 9, 10, 11, 12],
        [ 5,  6,  7,  8],
        [ 5,  6,  7,  8]])

Reordered columns:
tensor([[ 4,  3,  2,  1],
        [ 8,  7,  6,  5],
        [12, 11, 10,  9]])
```

More generally, given index arrays `idx0` and `idx1` with `N` elements each, `a[idx0, idx1]` is equivalent to:

```python
torch.tensor([
  a[idx0[0], idx1[0]],
  a[idx0[1], idx1[1]],
  ...,
  a[idx0[N - 1], idx1[N - 1]]
])

# --------------------------- List indexing --------------------------- 
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print('Original tensor:')
print(a)

idx0 = torch.arange(a.shape[0])  # Quick way to build [0, 1, 2, 3]
idx1 = torch.tensor([1, 2, 1, 0])
print('\nSelect one element from each row:')
print(a[idx0, idx1])
 
a[idx0, idx1] = 0 # Now set each of those elements to zero
print('\nAfter modifying one element from each row:')
print(a)
```

```python
Original tensor:
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])

Select one element from each row:
tensor([ 2,  6,  8, 10])

After modifying one element from each row:
tensor([[ 1,  0,  3],
        [ 4,  5,  0],
        [ 7,  0,  9],
        [ 0, 11, 12]])
```

## Reshaping operations

**View:** PyTorch provides many ways to manipulate the shapes of tensors. The simplest example isv `.view()`: This returns a new tensor with the same number of elements as its input, but with a different shape. As its name implies, a tensor returned by `.view()` shares the same data as the input, so changes to one will affect the other and vice-versa:

```python
x0 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print('Original tensor:')
print('shape:', x0.shape)

# Flatten x0 into a rank 1 vector of shape (8,)
x1 = x0.view(8)
print('\nFlattened tensor:')
print('shape:', x1.shape)

# Convert x1 to a rank 2 "row vector" of shape (1, 8)
x2 = x1.view(1, 8)
print('\nRow vector:')
print('shape:', x2.shape)

# Convert x1 to a rank 2 "column vector" of shape (8, 1)
x3 = x1.view(8, 1)
print('\nColumn vector:')
print('shape:', x3.shape)

# Convert x1 to a rank 3 tensor of shape (2, 2, 2):
x4 = x1.view(2, 2, 2)
print('\nRank 3 tensor:')
print('shape:', x4.shape)

# -1 operations
x5 = x1.view(-1)   # flatten()
x6 = x1.view(1,-1) # row_vector()
x7 = x1.view(-1,1) # col_vector()
print('\n-1 operations:')
print('shape:', x5.shape)
print('shape:', x6.shape)
print('shape:', x7.shape)
```

```python
Original tensor:
shape: torch.Size([2, 4])

Flattened tensor:
shape: torch.Size([8])

Row vector:
shape: torch.Size([1, 8])

Column vector:
shape: torch.Size([8, 1])

Rank 3 tensor:
shape: torch.Size([2, 2, 2])

-1 operations:
shape: torch.Size([8])
shape: torch.Size([1, 8])
shape: torch.Size([8, 1])
```

**Swapping axes:** Another common reshape operation you might want to perform is  transposing a matrix. You might be surprised if you try to transpose a matrix with `.view()`: The `view()` function takes elements in row-major order, so **you cannot transpose matrices with `.view()`**. In this case, the simplest such function is `.t()`, specificially for transposing matrices.

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('Original matrix:')
print(x)
print('\nTransposing with view DOES NOT WORK!')
print(x.view(3, 2))
print('\nTransposed matrix:')
print(torch.t(x))
print(x.t())
```

```python
Original matrix:
tensor([[1, 2, 3],
        [4, 5, 6]])

Transposing with view DOES NOT WORK!
tensor([[1, 2],
        [3, 4],
        [5, 6]])

Transposed matrix:
tensor([[1, 4],
        [2, 5],
        [3, 6]])
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

For tensors with more than two dimensions, we can use the function `torch.transpose` to swap arbitrary dimensions, or the `.permute` method to arbitrarily permute dimensions:

```python
# Create a tensor of shape (2, 3, 4)
x0 = torch.tensor([
     [[1,  2,  3,  4],
      [5,  6,  7,  8],
      [9, 10, 11, 12]],
     [[13, 14, 15, 16],
      [17, 18, 19, 20],
      [21, 22, 23, 24]]])
print('Original tensor:')
print(x0)
print('shape:', x0.shape)

# Swap axes 1 and 2; shape is (2, 4, 3)
x1 = x0.transpose(1, 2)
print('\nSwap axes 1 and 2:')
print(x1)
print(x1.shape)

# Permute axes; the argument (1, 2, 0) means:
# - Make the old dimension 1 appear at dimension 0;
# - Make the old dimension 2 appear at dimension 1;
# - Make the old dimension 0 appear at dimension 2
# This results in a tensor of shape (3, 4, 2)
x2 = x0.permute(1, 2, 0)
print('\nPermute axes')
print(x2)
print('shape:', x2.shape)
```

```python
# Create a tensor of shape (2, 3, 4)
x0 = torch.tensor([
     [[1,  2,  3,  4],
      [5,  6,  7,  8],
      [9, 10, 11, 12]],
     [[13, 14, 15, 16],
      [17, 18, 19, 20],
      [21, 22, 23, 24]]])
print('Original tensor:')
print(x0)
print('shape:', x0.shape)

# Swap axes 1 and 2; shape is (2, 4, 3)
x1 = x0.transpose(1, 2)
print('\nSwap axes 1 and 2:')
print(x1)
print(x1.shape)

# Permute axes; the argument (1, 2, 0) means:
# - Make the old dimension 1 appear at dimension 0;
# - Make the old dimension 2 appear at dimension 1;
# - Make the old dimension 0 appear at dimension 2
# This results in a tensor of shape (3, 4, 2)
x2 = x0.permute(1, 2, 0)
print('\nPermute axes')
print(x2)
print('shape:', x2.shape)
```

Some combinations of reshaping operations will fail with cryptic  errors. Here, `x0.transpose(1, 2)` transposes the second and third dimensions of `x0`, changing its shape to `(2, 4, 3)`. The subsequent `.view(8, 3)` attempts to reshape this tensor to a shape of `(8, 3)`. However, this operation fails because the tensor is not contiguous in memory after the transpose operation, leading to a `RuntimeError`. The `try-except` block catches this error and prints it. In this case, you can typically overcome these sorts of errors by either by calling `.contiguous()` before `.view()`, or by using `.reshape()` instead of `.view()`.

```python
x0 = torch.randn(2, 3, 4)

try:
  # This sequence of reshape operations will crash
  x1 = x0.transpose(1, 2).view(8, 3)
except RuntimeError as e:
  print(type(e), e)

# We can solve the problem using either .contiguous() or .reshape()
x1 = x0.transpose(1, 2).contiguous().view(8, 3) # .view() 处理连续张量
x2 = x0.transpose(1, 2).reshape(8, 3) # .reshape() 方法自动处理了非连续张量的问题
print('x1 shape: ', x1.shape)
print('x2 shape: ', x2.shape)
```

```python
<class 'RuntimeError'> view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
x1 shape:  torch.Size([8, 3])
x2 shape:  torch.Size([8, 3])
```

## Tensor operations

**Elementwise operations:** Basic mathematical functions operate elementwise on tensors, and are available as operator overloads, as functions in the `torch` module, and as instance methods on torch objects; all produce the same results:

```python
x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
y = torch.tensor([[5, 6, 7, 8]], dtype=torch.float32)

# Elementwise sum; all give the same result
print('Elementwise sum:')
print(x + y)
print(torch.add(x, y))
print(x.add(y))

# Elementwise difference
print('\nElementwise difference:')
print(x - y)
print(torch.sub(x, y))
print(x.sub(y))

# Elementwise product
print('\nElementwise product:')
print(x * y)
print(torch.mul(x, y))
print(x.mul(y))

# Elementwise division
print('\nElementwise division')
print(x / y)
print(torch.div(x, y))
print(x.div(y))

# Elementwise power
print('\nElementwise power')
print(x ** y)
print(torch.pow(x, y))
print(x.pow(y))
```

Torch also provides many standard mathematical functions; these are available both as functions in the `torch` module and as instance methods on tensors:

```python
x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

print('Square root:')
print(torch.sqrt(x))
print(x.sqrt())

print('\nTrig functions:')
print(torch.sin(x))
print(x.sin())
print(torch.cos(x))
print(x.cos())
```

**Reduction operations:** 

Like the elementwise operations above, most reduction operations are available both as functions in the `torch` module and as instance methods on `tensor` objects.

The simplest reduction operation is summation. We can use the `[.sum()](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Ftensors.html%23torch.Tensor.sum)` function (or eqivalently `[torch.sum](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Fgenerated%2Ftorch.sum.html)`) to reduce either an entire tensor, or to reduce along only one dimension of the tensor using the `dim` argument:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print('Original tensor:')
print(x)

print('\nSum over entire tensor:')
print(torch.sum(x))   # sum(x) aggragate over all elements
print(x.sum())

# We can sum over each row:
print('\nSum of each row:') 
print(torch.sum(x, dim=0)) # sum(x, dim = 0) aggragate over row elements
print(x.sum(dim=0)) 

# Sum over each column:
print('\nSum of each column:') # sum(x, dim = 1) aggragate over columns elements
print(torch.sum(x, dim=1))
print(x.sum(dim=1))
```

Other useful reduction operations include `[mean](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Ftorch.html%23torch.mean)`, `[min](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Ftorch.html%23torch.min)`, and `[max](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Ftorch.html%23torch.max)`.

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

# 使用 min 函数
# 整体最小值
overall_min = x.min()
print('Overall minimum:', overall_min.item())

# 沿特定维度的最小值
col_min_vals, col_min_idxs = x.min(dim=0) # idxs 是沿列索引， dim=1 -> 沿行索引
print('Minimum along each column:', col_min_vals)
print('Indices of min along each column:', col_min_idxs)

# 使用 max 函数
# 整体最大值
overall_max = x.max()
print('\nOverall maximum:', overall_max.item())

# 沿特定维度的最大值
col_max_vals, col_max_idxs = x.max(dim=0) # idxs 是沿列索引， dim=1 -> 沿行索引 
print('Maximum along each column:', col_max_vals)
print('Indices of max along each column:', col_max_idxs)

# 使用 mean 函数
# 整体平均值
overall_mean = x.mean()
print('\nOverall mean:', overall_mean.item())

# 沿特定维度的平均值
col_mean_vals = x.mean(dim=0)
print('Mean along each column:', col_mean_vals)
```

## Matrix operations

PyTorch provides a number of linear algebra functions that compute different types of vector and matrix products. The most commonly used are (`torch.matmul` is often used for generality):

[batch matrix](https://www.notion.so/batch-matrix-d059a21abe7d4f7d82d05dddc039985e?pvs=21)

```python
a = torch.tensor([2, 3], dtype=torch.float32)  # 向量
b = torch.tensor([4, 5], dtype=torch.float32)  # 向量
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 矩阵
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)  # 矩阵
C = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)  # 矩阵（用于加法操作）

# torch.dot - 向量内积
dot_product = torch.dot(a, b)
print("torch.dot(a, b):", dot_product)

# torch.mm - 矩阵乘法
matmul = torch.mm(A, B)
print("torch.mm(A, B):", matmul)

# torch.mv - 矩阵与向量乘法
matvec = torch.mv(A, a)
print("torch.mv(A, a):", matvec)

# torch.addmm - 矩阵乘法 + 偏差
addmm = torch.addmm(C, A, B, alpha=1, beta=1)
print("torch.addmm(C, A, B, alpha=1, beta=1):", addmm)

# 创建3D张量以示范批量操作
A_batch = A.repeat(2, 1, 1)  # 创建一个[批量矩阵](https://www.notion.so/batch-matrix-d059a21abe7d4f7d82d05dddc039985e?pvs=21)
B_batch = B.repeat(2, 1, 1)  # 创建另一个批量矩阵c

# torch.bmm - 批量矩阵乘法
batch_matmul = torch.bmm(A_batch, B_batch)
print("torch.bmm(A_batch, B_batch):", batch_matmul)

# torch.matmul - 通用矩阵乘法
matmul_general = torch.matmul(A, B)
print("torch.matmul(A, B):", matmul_general)
```

## Broadcasting

Broadcasting is a powerful mechanism that allows PyTorch to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller tensor and a larger tensor, and we want to use the smaller tensor multiple times to perform some operation on the larger tensor.

Broadcasting two tensors together follows these rules:

1. If the tensors do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
2. The two tensors are said to be *compatible* in a dimension if they have the same size in the dimension, or if one of the tensors has size 1 in that dimension.
3. The tensors can be broadcast together if they are compatible in all dimensions.
4. After broadcasting, each tensor behaves as if it had shape equal to the elementwise maximum of shapes of the two input tensors.
5. In any dimension where one tensor had size 1 and the other tensor had size greater than 1, the first tensor behaves as if it were copied along that dimension.

```python
# ------------------------ implicit broadcasting -------------------------
x = torch.tensor([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = torch.tensor([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)

# ------------------------ explicit broadcasting -------------------------
xx, vv = torch.broadcast_tensors(x, v)
y = xx + vv
```

Broadcasting can let us easily implement many different operations. For example we can compute an outer product of vectors:

```python
# Compute outer product of vectors
v = torch.tensor([1, 2, 3])  # v has shape (3,)
w = torch.tensor([4, 5])     # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
print(v.view(3,1) * w)
```

Multiply a tensor by a set of constants:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # x has shape (2, 3)
c = torch.tensor([1, 10, 11, 100])      # c has shape (4)
print('Here is the matrix:')
print(x)
print('\nHere is the vector:')
print(c)

# We do the following:
# 1. Reshape c from (4,) to (4, 1, 1)
# 2. x has shape (2, 3). Since they have different ranks, when we multiply the
#    two, x behaves as if its shape were (1, 2, 3)
# 3. The result of the broadcast multiplication between tensor of shape
#    (4, 1, 1) and (1, 2, 3) has shape (4, 2, 3)
# 4. The result y has shape (4, 2, 3), and y[i] (shape (2, 3)) is equal to
#    c[i] * x
y = c.view(-1, 1, 1) * x
print('\nMultiply x by a set of constants:')
print(y)
```

## Running on GPU

One of the most important features of PyTorch is that it can use graphics processing units (GPUs) to accelerate its tensor operations.

```python
import torch 

# ------------- check whether PyTorch is configured to use GPUs -------------
if torch.cuda.is_available:
	print('Pytorch can use GPUs')
else:
	print('Pytorch cannot use GPUs')
```

Just as with datatypes, we can use the `[.to()](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fpytorch.org%2Fdocs%2F1.1.0%2Ftensors.html%23torch.Tensor.to)` method to change the device of a tensor. We can also use the convenience methods `.cuda()` and `.cpu()` methods to move tensors between CPU and GPU.

```python
# Construct a tensor on the CPU
x0 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print('x0 device:', x0.device)

# Move it to the GPU using .to()
x1 = x0.to('cuda')
print('x1 device:', x1.device)

# Move it to the GPU using .cuda()
x2 = x0.cuda()
print('x2 device:', x2.device)

# Move it back to the CPU using .to()
x3 = x1.to('cpu')
print('x3 device:', x3.device)

# Move it back to the CPU using .cpu()
x4 = x2.cpu()
print('x4 device:', x4.device)

# We can construct tensors directly on the GPU as well
y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64, device='cuda')
print('y device / dtype:', y.device, y.dtype)

# Calling x.to(y) where y is a tensor will return a copy of x with the same
# device and dtype as y
x5 = x0.to(y)
print('x5 device / dtype:', x5.device, x5.dtype)
```