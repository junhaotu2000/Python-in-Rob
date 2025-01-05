import time
from datasets import load_dataset

# 启用流式加载
dataset = load_dataset("imagenet-1k", streaming=True)

breakpoint()
# 选择训练集
train_stream = dataset["train"]
# 定义样本计数
num_samples = 10000  # 设置需要加载的样本数

# 开始计时
start_time = time.time()

# 测试流式加载
count = 0
for example in train_stream:
    # 处理图像和标签
    image = example["image"]
    label = example["label"]
    print(image, label)

    # 打印部分进度
    if count % 10 == 0:
        print(f"Loaded {count} samples...")

    # 增加计数
    count += 1

    # 达到指定样本数时停止
    if count >= num_samples:
        break

# 结束计时
end_time = time.time()

# 输出加载速度
time_taken = end_time - start_time
print(f"Loaded {num_samples} samples in {time_taken:.2f} seconds.")
print(f"Average speed: {num_samples / time_taken:.2f} samples/second.")
