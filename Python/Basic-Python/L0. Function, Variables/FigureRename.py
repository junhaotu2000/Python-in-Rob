import os


def rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 确保文件夹路径以斜杠结尾
    if not folder_path.endswith("/"):
        folder_path += "/"

    # 初始化计数器
    count = 1

    # 循环处理每个文件
    for file_name in files:
        # 获取文件的完整路径
        old_path = folder_path + file_name

        # 检查是否是文件
        if os.path.isfile(old_path):
            # 获取文件的扩展名
            file_extension = os.path.splitext(file_name)[1]

            # 构建新的文件名
            new_name = f"Yixin_LIU_HR{count}.png"

            # 构建新的文件路径
            new_path = folder_path + new_name

            # 处理文件名冲突
            while os.path.exists(new_path):
                count += 1
                new_name = f"Yixin_LIU_C{count}.png"
                new_path = folder_path + new_name

            # 重命名文件
            os.rename(old_path, new_path)

            # 增加计数器
            count += 1
    return count


if __name__ == "__main__":
    # 指定包含图像文件的文件夹路径
    folder_path = "D:\OneDrive - Umich\Desktop\Deep fake\Figures\Yixin_LIU\Restoration"

    # 调用函数进行重命名
    count = rename_images(folder_path)
    print(f"\n{count} was renamed!!!")
