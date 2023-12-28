import os
import cv2
import dlib

def detect_and_crop_faces(input_path, output_folder):
    # 初始化人脸检测器
    face_detector = dlib.get_frontal_face_detector()

    # 读取输入图像
    image = cv2.imread(input_path)

    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 在图像中检测人脸
    faces = face_detector(gray)

    # 遍历每个检测到的人脸
    for i, face in enumerate(faces):
        # 获取人脸的边界框坐标
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # 裁剪出人脸区域
        face_image = image[y:y+h, x:x+w]

        # 构建新的文件名
        output_filename = f"{output_folder}/Yixin_LIU_C{i+1}.png"

        # 保存裁剪后的人脸图像
        cv2.imwrite(output_filename, face_image)
        print(f"Face {i+1} saved as {output_filename}")

def batch_process_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 遍历每个图像文件
    for image_file in image_files:
        # 构建输入图像的完整路径
        input_image_path = os.path.join(input_folder, image_file)

        # 调用人脸检测和裁剪函数
        detect_and_crop_faces(input_image_path, output_folder)

if __name__ == "__main__":
    
    input_images_folder = "D:\OneDrive - Umich\Desktop\Deep fake\Figures\Yixin_LIU\Original" # 指定包含图像文件的文件夹路径 
    output_faces_folder = "D:\OneDrive - Umich\Desktop\Deep fake\Figures\Yixin_LIU\Test" # 指定输出路径

    # 调用函数进行批量处理
    batch_process_images(input_images_folder, output_faces_folder)
    print('Detected was finished!!!')
