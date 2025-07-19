from PIL import Image
import os


# 定义函数将PNG文件转为JPEG文件
def convert_to_jpeg(filepath):
    # 打开PNG文件
    image = Image.open(filepath)

    # 获取文件名（不包含后缀）
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # 构建新的JPEG文件路径
    new_filename = f"{filename}.jpg"
    jpeg_filepath = os.path.join(os.path.dirname(filepath), new_filename)

    # 保存为JPEG文件
    image.save(jpeg_filepath, "JPEG")

    print("已成功将", filepath, "转为", jpeg_filepath)


# 遍历指定目录中所有的PNG文件并转为JPEG
directory = './VOCdevkit/VOC2012/JPEGImages'  # 修改为你想要操作的目录路径
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".png"):
            png_filepath = os.path.join(root, file)
            convert_to_jpeg(png_filepath)