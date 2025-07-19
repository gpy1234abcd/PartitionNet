import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def rotate_image(image, angle):
    """旋转图像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def perspective_transform(image, tilt_x=0, tilt_y=0, perspective_factor=0.1):
    """透视变换"""
    h, w = image.shape[:2]

    tilt_x_rad = np.radians(tilt_x)
    tilt_y_rad = np.radians(tilt_y)

    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    x_offset_top = w * perspective_factor * np.sin(tilt_x_rad)
    x_offset_bottom = -x_offset_top
    y_offset_left = h * perspective_factor * np.sin(tilt_y_rad)
    y_offset_right = -y_offset_left

    dst_points = np.float32([
        [x_offset_top, y_offset_left],
        [w + x_offset_top, y_offset_right],
        [w + x_offset_bottom, h + y_offset_right],
        [x_offset_bottom, h + y_offset_left]
    ])

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(image, perspective_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return transformed


def affine_transform(image, angle=0, scale_x=1.0, scale_y=1.0, shear_x=0, shear_y=0):
    """仿射变换"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    transform_matrix = np.float32([
        [scale_x * cos_a - shear_y * sin_a, -scale_x * sin_a - shear_y * cos_a, 0],
        [scale_y * sin_a + shear_x * cos_a, scale_y * cos_a - shear_x * sin_a, 0]
    ])

    transform_matrix[0, 2] = center[0] - center[0] * transform_matrix[0, 0] - center[1] * transform_matrix[0, 1]
    transform_matrix[1, 2] = center[1] - center[0] * transform_matrix[1, 0] - center[1] * transform_matrix[1, 1]

    transformed = cv2.warpAffine(image, transform_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return transformed


def generate_robustness_test_images(image):
    """
    生成5个关键几何变换

    Returns:
        dict: {变换名称: (变换图像, 显示标签)}
    """
    test_cases = {}

    # 1. 旋转变换系列
    test_cases["I_roll_15"] = (
        rotate_image(image, 15),
        "$I$_roll_15°"
    )

    test_cases["I_roll_30"] = (
        rotate_image(image, 30),
        "$I$_roll_30°"
    )

    # 2. 透视变换系列 - 模拟地面倾斜
    test_cases["I_pitch_15"] = (
        perspective_transform(image, 15, 0, 0.3),
        "$I$_pitch_15°"
    )

    test_cases["I_yaw_15"] = (
        perspective_transform(image, 0, 15, 0.3),
        "$I$_yaw_15°"
    )

    # 3. 复合变换 - 模拟相机角度变化
    test_cases["I_combined"] = (
        affine_transform(image, 20, 0.9, 1.1, 0.1, 0),
        "$I$_combined"
    )

    return test_cases


def save_images_to_directory(original_image, test_cases, output_dir, original_filename):
    """
    将原图和变换后的图像保存到指定目录

    Args:
        original_image: 原始图像
        test_cases: 变换后的图像字典
        output_dir: 输出目录路径
        original_filename: 原始文件名（不包含路径）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取原始文件名（不包含扩展名）
    base_name = os.path.splitext(original_filename)[0]
    file_extension = os.path.splitext(original_filename)[1]

    # 保存原始图像
    original_save_path = os.path.join(output_dir, f"{base_name}_original{file_extension}")
    cv2.imwrite(original_save_path, original_image)
    print(f"Saved original image: {original_save_path}")

    # 保存变换后的图像
    saved_files = []
    for name, (transformed_img, label) in test_cases.items():
        save_path = os.path.join(output_dir, f"{base_name}_{name}{file_extension}")
        cv2.imwrite(save_path, transformed_img)
        saved_files.append(save_path)
        print(f"Saved transformed image: {save_path}")

    return saved_files


def visualize_six_images(image, test_cases):
    """
    可视化6个图像：原图 + 5个变换，按照2行3列排列
    优化版本：减少留白，图像更饱满
    """
    # 创建子图，减少图像间距
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 调整子图之间的间距，减少留白但保留标题空间
    plt.subplots_adjust(
        left=0.03,  # 左边距
        right=0.97,  # 右边距
        top=0.92,  # 上边距（为上排标题留空间）
        bottom=0.12,  # 下边距（为下排标题留空间）
        wspace=0.08,  # 子图间水平间距
        hspace=0.25  # 子图间垂直间距（为标题留足够空间）
    )

    # 将2D数组展平以便索引
    axes = axes.flatten()

    # 处理图像颜色空间转换
    def convert_to_rgb(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return img

    # 显示原始图像（第一个位置）
    original_rgb = convert_to_rgb(image)
    axes[0].imshow(original_rgb)
    axes[0].set_title("$I$ (Original)", fontsize=28, fontfamily='Palatino Linotype', pad=8)
    axes[0].axis('off')

    # 显示5个变换图像
    for i, (name, (transformed_img, label)) in enumerate(test_cases.items()):
        transformed_rgb = convert_to_rgb(transformed_img)

        axes[i + 1].imshow(transformed_rgb)
        axes[i + 1].set_title(f"{label}", fontsize=28, fontfamily='Palatino Linotype', pad=8)
        axes[i + 1].axis('off')

    plt.show()


def show_transformations_grid(image_path, output_dir=None):
    """
    主函数：加载图像并显示2行3列的变换网格，同时保存图像

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录路径，如果为None则不保存
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    print(f"Image loaded successfully, shape: {image.shape}")

    # 生成5个关键变换
    test_cases = generate_robustness_test_images(image)

    print(f"\nGenerated {len(test_cases)} geometric transformation test cases:")
    for i, (name, (_, label)) in enumerate(test_cases.items(), 1):
        print(f"{i}. {name}: {label}")

    # 显示2行3列的网格
    print("\nShowing 2x3 grid: Camera angles and tilted ground planes simulation...")
    visualize_six_images(image, test_cases)

    # 如果指定了输出目录，则保存图像
    if output_dir is not None:
        original_filename = os.path.basename(image_path)
        print(f"\nSaving images to directory: {output_dir}")
        saved_files = save_images_to_directory(image, test_cases, output_dir, original_filename)
        print(f"\nTotal saved files: {len(saved_files) + 1}")  # +1 for original image

    return test_cases


# 使用示例
if __name__ == "__main__":
    # 修改为你的图像路径
    image_path = "E://gpy//centernet_relative_kitti-paixu//img_out//7.png"

    # 修改为你想要保存图像的输出目录
    output_directory = "E://gpy//centernet_relative_kitti-paixu//img_out//transformed_images"

    # 调用主函数，传入输出目录参数
    show_transformations_grid(image_path, output_directory)