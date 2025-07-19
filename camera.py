import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    生成8个关键几何变换，专门应对审稿人关切

    Returns:
        dict: {变换名称: (变换图像, 描述)}
    """

    test_cases = {}

    # 1. 旋转变换系列 - 应对 "camera angles"
    test_cases["roll_15"] = (
        rotate_image(image, 15),
        "Rotate 15deg - Simulate slight camera roll angle change"
    )




    # cv2.imwrite("E://gpy//centernet_relative_kitti-paixu//img_out//roll_9.png", test_cases["roll_15"][0])

    test_cases["roll_30"] = (
        rotate_image(image, 30),
        "Rotate 30deg - Simulate moderate camera roll angle change"
    )
    #
    # test_cases["roll_45"] = (
    #     rotate_image(image, 45),
    #     "Rotate 45deg - Simulate significant camera roll angle change"
    # )

    # 2. 透视变换系列 - 应对 "tilted ground planes" 和 "perspective distortions"
    test_cases["ground_pitch"] = (
        perspective_transform(image, 15, 0, 0.3),
        "X-axis tilt 30deg - Simulate ground pitch tilt"
    )
    # cv2.imwrite("E://gpy//centernet_relative_kitti-paixu//img_out//pitch_1.png", test_cases["ground_pitch"][0])

    test_cases["ground_yaw"] = (
        perspective_transform(image, 0, 15, 0.3),
        "Y-axis tilt 30deg - Simulate ground yaw tilt"
    )
    # cv2.imwrite("E://gpy//centernet_relative_kitti-paixu//img_out//yaw_9.png", test_cases["ground_yaw"][0])

    # test_cases["strong_distortion"] = (
    #     perspective_transform(image, 25, 0, 0.5),
    #     "Strong perspective distortion - Simulate strong distortions"
    # )
    #
    # test_cases["complex_tilt"] = (
    #     perspective_transform(image, 25, 25, 0.3),
    #     "Dual-axis tilt 25deg - Simulate complex ground plane deviation"
    # )

    # 3. 复合变换 - 综合测试
    test_cases["combined"] = (
        affine_transform(image, 20, 0.9, 1.1, 0.1, 0),
        "Combined transform - Rotate 20deg + scale + shear"
    )

    return test_cases


def visualize_before_after(original_image, transformed_image, title="Transformation Comparison"):
    """可视化变换前后对比"""
    if len(original_image.shape) == 3:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        transformed_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original_image
        transformed_rgb = transformed_image

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(original_rgb)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')

    ax2.imshow(transformed_rgb)
    ax2.set_title("Transformed Image", fontsize=14)
    ax2.axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_all_transformations(image, test_cases):
    """一次性可视化所有8个变换"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # 显示原始图像
    if len(image.shape) == 3:
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = image

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    # 显示8个变换
    for i, (name, (transformed_img, description)) in enumerate(test_cases.items()):
        if len(transformed_img.shape) == 3:
            transformed_rgb = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
        else:
            transformed_rgb = transformed_img

        axes[i + 1].imshow(transformed_rgb)
        axes[i + 1].set_title(f"{name}", fontsize=10)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def test_geometric_robustness(image_path, show_individual=True):
    """
    完整的几何鲁棒性测试流程
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    print(f"Image loaded successfully, shape: {image.shape}")

    # 生成8个关键变换
    test_cases = generate_robustness_test_images(image)

    print(f"\nGenerated {len(test_cases)} geometric transformation test cases:")

    if show_individual:
        # 单独可视化每个变换
        for name, (transformed_img, description) in test_cases.items():
            print(f"\nShowing: {name} - {description}")
            visualize_before_after(image, transformed_img, f"{name}: {description}")

    # 也可以显示总览图
    print("\nShowing overview of all transformations...")
    visualize_all_transformations(image, test_cases)

    # 返回测试用例供进一步处理
    return test_cases


def show_each_transformation(image_path):
    """
    逐个显示每个变换的前后对比
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    # 生成8个关键变换
    test_cases = generate_robustness_test_images(image)

    print("=== Geometric Robustness Test - 8 Key Transformations ===\n")

    # 逐个显示每个变换
    for i, (name, (transformed_img, description)) in enumerate(test_cases.items(), 1):
        print(f"{i}. {name}")
        print(f"   Description: {description}")

        if 'roll' in name:
            purpose = "Address reviewer concerns about camera angles"
        elif name == 'combined':
            purpose = "Address complex geometric scenarios"
        else:
            purpose = "Address reviewer concerns about tilted ground planes & perspective distortions"

        print(f"   Purpose: {purpose}")

        # 可视化这个变换
        visualize_before_after(image, transformed_img, f"{i}. {name}: {description}")
        print("   ✓ Displayed\n")

    print("All 8 transformations have been displayed!")
    return test_cases

show_each_transformation("E://gpy//centernet_relative_kitti-paixu//img_out//9.png")

#E:\gpy\centernet_relative_kitti-paixu\img_out\roll_2.png
#E:\gpy\centernet_relative_kitti-paixu\img_out\pitch_9.png
#E:\gpy\centernet_relative_kitti-paixu\img_out\yaw_9.png