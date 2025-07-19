#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
from centernet import CenterNet
from PIL import Image, ImageFilter,ImageEnhance


def adjust_brightness(image, change_percent):
    """
    调整图像亮度

    Args:
        image: PIL Image对象
        change_percent: 亮度变化百分比 (-40, -20, 20, 40)

    Returns:
        调整后的PIL Image对象
    """
    brightness_factor = 1.0 + (change_percent / 100.0)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)
# 方法1：指定特定强度的高斯噪声
def add_gaussian_noise(image, sigma=10):
    """
    对图像添加高斯噪声

    Args:
        image: PIL Image对象
        sigma: 噪声标准差 (5, 10, 15, 20)

    Returns:
        添加噪声后的PIL Image对象
    """
    # 转换PIL图像为numpy数组
    img_array = np.array(image, dtype=np.float32)

    # 生成高斯噪声
    noise = np.random.normal(0, sigma, img_array.shape)

    # 添加噪声
    noisy_img = img_array + noise

    # 限制像素值在0-255范围内
    noisy_img = np.clip(noisy_img, 0, 255)

    # 转换回PIL图像
    return Image.fromarray(noisy_img.astype(np.uint8))


# 方法1：指定特定核大小的高斯模糊
def apply_gaussian_blur(image, kernel_size=3):
    """
    对图像应用高斯模糊

    Args:
        image: PIL Image对象
        kernel_size: 核大小 (3, 5, 7)

    Returns:
        模糊后的PIL Image对象
    """
    # 将核大小转换为半径 (PIL的GaussianBlur使用半径参数)
    radius_map = {3: 1, 5: 2, 7: 3}
    radius = radius_map.get(kernel_size, 1)

    return image.filter(ImageFilter.GaussianBlur(radius=radius))

if __name__ == "__main__":
    centernet = CenterNet()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # mode ="dir_predict"
    # mode ='video'
    mode = "evaluate"

    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = './img/2_out.mp4'
    video_save_path = "./img_out/2.mp4"
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/testing/image_2"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    if mode=="evaluate":
        gt = '2012_testt.txt'
        with open(gt, 'r') as f:
            lines = f.readlines()
        # 提取每行中的图像路径
        image_paths = []
        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # 提取该图像的所有目标标注
            line_annotations = parts[1:]
            # 只处理包含两个或更多目标的图像
            if len(line_annotations) >= 2:
                # 提取图像路径
                image_path = parts[0]
                image_paths.append(image_path)
                annotations.append(line_annotations)
        #每一张图像都进去网络输出结果
        #每一张照片输出的都是一个列表进行存储
        all_txt_pre=[]
        all_txt_GT=[]
        all_txt_path=[]
        for i in range(len(image_paths)):
            img=image_paths[i]
            image = Image.open(img)

            #改变亮度
            # image = adjust_brightness(image, -20)

            # 应用5x5高斯模糊
            # image = apply_gaussian_blur(image, kernel_size=5)

            # 只添加噪声
            # image = add_gaussian_noise(image, sigma=5)


            pred_results= centernet.detect(image, crop=crop, count=count)
            # 检查预测目标数量是否大于等于2
            if len(pred_results) < 2:
                # print(f"预测目标数量不足: {len(pred_results)}，跳过此图像")
                continue
            all_txt_pre.append(pred_results)
            # 获取真值标注
            gt_annotations = annotations[i]
            # 解析真值标注
            gt_objects = []
            for annotation in gt_annotations:
                values = annotation.split(',')
                if len(values) != 6:
                    continue
                x1, y1, x2, y2 = map(float, values[:4])
                class_id = int(values[4])
                depth_relation = int(values[5])
                gt_obj = {
                    'box': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'depth': depth_relation,
                }
                gt_objects.append(gt_obj)
            all_txt_GT.append(gt_objects)
            all_txt_path.append(img)

        # 用于存储最终匹配后的结果
        final_matched_predictions = []
        final_matched_ground_truths = []
        final_matched_image_paths = []

        # 遍历所有图像，进行匹配
        for i in range(len(all_txt_pre)):
            # 获取当前图像的预测结果和真值标注
            pred_results = all_txt_pre[i]
            gt_objects = all_txt_GT[i]
            img = all_txt_path[i]
            # 计算IoU矩阵
            iou_matrix = np.zeros((len(gt_objects), len(pred_results)))
            for m, gt_obj in enumerate(gt_objects):
                for n, pred_obj in enumerate(pred_results):
                    # 计算IoU
                    gt_box = gt_obj['box']
                    pred_box = pred_obj['box']
                    # 交集区域
                    x1 = max(gt_box[0], pred_box[0])
                    y1 = max(gt_box[1], pred_box[1])
                    x2 = min(gt_box[2], pred_box[2])
                    y2 = min(gt_box[3], pred_box[3])
                    if x2 < x1 or y2 < y1:
                        iou = 0
                    else:
                        # 交集面积
                        intersection = (x2 - x1) * (y2 - y1)
                        # 并集面积
                        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        union = gt_area + pred_area - intersection
                        iou = intersection / union if union > 0 else 0
                    iou_matrix[m, n] = iou

            # 贪心匹配 - 按IoU从大到小排序
            flat_indices = np.argsort(iou_matrix.flatten())[::-1]
            gt_matched = [False] * len(gt_objects)
            pred_matched =[False] * len(pred_results)
            matched_pairs =[]
            for idx in flat_indices:
                gt_idx = idx // len(pred_results)
                pred_idx = idx % len(pred_results)
                # 如果IoU低于阈值则跳过
                if iou_matrix[gt_idx, pred_idx] < 0.5:  # IoU阈值可调整
                    continue
                # 如果两者都未匹配，则建立匹配
                if not gt_matched[gt_idx] and not pred_matched[pred_idx]:
                    matched_pairs.append((gt_idx, pred_idx))
                    gt_matched[gt_idx] = True
                    pred_matched[pred_idx] = True
            # 如果匹配数量不足，跳过此图像
            if len(matched_pairs) < 2:
                # print("匹配对数不足以评估关系，跳过此图像")
                continue
            # 创建有序的真值对象列表，与预测结果对应
            matched_gt_objects = []
            matched_pred_objects = []
            # 重要修改：按真值顺序重新排列匹配结果
            # 首先按照真值的索引(gt_idx)排序匹配对
            matched_pairs.sort(key=lambda x: x[0])
            # 记录每个预测对象的新顺序
            ordered_pred_indices = [pred_idx for _, pred_idx in matched_pairs]
            # 按真值顺序添加到最终列表
            for gt_idx, pred_idx in matched_pairs:
                matched_gt_objects.append(gt_objects[gt_idx])
                matched_pred_objects.append(pred_results[pred_idx])
            # 将匹配结果添加到最终列表
            final_matched_predictions.append(matched_pred_objects)
            final_matched_ground_truths.append(matched_gt_objects)
            final_matched_image_paths.append(img)




        DEPTH_CATEGORIES = {
            0: "F&B",  # 前背：主体在客体前面
            1: "B&F",  # 背前：主体在客体后面
            2: "S&D"  # 同深度：主体和客体深度差值在阈值内
        }

        print("开始计算深度关系混淆矩阵...")

        # 存储所有目标对的真值关系和预测关系
        true_relations = []
        pred_relations = []

        # 遍历每张图像
        for i in range(len(final_matched_predictions)):
            pred_objects = final_matched_predictions[i]
            gt_objects = final_matched_ground_truths[i]







            # 目标数量检查
            if len(pred_objects) < 2 or len(gt_objects) < 2:
                continue

            # 计算图像中所有可能的目标对关系
            for j in range(len(gt_objects)):
                for k in range(j + 1, len(gt_objects)):
                    # 获取真值目标信息
                    gt_obj1 = gt_objects[j]
                    gt_obj2 = gt_objects[k]


                    # 获取预测目标信息
                    pred_obj1 = pred_objects[j]
                    pred_obj2 = pred_objects[k]

                    # 获取真值深度关系
                    if 'depth' in gt_obj1 and 'depth' in gt_obj2:
                        # 直接使用真值提供的深度关系
                        gt_depth1 = gt_obj1['depth']
                        gt_depth2 = gt_obj2['depth']

                        # 确定真实的深度关系
                        # 逻辑：如果obj1的depth=0且obj2的depth=1，表示obj1在前面(F&B, 0)
                        #      如果obj1的depth=1且obj2的depth=0，表示obj1在后面(B&F, 1)
                        #      如果两者深度标记相同，表示同深度(S&D, 2)

                        gt_depth_diff = gt_depth1-gt_depth2
                        depth_threshold = 1.5  # 可以根据实际情况调整阈值

                        # 根据深度差值确定真实关系
                        if abs(gt_depth_diff) <= depth_threshold:
                            true_relation = 2  # S&D: 同深度
                        elif gt_depth_diff < -depth_threshold:
                            true_relation = 1  # B&F: obj1在obj2后面
                        else:  # gt_depth_diff > depth_threshold
                            true_relation = 0  # F&B: obj1在obj2前面

                        true_relations.append(true_relation)


                        # 获取预测目标的深度值（假设预测结果中包含depth_value字段）
                        # 注意：如果你的预测结果中没有depth_value，需要根据实际情况修改
                        if 'depth' in pred_obj1 and 'depth' in pred_obj2:
                            depth1 = pred_obj1['depth']
                            depth2 = pred_obj2['depth']

                            # 根据深度值计算深度关系
                            depth_thresholdd = 1.5
                            depth_diff = depth1 - depth2

                            if abs(depth_diff) <= depth_thresholdd:
                                pred_relation = 2  # S&D: 同深度
                            elif depth_diff < -depth_thresholdd:
                                pred_relation = 1  # B&F: obj1在obj2后面
                            else:
                                pred_relation = 0  # F&B: obj1在obj2前面

                            pred_relations.append(pred_relation)
                        else:
                            # 如果预测结果中没有深度值，你可能需要添加其他方式来获取或计算深度关系
                            print(f"警告：预测目标对 ({j},{k}) 在图像 {i} 中缺少深度信息")



        print(true_relations)
        print(pred_relations)

        # 计算混淆矩阵
        confusion_matrix = np.zeros((3, 3), dtype=int)

        for i in range(len(true_relations)):
            true_rel = true_relations[i]
            pred_rel = pred_relations[i]
            confusion_matrix[true_rel, pred_rel] += 1

        # 打印混淆矩阵
        print("\n深度关系混淆矩阵:")
        print(f"{'':10} | {'预测 F&B':10} | {'预测 B&F':10} | {'预测 S&D':10}")
        print("-" * 50)
        print(
            f"{'真值 F&B':10} | {confusion_matrix[0, 0]:10} | {confusion_matrix[0, 1]:10} | {confusion_matrix[0, 2]:10}")
        print(
            f"{'真值 B&F':10} | {confusion_matrix[1, 0]:10} | {confusion_matrix[1, 1]:10} | {confusion_matrix[1, 2]:10}")
        print(
            f"{'真值 S&D':10} | {confusion_matrix[2, 0]:10} | {confusion_matrix[2, 1]:10} | {confusion_matrix[2, 2]:10}")

        # 计算评估指标
        accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0
        print(f"\n总体准确率: {accuracy:.4f}")

        # 计算每个类别的精确率和召回率
        class_metrics = []
        for i in range(3):
            precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]) if np.sum(
                confusion_matrix[:, i]) > 0 else 0
            recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) if np.sum(
                confusion_matrix[i, :]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{DEPTH_CATEGORIES[i]} - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
            class_metrics.append((precision, recall, f1))

        # 绘制混淆矩阵热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('深度关系混淆矩阵')
        plt.colorbar()

        # 设置坐标轴
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, [DEPTH_CATEGORIES[i] for i in range(3)])
        plt.yticks(tick_marks, [DEPTH_CATEGORIES[i] for i in range(3)])

        # 添加数值标签
        thresh = confusion_matrix.max() / 2.
        for i in range(3):
            for j in range(3):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.ylabel('真实类别')
        plt.xlabel('预测类别')
        plt.tight_layout()
        # plt.savefig('depth_relation_confusion_matrix.png', dpi=300)
        plt.show()

        #
        #
        #







    elif mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入centernet.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入centernet.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入centernet.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:


                r_image = centernet.detect_image(image, crop = crop, count=count)
                r_image.show()
                r_image.save('out.png')


    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(centernet.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = centernet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = centernet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                centernet.detect_heatmap(image, heatmap_save_path)
        
    elif mode == "export_onnx":
        centernet.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
