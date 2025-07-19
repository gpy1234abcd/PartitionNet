# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import h5py
# import os
# from PIL import Image
# import cv2
#
# f = h5py.File("D:\\BaiduNetdiskDownload\\nyu_depth_v2_labeled.mat",mode='r')
#
# # extract images
# images = f["images"]
# images = np.array(images)
#
# path_converted = 'D:\\NYU\\nyu_images'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)
#
# images_number = []
# for i in range(len(images)):
#     images_number.append(images[i])
#     a = np.array(images_number[i])
#     r = Image.fromarray(a[0]).convert('L')
#     g = Image.fromarray(a[1]).convert('L')
#     b = Image.fromarray(a[2]).convert('L')
#     img = Image.merge("RGB", (r, g, b))
#     img = img.transpose(Image.ROTATE_270)
#     iconpath = 'D:\\NYU\\nyu_images/' + str(i) + '.png'
#     img.save(iconpath, optimize=True)
#     # exit(0)
#
# print("image extract finished!!!!!")
#
# # extract depths
# depths = f["depths"]
# depths = np.array(depths)
#
# path_converted = 'D:\\NYU\\nyu_depths/'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)
#
# max = depths.max()
# depths = depths / max * 65535
# depths = depths.transpose((0, 2, 1))
#
# for i in range(len(depths)):
#     print(str(i) + '.png')
#     depths_img = np.uint16(depths[i])
#     depths_img_new = cv2.flip(depths_img, 1)
#     print(depths_img_new.max(), depths_img_new.min())
#     iconpath = path_converted + str(i) + '.png'
#     cv2.imwrite(iconpath, depths_img_new)
#     # exit(0)
# print("depths extract finished!!!!!")
#
#
# labels = f["labels"]
# labels = np.array(labels)
#
# path_converted = 'D:\\NYU\\nyu_labels/'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)
#
# labels_max = labels.max()
# labels = labels / labels_max * 65535
# labels = labels.transpose((0, 2, 1))
#
# for i in range(len(labels)):
#
#     label_img = np.uint16(labels[i])
#     label_img_new = cv2.flip(label_img, 1)
#
#     iconpath = '.\\NYU\\nyu_labels/' + str(i) + '.png'
#     cv2.imwrite(iconpath, label_img_new)
#     # exit(0)
#
# print("labels extract finished!!!!!")
#
#
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

classes_path        = 'model_data/voc_classes.txt'
def convert_annotation(year, image_id, list_file):
    # 打开对应的xml标签文件
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')

    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)

        # 找到标签中目标的深度距离
        dep = obj.find('depth_object').text

        # if dep not in classes or int(difficult) == 1:
        #     continue
        # cls_dep=classes.index(dep)

        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id) + ',' + str(dep))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
VOCdevkit_sets  = [('2012', 'testt')]
VOCdevkit_path  = 'VOCdevkit'
for year, image_set in VOCdevkit_sets:
    image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                     encoding='utf-8').read().strip().split()

    # 打开 train_2012.txt文件进行写入
    list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
    #
    # print(image_ids) ['000000', '000001', '000002',。。。。。。。。]

    for image_id in image_ids:
        # 把图像的路径写入tain_2012.txt中
        list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))

        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    photo_nums[type_index] = len(image_ids)
    type_index += 1
    list_file.close()
print("Generate 2012_train.txt and 2007_val.txt for train done.")