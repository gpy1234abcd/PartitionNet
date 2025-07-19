import os
import xml.etree.ElementTree as ET
import numpy as np

# np.set_printoptions(suppress=True, threshold=np.nan)
import matplotlib
from PIL import Image


def parse_obj(xml_path, filename):

    tree = ET.parse(xml_path + filename)
    objects = []
    count=0
    for obj in tree.findall('object'):
        count+=1
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        objects.append(obj_struct)

    return objects,count


def read_image(image_path, filename):
    im = Image.open(image_path + filename)
    W = im.size[0]
    H = im.size[1]
    area = W * H
    im_info = [W, H, area]
    return im_info


if __name__ == '__main__':
    xml_path = './VOCdevkit/VOC2012/Annotations/'
    # filenamess里面文件的名字
    filenamess = os.listdir(xml_path)
    filenames = []

    for name in filenamess:
        name = name.replace('.xml', '')
        filenames.append(name)
    # filenames是所有xml文件名字去掉后缀组成的列表
    recs = {}
    obs_shape = {}
    classnames = []
    num_objs = {}
    obj_avg = {}
    #  filenames=['0', '1', '10', '100', '1000', '1001', '1002', '1003', '1004']包含了所有的xml文件的名字
    #用于训练目标与目标之间的空间方位关系，删除了只有一个目标的图像
    for i, name in enumerate(filenames):
        #这个函数输入XML文件的路径，然后输出xml文件里面的目标类别
        recs[name],count = parse_obj(xml_path, name + '.xml')

        #当只有一个目标的图片删除了
        if count<=1:
            print("deleate",name)
            os.remove('./VOCdevkit/VOC2012/Annotations/%s.xml'%name)
            os.remove('./VOCdevkit/VOC2012/JPEGImages/%s.jpg' % name)


