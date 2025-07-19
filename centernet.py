import colorsys
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50,CenterNet_Resnet101
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_bbox, postprocess
import matplotlib
matplotlib.use('tkaGg')  # 大小写无所谓 tkaGg ,TkAgg 都行
import matplotlib.pyplot as plt
import ast
import prettytable
import math
#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#


class DashedImageDraw(ImageDraw.ImageDraw):

    def thick_line(self, xy, direction, fill=None, width=0):
        # xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        # direction – Sequence of 2-tuples like [(x, y), (x, y), ...]
        if xy[0] != xy[1]:
            self.line(xy, fill=fill, width=width)
        else:
            x1, y1 = xy[0]
            dx1, dy1 = direction[0]
            dx2, dy2 = direction[1]
            if dy2 - dy1 < 0:
                x1 -= 1
            if dx2 - dx1 < 0:
                y1 -= 1
            if dy2 - dy1 != 0:
                if dx2 - dx1 != 0:
                    k = - (dx2 - dx1) / (dy2 - dy1)
                    a = 1 / math.sqrt(1 + k ** 2)
                    b = (width * a - 1) / 2
                else:
                    k = 0
                    b = (width - 1) / 2
                x3 = x1 - math.floor(b)
                y3 = y1 - int(k * b)
                x4 = x1 + math.ceil(b)
                y4 = y1 + int(k * b)
            else:
                x3 = x1
                y3 = y1 - math.floor((width - 1) / 2)
                x4 = x1
                y4 = y1 + math.ceil((width - 1) / 2)
            self.line([(x3, y3), (x4, y4)], fill=fill, width=1)
        return

    def dashed_line(self, xy, dash=(2, 2), fill=None, width=0):
        # xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]
            x_length = x2 - x1
            y_length = y2 - y1
            length = math.sqrt(x_length ** 2 + y_length ** 2)
            dash_enabled = True
            postion = 0
            while postion <= length:
                for dash_step in dash:
                    if postion > length:
                        break
                    if dash_enabled:
                        start = postion / length
                        end = min((postion + dash_step - 1) / length, 1)
                        self.thick_line([(round(x1 + start * x_length),
                                          round(y1 + start * y_length)),
                                         (round(x1 + end * x_length),
                                          round(y1 + end * y_length))],
                                        xy, fill, width)
                    dash_enabled = not dash_enabled
                    postion += dash_step
        return

    def dashed_rectangle(self, xy, dash=(2, 2), outline=None, width=0):
        # xy - Sequence of [(x1, y1), (x2, y2)] where (x1, y1) is top left corner and (x2, y2) is bottom right corner
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        halfwidth1 = math.floor((width - 1) / 2)
        halfwidth2 = math.ceil((width - 1) / 2)
        min_dash_gap = min(dash[1::2])
        end_change1 = halfwidth1 + min_dash_gap + 1
        end_change2 = halfwidth2 + min_dash_gap + 1
        odd_width_change = (width - 1) % 2
        self.dashed_line([(x1 - halfwidth1, y1), (x2 - end_change1, y1)],
                         dash, outline, width)
        self.dashed_line([(x2, y1 - halfwidth1), (x2, y2 - end_change1)],
                         dash, outline, width)
        self.dashed_line([(x2 + halfwidth2, y2 + odd_width_change),
                          (x1 + end_change2, y2 + odd_width_change)],
                         dash, outline, width)
        self.dashed_line([(x1 + odd_width_change, y2 + halfwidth2),
                          (x1 + odd_width_change, y1 + end_change2)],
                         dash, outline, width)
        return










class CenterNet(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : 'model_data/centernet_resnet50_voc.pth',

        # "model_path": 'model_data/ep160-loss4.100-val_loss0.000.pth',
        "model_path": 'model_data/ep400-loss3.575-val_loss0.000.pth',
        # "model_path": 'model_data/ep240-loss3.908-val_loss0.000.pth',



        "classes_path"      : 'model_data/voc_classes.txt',
        #--------------------------------------------------------------------------#
        #   用于选择所使用的模型的主干
        #   resnet50, hourglass
        #--------------------------------------------------------------------------#
        "backbone"          : 'resnet101',

        #--------------------------------------------------------------------------#
        #   输入图片的大小，设置成32的倍数
        #--------------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        #--------------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #--------------------------------------------------------------------------#
        "confidence"        : 0.3,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #--------------------------------------------------------------------------#
        #   是否进行非极大抑制，可以根据检测效果自行选择
        #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
        #--------------------------------------------------------------------------#
        "nms"               : True,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化centernet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        assert self.backbone in ['resnet50', 'hourglass','resnet101']
        if self.backbone == "resnet50":
            self.net = CenterNet_Resnet50(num_classes=self.num_classes, pretrained=False)
        if self.backbone == "resnet101":
            self.net = CenterNet_Resnet101(num_classes=self.num_classes, pretrained=False)
        else:
            self.net = CenterNet_HourglassNet({'hm': self.num_classes, 'wh': 2, 'reg':2})

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def classfi(self, image, A_):
        # 先把真实标签的信息转化成[[x1,y1,class,depth],[x2,y2,class,depth],[].....]
        Annotation=A_
        # print(Annotation[1:],Annotation_depth)
        # print(type(Annotation_depth))
        # 装原始的标签信息的
        list_boj=[]
        # 装三元组的标签信息的
        list_Triplet=[]
        # 定义一个放预测的目标的列表
        list_obj_pre = []
        # 定义一个放预测的三元组的列表
        list_Triplet_pre=[]
        for ii in range(len(Annotation[1:])):
            x=(int(Annotation[1:][ii].split(",")[0])+int(Annotation[1:][ii].split(",")[2]))/2
            y=int(Annotation[1:][ii].split(",")[3])
            cla=int(Annotation[1:][ii].split(",")[4])
            depth=int(Annotation[1:][ii].split(",")[5])
            list_boj.append([x,y,cla,depth])
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            # ---------------------------------------------------------#
            #   设置字体与边框厚度
            # ---------------------------------------------------------#
            font = ImageFont.truetype(font='model_data/simhei.ttf',
                                      size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
            thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
            # --------------------------------------#
            #   如果没有检测到物体，则返回原图，画上GT标签框
            # --------------------------------------#
            if results[0] is None:
               return [],[]
            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf =  results[0][:, 4]
            top_boxes = results[0][:,:4]
            top_depth = results[0][:, -1]


        #---------------在这里输出预测的图像上目标之间的前后关系三元组--------------------#
        # 先把预测的所有目标按照 [[x1,y1,class,depth],[x2,y2,class,depth],[].....]的形式存放

        top_label_list=top_label.tolist()
        top_conf_list=top_conf.tolist()
        top_boxes_list=top_boxes.tolist()
        top_depth_list=top_depth.tolist()
        for l in range(len(top_label_list)):
            list_obj_pre.append([int((top_boxes_list[l][1]+top_boxes_list[l][3])/2),int(top_boxes_list[l][2]),top_label_list[l],int(top_depth_list[l]),round(top_conf_list[l],2)])


        # 第一步、先删除预测的里面，重复的预测，
        list_obj_pre_copy=list_obj_pre
        indexes=[]
        for i_ in range(len(list_obj_pre) - 1):
            for j_ in range(i_ + 1, len(list_obj_pre)):
                # 拿到A的深度和B的深度，做一个差，通过差判断是哪个前后关系的类别
                if abs(list_obj_pre[i_][0]-list_obj_pre[j_][0])<=7 and abs(list_obj_pre[i_][1]-list_obj_pre[j_][1])<=7\
                    and list_obj_pre[i_][2]==list_obj_pre[j_][2]:
                    indexes.append(j_)
                if abs(list_obj_pre[i_][1]-list_obj_pre[j_][1])<=3:
                    if abs(list_obj_pre[i_][0]-list_obj_pre[j_][0])<=12:
                        indexes.append(j_)

        # 删除重复的元素
        indexes = list(set(indexes))
        for index in sorted(indexes, reverse=True):
            del list_obj_pre_copy[index]


        # print(list_boj)
        # print(list_obj_pre_copy)


        # 第二步数，还有删除真实框里没有的，遍历一遍，匹配上就返回下标，匹配不上就删除掉。,匹配上了就保存匹配上的索引
        list_boj_new=[]




        #预测的要删除的索引
        indexx=[]
        #匹配上real的索引
        inde=[]
        for nn in range(len(list_obj_pre_copy)):
            p_obj=list_obj_pre_copy[nn]
            for mm in range(len(list_boj)):
                gt_obj=list_boj[mm]
                if p_obj[2]==gt_obj[2] and ( abs(p_obj[0]-gt_obj[0])<=15 and abs(p_obj[1]-gt_obj[1])<=15):
                    list_boj_new.append(list_boj[mm])
                    inde.append(mm)
                    # del list_boj[mm]
                    break
                if p_obj[2]==gt_obj[2] and  (abs(p_obj[0]-gt_obj[0])<=5 and abs(p_obj[1]-gt_obj[1]<=20) ):
                    inde.append(mm)
                    break
                #or abs(p_obj[1] - gt_obj[1] < 5 and abs(p_obj[0] - gt_obj[0]) <= 25))
                if p_obj[2]==gt_obj[2] and (abs(p_obj[1] - gt_obj[1] <= 5 and abs(p_obj[0] - gt_obj[0]) <= 20)):
                    inde.append(mm)
                    break
                else:
                    if mm==len(list_boj)-1:
                        # print("判断是不是都匹配不上，就删除这个预测的这个")
                        indexx.append(nn)

        indexx = list(set(indexx))
        for index in sorted(indexx, reverse=True):
            del list_obj_pre_copy[index]




        print("real",list_boj)
        print("pre",list_obj_pre_copy)
        print("匹配上的索引",inde)
        # print("----------------------")


        p = []
        g = []

        #现在为止都是预测的都在真的里面了，然后就开始判断，
        # 1 预测的是空，就直接返回[][]
        # 2 预测的len是1 ，就看这个一个与GT的前后关系
        # 3 预测的len大于1 ，就开始配对与GT对比，看看前后关系。
        if len(list_obj_pre_copy)==0 or len(list_boj)==0:
            # 说明是没预测出来，直接返回空列表即可
            return [],[]
        if len(list_obj_pre_copy)==1:
            # 当只预测出一个的时候 ，直接就判断跟GT的前后关系，
            # pre_depth=list_obj_pre_copy[0][3]
            # gt_depth=list_boj[inde[0]][3]
            # if (gt_sub_depth - gt_obj_depth) >= 8:
            #     g.append(2)
            # elif abs(gt_sub_depth - gt_obj_depth) < 8:
            #     g.append(0)
            # else:
            #     g.append(1)
            #
            # if (pre_sub_depth - pre_obj_depth) >= 8:
            #     g.append(2)
            # elif abs(pre_sub_depth - pre_obj_depth) < 8:
            #     g.append(0)
            # else:
            #     g.append(1)
            return [],[]


        if len(list_obj_pre_copy)>1:
            #这是有多个目标的场景，配对看。
            for i_ in range(len(list_obj_pre_copy) - 1):
                for j_ in range(i_+1, len(list_obj_pre_copy)):

                    if i_<=len(list_obj_pre_copy)/2:
                    #
                        pre_sub_depth   =list_obj_pre_copy[i_][3]
                        pre_obj_depth   =list_obj_pre_copy[j_][3]
                        gt_sub_depth    =list_boj[inde[i_]][3]
                        gt_obj_depth    =list_boj[inde[j_]][3]
                    else:
                        pre_sub_depth = list_obj_pre_copy[j_][3]
                        pre_obj_depth = list_obj_pre_copy[i_][3]
                        gt_sub_depth = list_boj[inde[j_]][3]
                        gt_obj_depth = list_boj[inde[i_]][3]

                    # 预测的前后关系的判断
                    # if (pre_sub_depth-pre_obj_depth)>=8:
                    #     p.append(2)
                    #     if (gt_sub_depth - gt_obj_depth) >= 8:
                    #         g.append(2)
                    #     elif abs(gt_sub_depth - gt_obj_depth) < 8:
                    #         g.append(0)
                    #     else:
                    #         g.append(1)
                    # elif abs(pre_sub_depth-pre_obj_depth)<8:
                    #     p.append(0)
                    #     if (gt_sub_depth - gt_obj_depth) >= 8:
                    #         g.append(2)
                    #     elif abs(gt_sub_depth - gt_obj_depth) < 8:
                    #         g.append(0)
                    #     else:
                    #         g.append(1)
                    # else:
                    #     p.append(1)
                    #     if abs(gt_sub_depth-gt_obj_depth)<=50:
                    #         g.append(1)
                    #
                    #     elif (gt_sub_depth-gt_obj_depth)>50:
                    #         g.append(2)
                    #     else:
                    #         g.append(0)


                    if (gt_sub_depth-gt_obj_depth)>=5:
                        g.append(2)
                    elif abs(gt_sub_depth-gt_obj_depth)<5:
                        g.append(0)
                    else:
                        g.append(1)

                    if (pre_sub_depth-pre_obj_depth)>=5:
                        p.append(2)
                    elif abs(pre_sub_depth-pre_obj_depth)<5:
                        p.append(0)
                    else:
                        p.append(1)


                    # if pre_sub_depth-pre_obj_depth > 0:
                    #     if pre_sub_depth-pre_obj_depth>5:
                    #         p.append(2)
                    #     else:
                    #         p.append(0)
                    # else:
                    #     if pre_obj_depth-pre_sub_depth > 3:
                    #         p.append(1)
                    #     else:
                    #         p.append(0)
                    # 真实的前后关系的判断
                    # if gt_sub_depth-gt_obj_depth > 0:
                    #     if gt_sub_depth-gt_obj_depth > 8:
                    #         g.append(2)
                    #     else:
                    #         g.append(0)
                    # else:
                    #     if gt_obj_depth-gt_sub_depth > 3:
                    #         g.append(1)
                    #     else:
                    #         g.append(0)
            print(p,g)
            return p,g


    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            # hm,wh,offset,hdepth



            # print(outputs[3].shape)
            # for i in range(3):
            #     s=outputs[1].cpu()[0,i,:,:]
            #     plt.imshow(s)
            #     plt.colorbar()
            #     plt.show()
            # for i in range(20):
            #     s=outputs[0].cpu()[0,i,:,:]
            #     plt.imshow(s)
            #     plt.colorbar()
            #     plt.show()


            # for i in range(3):
            #     s=outputs[1].cpu()[0,i,:,:]
            #     plt.imshow(s)
            #     plt.colorbar()
            #     plt.show()


            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            #-----------------------------------------------------------#
            #   利用预测结果进行解码
            #-----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            #-------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            #-------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if results[0] is None:
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
            top_depth   = results[0][:, -1]

        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        draw = ImageDraw.Draw(image)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            dep             = top_depth[i]
            top, left, bottom, right = box
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))
            label = '{} {:.2f}'.format(predicted_class,score)
            # draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font,dep)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right, dep)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                # draw.dashed_rectangle([(left,top),(right,bottom)],
                #                    dash=(5, 3), outline=self.colors[c],width=2)

            # dash_style = [3] * 4 + [1] * 4
            # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 0))

            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            # del draw

            # 在这里写上画目标框之间箭头的程序，subject用箭头指向object的框。线的中心点处写上前后的类别，然后置信度。
            # 1、先读取主体目标的目标框底部位置，在读取object的底部位置，然后画一个箭头线

            # 2、在两个箭头中间plot上目标之间的空间关系类别和置信度。设置两个参数 T G 作为阈值
            #

        T, G = 4, 25
        for i in range(len(top_label) - 1):
            for j in range(i + 1, len(top_label)):
                # 主体目标的下方中心点
                sub_point = [top_boxes[i][3], top_boxes[i][2]]
                obj_point = [top_boxes[j][1], top_boxes[j][2]]

                # 绘制从点A到点B的线段
                point_A = sub_point
                point_B = obj_point
                # draw.line([tuple(point_A), tuple(point_B)], fill="red", width=2)
                draw.line([tuple(point_A), tuple(point_B)], fill="green", width=thickness)

                # 计算箭头的两个点
                dx = point_B[0] - point_A[0]
                dy = point_B[1] - point_A[1]
                length = (dx ** 2 + dy ** 2) ** 0.5

                if length != 0:
                    unit_dx = dx / length
                    unit_dy = dy / length

                    # 箭头的长度和宽度
                    arrow_length = 10
                    arrow_width = 5

                    # 计算箭头的两个点
                    arrow_point1 = (
                        point_B[0] - arrow_length * unit_dx + arrow_width * unit_dy,
                        point_B[1] - arrow_length * unit_dy - arrow_width * unit_dx
                    )
                    arrow_point2 = (
                        point_B[0] - arrow_length * unit_dx - arrow_width * unit_dy,
                        point_B[1] - arrow_length * unit_dy + arrow_width * unit_dx
                    )

                    # 绘制箭头
                    draw.polygon([tuple(point_B), arrow_point1, arrow_point2], fill="green")

                # 计算箭头线段的中点
                midpoint = (
                    (point_A[0] + point_B[0]) / 2 + 1,  # 中点的 x 坐标
                    (point_A[1] + point_B[1]) / 2 + 1  # 中点的 y 坐标
                )
                # 在中点处写上前后关系的类别和置信度
                # 先判断是什么前后关系的类别

                if abs(top_depth[i] - top_depth[j]) <= 2.5:
                    # print("S&S")
                    gap = abs(top_depth[i] - top_depth[j])  # 计算两个数值之间的绝对差值
                    confidence = 1 - (gap / T)  # 归一化到 0-1
                    confidence = max(0, min(confidence, 1))  # 确保置信度在 0 到 1 之间
                    confidence = round(confidence, 2)
                    confidence =float(f"{confidence:.2f}")
                    # draw.text(midpoint, "S&D" + "-" + str(confidence), fill="red", font=font)
                    relation_label = "S&D" + "-" + str(confidence)
                    relation_label_size = draw.textsize(relation_label, font)

                    # 绘制背景矩形
                    draw.rectangle([
                        (midpoint[0], midpoint[1]),
                        (midpoint[0] + relation_label_size[0], midpoint[1] + relation_label_size[1])
                    ], fill=(255, 255, 0))
                    draw.text(midpoint, "S&D" + "-" + str(confidence), fill="black", font=font)

                else:
                    if top_depth[i] > top_depth[j]:
                        # print("B&F")
                        gap = abs(top_depth[i] - top_depth[j])  # 计算两个数值之间的绝对差值
                        confidence = gap/G  # 归一化到 0-1
                        confidence = max(0, min(confidence, 1))  # 确保置信度在 0 到 1 之间
                        confidence = round(confidence, 2)
                        confidence = float(f"{confidence:.2f}")
                        # draw.text(midpoint, "B&F" + "-" + str(confidence), fill="red", font=font)
                        # 计算关系标签文本
                        relation_label = "B&F" + "-" + str(confidence)
                        relation_label_size = draw.textsize(relation_label, font)

                        # 绘制背景矩形
                        draw.rectangle([
                            (midpoint[0], midpoint[1]),
                            (midpoint[0] + relation_label_size[0], midpoint[1] + relation_label_size[1])
                        ], fill=(255, 255, 0))

                        draw.text(midpoint, "B&F" + "-" + str(confidence), fill="black", font=font)
                    else:
                        # print("F&B")
                        gap = abs(top_depth[i] - top_depth[j])  # 计算两个数值之间的绝对差值
                        confidence = gap/G  # 归一化到 0-1
                        confidence = max(0, min(confidence, 1))  # 确保置信度在 0 到 1 之间
                        confidence = round(confidence, 2)
                        confidence = float(f"{confidence:.2f}")
                        # draw.text(midpoint, "F&B" + "-" + str(confidence), fill="red", font=font)
                        # 计算关系标签文本
                        relation_label = "F&B" + "-" + str(confidence)
                        relation_label_size = draw.textsize(relation_label, font)

                        # 绘制背景矩形
                        draw.rectangle([
                            (midpoint[0], midpoint[1]),
                            (midpoint[0] + relation_label_size[0], midpoint[1] + relation_label_size[1])
                        ], fill=(255, 255, 0))

                        draw.text(midpoint, "F&B" + "-" + str(confidence), fill="black", font=font)

            # del draw


        return image


    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            #-----------------------------------------------------------#
            #   利用预测结果进行解码
            #-----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            #-------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            #-------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                if self.backbone == 'hourglass':
                    outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
                #-----------------------------------------------------------#
                #   利用预测结果进行解码
                #-----------------------------------------------------------#
                outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

                #-------------------------------------------------------#
                #   对于centernet网络来讲，确立中心非常重要。
                #   对于大目标而言，会存在许多的局部信息。
                #   此时对于同一个大目标，中心点比较难以确定。
                #   使用最大池化的非极大抑制方法无法去除局部框
                #   所以我还是写了另外一段对框进行非极大抑制的代码
                #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
                #-------------------------------------------------------#
                results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt

        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask        = np.zeros((image.size[1], image.size[0]))
        score       = np.max(outputs[0][0].permute(1, 2, 0).cpu().numpy(), -1)
        score       = cv2.resize(score, (image.size[0], image.size[1]))
        normed_score    = (score * 255).astype('uint8')
        mask            = np.maximum(mask, normed_score)
        
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            #-----------------------------------------------------------#
            #   利用预测结果进行解码
            #-----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            #-------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            #-------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if results[0] is None:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]


        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
