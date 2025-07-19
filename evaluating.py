                                # 空间前后关系得评价指标。
# 一、在自己通过pascal voc制作的pascal_relation 上测评一下准确率啥的
# 二、在kitti上与当前的3D目标检测的算法（找出最新的三四个算法） ，做一个后处理 检测的目标按照相机坐标系的z（深度），弄成前后关系。看看准确率啥的，对比一下。
#并且自己的算法可以换一下backbone ，给出resnet50 和resnet101两个版本。




import time
import cv2
import numpy as np
import ast
import random
from PIL import Image
from centernet import CenterNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import prettytable
#---------------------------------------------------------------#
# 1、首先可以先统计测试集上有M个图片，其中某张图片上有n个目标，可以统计出测试集上一共有多少种前后的关系对
#---------------------------------------------------------------#
#  读取测试集的txt文本
test_annotation_path = '2012_testt.txt'
with open(test_annotation_path) as f:
    test_lines = f.readlines()
# # test_annotation_depth_path = '2012_val_depth2.txt'
# # with open(test_annotation_depth_path) as f:
# #     test_depth_lines = f.readlines()
# print("Pascal_voc2012_depth_relation 数据集一共有%s张图像进行测试"%len(test_lines))
# #测试集中总的前后关系的个数
# depth_relation_count=0
# # 每对前后关系的个数
# one_,qianhou,yiyang=0,0,0
# for i in range(len(test_lines)):
#     # 算一算各类前后关系分别是多少种，用深度的Gt更方便
#     dep=test_depth_lines[i]
#     dep=ast.literal_eval(dep)
#     if len(dep)==1:
#         one_+=1
#     else:
#         for j_ in range(len(dep)-1):
#             for j__ in range(j_+1,len(dep)):
#                 if abs(dep[j_]-dep[j__])<=8:
#                     yiyang+=1
#                 else:
#                     qianhou+=1
#
#
#
#     object_count=len(test_lines[i].split())-1
#     #如果一张图上就一个目标 ，能检测出来就是对了，默认是一对目标和背景的关系。
#     if object_count==1:
#         depth_relation_count +=1
#     else:
#         # n(n-1)/2
#         object_relation_count=object_count*(object_count-1)/2
#         depth_relation_count+=object_relation_count
# print("Pascal_voc2012_depth_relation 数据集一共有%s对目标的前后关系进行测试"%int(depth_relation_count))
# print("-1:%s"%one_,"0:%s"%yiyang,"1和2:%s"%qianhou)
#---------------------------------------------------------------#
# 2、读取训练的模型，输入图像 输出预测的结果。
# mode=predict_img or mode=predict_dir
#---------------------------------------------------------------#
#一、评测一张图片的模式
mode = "predict_img"
# 二、测评难度很大的关系的map
# mode="evaluating"
# 三、测评R@K指标
# mode = "R@K"
# 四、P/R/f1
mode="classification"
centernet = CenterNet()

def calculate_prediction_recall(label, pre, classes=None):
    """
    计算准确率和召回率:传入预测值及对应的真实标签计算
    :param label:标签
    :param pre:对应的预测值
    :param classes:类别名（None则为数字代替）
    :return:
    """
    # if classes:
    #     classes = list(range(classes))
    classes = classes

    confMatrix = confusion_matrix(label, pre)
    print(confMatrix)
    total_prediction = 0
    total_recall = 0
    result_table = prettytable.PrettyTable()
    class_multi = 1
    result_table.field_names = ['Type', 'Prediction(精确率)', 'Recall(召回率)', 'F1_Score']
    for i in range(len(confMatrix)):
        label_total_sum_col = confMatrix.sum(axis=0)[i]
        label_total_sum_row = confMatrix.sum(axis=1)[i]
        if label_total_sum_col:  # 防止除0
            prediction = confMatrix[i][i] / label_total_sum_col
        else:
            prediction = 0
        if label_total_sum_row:
            recall = confMatrix[i][i] / label_total_sum_row
        else:
            recall = 0
        if (prediction + recall) != 0:
            F1_score = prediction * recall * 2 / (prediction + recall)
        else:
            F1_score = 0
        result_table.add_row([classes[i], np.round(prediction, 3), np.round(recall, 3),
                              np.round(F1_score, 3)])

        total_prediction += prediction
        total_recall += recall
        class_multi *= prediction
    total_prediction = total_prediction / len(confMatrix)
    total_recall = total_recall / len(confMatrix)
    total_F1_score = total_prediction * total_recall * 2 / (total_prediction + total_recall)
    geometric_mean = pow(class_multi, 1 / len(confMatrix))

    return total_prediction, total_recall, total_F1_score, result_table, geometric_mean, confMatrix






def look(p,t):
    pre,gt=p,t
    print(pre)
    print(gt)
    if len(gt)<1:
        return 0
    else:
        # 开始把sub 和obj 对应上，对应上了就看一下关系对不对，对的话就返回一个1
        sub,obj=pre[0],pre[1]
        sub_dep,obj_dep=0,0
        gt_rela=0
        for w in range(len(gt)):
            # 下面开始判断哪一个预测的目标 能和Gt标签的第一个对应上
            if sub[2] == gt[w][2] and (abs(sub[0] - gt[w][0] <= 20) and abs(sub[1] - gt[w][1] <= 15) and abs((sub[0]+sub[1])-(gt[w][0]+gt[w][1]))<=30):
                sub_dep=gt[w][3]
                break
        for w in range(len(gt)):
            if obj[2] == gt[w][2] and (abs(obj[0] - gt[w][0] <= 20) and abs(obj[1] - gt[w][1] <= 15) and abs((obj[0]+obj[1])-(gt[w][0]+gt[w][1]))<=30):
                obj_dep=gt[w][3]
                break
            #然后判断sub和obj的前后关系
        print(sub_dep,obj_dep)
        if (sub_dep-obj_dep)<=0:
            if abs(sub_dep-obj_dep)<=8:
                gt_rela=0
            else:
                gt_rela=1
        else:
            if abs(sub_dep-obj_dep)<=8:
                gt_rela=0
            else:
                gt_rela=2

        if gt_rela==predict[2]:
            return 1
        else:
            return 0


# if mode=="predict_img":
#     while True:
#         #读测试集数量的列表长度内随机一个数目，随机的这个数就是用来 拿到二维的和深度的Gt
#         # 随机在测试集上抽取一个,获得二维的一个位置框和类别
#         random_int=random.randint(0,len(test_lines))
#         for jjj in range(0,len(test_lines)):
#             random_int=jjj
#             annotation= test_lines[random_int].split()
#             annotation_depth=test_depth_lines[random_int]
#             img=annotation[0]
#             print("图像路径:",img)
#             # 获得对应的GT的annotation信息
#             # index = img.index("_")
#             # annotation=img[index-4:index]+img[index:index+7]
#             try:
#                 image = Image.open(img)
#             except:
#                 print('Open Error! Try again!')
#             else:
#                 r_image = centernet.evaluating_image(image, crop=False, count=False, A_=annotation,A_D=annotation_depth)
#
#                 r_image.show()
#                 # one_TP, one_FP, one_FN, yiyang_TP, yiyang_FP, yiyang_FN, qianhou_TP, qianhou_FP, qianhou_FN = centernet.evaluating_indicator(
#                 #     image, crop=False, count=False, A_=annotation, A_D=annotation_depth)
#                 # print(one_TP, one_FP, one_FN, yiyang_TP, yiyang_FP, yiyang_FN, qianhou_TP, qianhou_FP, qianhou_FN )
#
# if mode=="evaluating":
#     # 新建六个变量，用来计算三个前后关系类别的指标
#     one_TP, one_FP, one_FN, yiyang_TP, yiyang_FP, yiyang_FN, qianhou_TP, qianhou_FP, qianhou_FN = 0, 0, 0, 0, 0, 0, 0, 0, 0
#     for i in range(len(test_lines)):
#         annotation = test_lines[i].split()
#         annotation_depth = test_depth_lines[i]
#         img = annotation[0]
#         print(img)
#         image = Image.open(img)
#         one_tp, one_fp, one_fn, yiyang_tp, yiyang_fp, yiyang_fn, qianhou_tp, qianhou_fp, qianhou_fn = centernet.evaluating_indicator(
#         image, crop=False, count=False, A_=annotation, A_D=annotation_depth)
#
#         one_TP+=one_tp
#         one_FP+=one_fp
#         one_FN+=one_fn
#         yiyang_FP+=yiyang_fp
#         yiyang_FN+=yiyang_fn
#         yiyang_TP+=yiyang_tp
#         qianhou_FP+=qianhou_fp
#         qianhou_FN+=qianhou_fn
#         qianhou_TP+=qianhou_tp
#     print(one_TP, one_FP, one_FN, yiyang_TP, yiyang_FP, yiyang_FN, qianhou_TP, qianhou_FP, qianhou_FN)
#
# if mode=="R@K":
#     import heapq
#     k=50
#     gt=[]
#     pre=[]
#     pre_zhangkai=[]
#     pre_zhangkai_conf=[]
#     pre_zhangkai_conf_img=[]
#
#     # 1先把测试集里面的所有图片都跑一遍，输出一个列表，每张图片检测的一个目标是里面的一个列表
#     # [img1:[[x1,y1,c1,conf1,depth1],[x2,y2,c2,conf2,depth2]....[]...],img2:[],img3:[],.......]
#     #2 然后每张图里面的目标两两配搭，一对目标的平均置信度排序，取前K对，作为Top K ,
#     # 3 然后看这K对中的前后关系跟GT对比一下，看看算对了多少，就是求一下准确率，就完事了。
#     for i in range(len(test_lines)):
#         annotation = test_lines[i].split()
#         annotation_depth = test_depth_lines[i]
#         img = annotation[0]
#         # print(img)
#         image = Image.open(img)
#         p=centernet.recallk(image, A_=annotation, A_D=annotation_depth)
#         # 把p 里面每一对都添加上页码
#         for v in range(len(p)):
#             p[v].append(i)
#         pre.append(p)
#     # 把pre 展开，
#     for l in pre:
#         for ll in l:
#             pre_zhangkai.append(ll)
#
#     for lll in range(len(pre_zhangkai)):
#         if len(pre_zhangkai[lll])==5:
#             pre_zhangkai_conf.append(pre_zhangkai[lll][-2])
#         else:
#             pre_zhangkai_conf.append(0)
#
#     #排序 从小到大排序，R@100 就取后100个
#     sorted_numbers = sorted(pre_zhangkai_conf)
#     sorted_indices = [i[0] for i in sorted(enumerate(pre_zhangkai_conf), key=lambda x: x[1])]
#     # 获得r@K的索引
#     topk= sorted_indices[-k:]
#
#     for count in range(k):
#         pre_zhangkai_conf_img.append(pre_zhangkai[topk[count]])
#     pp=[]
#     for xx in range(len(pre_zhangkai_conf_img)):
#         predict=pre_zhangkai_conf_img[xx]
#         # print(predict)
#         annotation = test_lines[predict[-1]].split()
#         annotation_depth = test_depth_lines[predict[-1]]
#         # print(annotation)
#         #预测的GT 所有的关系
#         GT=centernet.gt_relation_image(annotation,annotation_depth)
#         # 然后看看预测的关系在不在Gt里面，在的话gt[]列表加上一个1 错的话加上一个0，最后求一下正确了多少。
#         pp.append(look(predict,GT))
#     print("R@K:",round(sum(pp)/k,4))
#



if mode=="classification":
    # 写一个预测的每一类关系，都从真实标签里面找一下，找到了就看看类别一致不一致，找不到就扔掉了。
    # 跑出四类的一个准确率和召回率，四类分别是
    # F@1(就一个目标和背景的前后关系) F&B(主在前，客在后) B@F(主在后，客在前) S@S(一样的水平线)
    # 置信度就是两个目标置信度的平均

    # 可以先写一个输出目标的，把可能是两个重复检测的给抑制掉。
    predict_label=[]
    gt_label=[]
    for i in range(len(test_lines)):
        annotation = test_lines[i].split()
        # A=annotation[1:]
        # print(A)
        # annotation_depth = test_depth_lines[i]
        img = annotation[0]

        # print(img)

        image = Image.open(img)
        # 返回 [p,gt] 的 -1,0,1,2
        p,g=centernet.classfi(image, A_=annotation)
        print(p,g)


        if len(p)==len(g) and len(p)>=1:
            predict_label.extend(p)
            gt_label.extend(g)



    print(gt_label)
    print(predict_label)

    C = confusion_matrix(gt_label, predict_label, labels=['0', '1', '2'])
    p= calculate_prediction_recall(gt_label, predict_label, classes=['0', '1', '2'])
    print(p)
    # C=[[1180,259,79],[167,5580,18],[85,37,567]]
    # C = np.array(C)
    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # cb = plt.colorbar(C)

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', size=15)

    plt.tick_params(labelsize=10)  # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 18})
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xticks(range(0, 3), labels=['S@S','F@B','B@F'])  # 将x轴或y轴坐标，刻度 替换为文字/字符
    plt.yticks(range(0, 3), labels=['S@S','F@B','B@F'])
    plt.colorbar()
    plt.show()


        #然后看看predict 在不在GT里面，在的话就是正确的预测,pp里面添加1 不在添加0










