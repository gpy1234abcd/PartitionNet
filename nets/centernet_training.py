import math
from functools import partial

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('tkaGg')  # 大小写无所谓 tkaGg ,TkAgg 都行
import matplotlib.pyplot as plt

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def relation_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    # -------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    # -------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    # -------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    # -------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    # -------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    # -------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss



#在这里面写上对预测的深度数值的排位损失
def reg_l1_whdepth_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,3)
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def depth_ranking_loss(pred, target, mask):
    # --------------------------------#
    # 排位损失，根据mask找到下方中心点的位置，然后根据位置找到深度，建立关系的标签‘
    # 然后写上每一对目标的前后关系的排序损失
    # --------------------------------#
    pred = pred.permute(0, 2, 3, 1)
    #pred=[batch,128,128,3]  target=[batch,128,128,3]  mask=[batch,128,128]
    #这是目标深度的ground truth
    target_depth=target[0,:,:,2]
    batch_img=mask.shape[0]

    # plt.imshow(mask[0,:,:].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()
    # print(type(mask[0,:,:].cpu().detach().numpy()))

    loss=0
    #拿到每一个bach图片的mask的位置，即目标中心点的位置
    for i in range(batch_img):
        indices = torch.nonzero(torch.eq(mask[i,:,:], 1))

        # print(indices.shape[0])
        # plt.imshow(mask[i, :, :].cpu().detach().numpy())
        #         # plt.colorbar()
        #         # plt.show()
        # 有indices.shape[0]个目标下中心点
        #当一张图像上多个目标的时候，就开始构造一个ranking 损失
        #这是对第一个batch做的损失 ，就是相当于一张图片上的ranking损失
        if indices.shape[0]>1:

            xx=0
            for i_ in range(indices.shape[0]-1):
                for j_ in range(i_+1,indices.shape[0]):
                    xx+=1
                    #拿到两个配对的点的坐标和深度数值
                    ii=indices[i_]
                    jj=indices[j_]
                    #再从target里拿到深度信息，看看这俩目标哪个在前 哪个在后
                    ii_depth = target[i,ii[0],ii[1],2]
                    jj_depth = target[i,jj[0],jj[1],2]
                    #按照设定的深度阈值是8 来构建目标与目标之间前后关系的类别
                    if ii_depth - jj_depth < 0:
                        #同一个深度水平
                        if abs(ii_depth-jj_depth)<=8:
                            #在这里拿到预测的两个目标的深度数值，求一个相对关系的ranking 损失
                            loss+=(pred[i,ii[0],ii[1],2]-pred[i,jj[0],jj[1],2])**2
                            # print("i在j在同一个深度的关系")
                        else:
                            loss+=torch.log(1+torch.exp(ii_depth-jj_depth))
                            # print("i在j的前面")
                    else:
                        if abs(ii_depth-jj_depth)<=8:
                            loss += (pred[i, ii[0], ii[1], 2] - pred[i, jj[0], jj[1], 2]) ** 2
                        else:
                            loss += torch.log(1 + torch.exp(jj_depth-ii_depth))
            loss=loss/xx
    loss=loss/batch_img

                            # print("i在j的后面")

    #如果没有相对的前后关系，就把int类型的loss 转成tensor类型，要不就报错了
    loss=torch.tensor(loss)
    return loss
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
