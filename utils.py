import datetime
import math
import os
import sys
import json
import pickle
import random
from collections import defaultdict, deque
import time
import torch.nn.functional as F
from medpy import metric
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import cv2
import torch.distributed as dist
import matplotlib.pyplot as plt


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)  # 12, 6, 256, 256
        target = self._one_hot_encoder(target)  # [12, 6, 256, 256]
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


# def criterion(inputs, target, device, orig_x,epoch,n_classes=9):
#     # loss_weight = torch.as_tensor([0.1131, 0.9046, 0.9929, 0.8553, 0.9524, 0.5534, 0.3579, 0.8089, 0.4893,
#     #                                0.9283, 0.7771, 0.5604, 0.5726, 0.8127, 0.5917, 0.6896, 0.9264, 0.5986,
#     #                                0.5325, 0.4841, 0.9594], device=device)
#     #loss_weight=torch.as_tensor([0.04148 ,0.04227, 0.06193, 0.59461, 0.2381,0.03309],device=device)
#     #loss_weight = torch.as_tensor([0.04148, 0.04227, 0.06193, 0.59461, 0.0001, 0.03309], device=device)
#     #loss_weight = torch.as_tensor([1.0, 1., 1., 1., 1., 1.], device=device)
#     #DRIVE(0:背景，1:前景)
#     # loss_weight = torch.as_tensor([1.0, 2.0], device=device)
#     #synapse
#     loss_weight = torch.as_tensor([1, 1., 1., 1., 1., 1.,1.,1.,1.], device=device)
#     #ACDC
#     #loss_weight = torch.as_tensor([1, 1., 1., 1.], device=device)
#     #Polyp
#     #loss_weight = torch.as_tensor([1.0, 1.0], device=device)
#     losses = {}
#     diceloss=DiceLoss(n_classes=n_classes)
#     #focalloss=FocalLoss(weight=loss_weight)
#
#
#     for name, x in inputs.items():
#         losses[name] = nn.functional.cross_entropy(x, target, loss_weight, ignore_index=255)+\
#             diceloss(x,target,loss_weight)
#
#     if len(losses) == 1:
#         return losses['out']
#
#     return losses['out'] + 0.5 * losses['aux']

def criterion(inputs, target, device, orig_x, epoch, n_classes=9):
    # loss_weight = torch.as_tensor([0.1131, 0.9046, 0.9929, 0.8553, 0.9524, 0.5534, 0.3579, 0.8089, 0.4893,
    #                                0.9283, 0.7771, 0.5604, 0.5726, 0.8127, 0.5917, 0.6896, 0.9264, 0.5986,
    #                                0.5325, 0.4841, 0.9594], device=device)
    # loss_weight=torch.as_tensor([0.04148 ,0.04227, 0.06193, 0.59461, 0.2381,0.03309],device=device)
    # loss_weight = torch.as_tensor([0.04148, 0.04227, 0.06193, 0.59461, 0.0001, 0.03309], device=device)
    # loss_weight = torch.as_tensor([1.0, 1., 1., 1., 1., 1.], device=device)
    # DRIVE(0:背景，1:前景)
    # loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    # synapse
    loss_weight = torch.as_tensor([1, 1., 1., 1., 1., 1.,1.,1.,1.], device=device)
    # ACDC
    # loss_weight = torch.as_tensor([1., 1., 1., 1.], device=device)
    # Polyp
    #loss_weight = torch.as_tensor([1.0, 1.0], device=device)
    losses = {}
    diceloss = DiceLoss(n_classes=n_classes)
    # focalloss=FocalLoss(weight=loss_weight)

    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, loss_weight, ignore_index=255) + \
                       diceloss(x, target, loss_weight)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def train_one_epoch_seg(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, n_classes=2,
                        scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, device, image, epoch, n_classes)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别的召回率
        rec = torch.diag(h) / h.sum(0)
        # 计算F1值
        f1 = 2 * acc * rec / (acc + rec)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu, rec, f1

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu, rec, f1 = self.compute()
        return (
            'global correct: {:.2f}\n'
            'precision: {}\n'
            'recall: {}\n'
            'f1_score: {}\n'
            'mean f1: {:.2f}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}').format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (rec * 100).tolist()],
            ['{:.2f}'.format(i) for i in (f1 * 100).tolist()],
            f1.mean().item() * 100,
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100
        )


# 获取dice
def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def calculate_metric_percase(pred, gt):
    # print(pred.shape,gt.shape)
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        print(1)
        return 1, 0
    else:
        print(2)
        return 0, 0


def multiclass_metric(x: torch.Tensor, target: torch.Tensor, n_classes, ignore_index: int = -100, epsilon=0):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    hd95 = 0.

    metric_list = []
    metric_dice = [0] * n_classes
    metric_hd95 = [0] * n_classes
    for i in range(0, n_classes):
        # 处理成numpy
        print(i)
        target1 = target.cpu().detach().numpy()
        x1 = x.cpu().detach().numpy()
        # print(np.sum(x1),np.sum(target1))
        # print(x1.shape,target1.shape)
        dice_inc = calculate_metric_percase(x1 == i, target1 == i)[0]
        dice += dice_inc
        metric_dice[i] += dice_inc
        hd95_inc = calculate_metric_percase(x1 == i, target1 == i)[1]
        # print(i)
        # print(np.sum(x1[0][i]),np.sum(target1[0][i]))
        # print(hd95_inc)
        hd95 += hd95_inc
        metric_hd95[i] += hd95_inc

    # print("dice: {}".format(dice))
    # print(metric_dice)
    dice = dice / n_classes
    hd95 = hd95 / n_classes
    metric_list.append(dice)
    metric_list.append(hd95)
    metric_list.append(metric_dice)
    metric_list.append(metric_hd95)

    return metric_list


# def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
#     # Dice loss (objective to minimize) between 0 and 1
#     x = nn.functional.softmax(x, dim=1)
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(x, target, ignore_index=ignore_index)

class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None
        self.cumulative_hd = None
        self.cumulative_metric_dice = None
        self.cumulative_metric_hd = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat_pre = None
        self.mat_tar = None

    def update(self, pred, target):
        # compute the Dice score, ignoring background
        if self.mat_tar is None and self.mat_pre is None:
            self.mat_pre = pred
            self.mat_tar = target
        self.mat_pre = torch.cat([self.mat_pre, pred], dim=0)
        self.mat_tar = torch.cat([self.mat_tar, target], dim=0)

        # pred=pred.float()
        # pred = F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        # dice_target = build_target(target, self.num_classes, self.ignore_index)
        # metric_all=multiclass_metric(pred[:, :], target[:, :], ignore_index=self.ignore_index)
        # self.cumulative_dice += metric_all[0]
        # self.cumulative_hd += metric_all[1]
        # self.cumulative_metric_dice=list(np.add(self.cumulative_metric_dice, metric_all[2]))
        # #print(self.cumulative_metric_dice)
        # self.cumulative_metric_hd=list(np.add(self.cumulative_metric_hd, metric_all[3]))
        # #print(self.cumulative_metric_hd)
        # self.count += 1

    @property
    def value(self):
        pred = self.mat_pre
        target = self.mat_tar

        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.cumulative_hd is None:
            self.cumulative_hd = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.cumulative_metric_dice is None:
            self.cumulative_metric_dice = [0.0] * self.num_classes
        if self.cumulative_metric_hd is None:
            self.cumulative_metric_hd = [0.0] * self.num_classes

        # print(pred.shape,target.shape)
        metric_all = multiclass_metric(pred, target, n_classes=self.num_classes, ignore_index=self.ignore_index)
        self.cumulative_dice += metric_all[0]
        self.cumulative_hd += metric_all[1]
        self.cumulative_metric_dice = list(np.add(self.cumulative_metric_dice, metric_all[2]))
        # print(self.cumulative_metric_dice)
        self.cumulative_metric_hd = list(np.add(self.cumulative_metric_hd, metric_all[3]))

        return [self.cumulative_dice, self.cumulative_hd \
            , [i for i in self.cumulative_metric_dice] \
            , [i for i in self.cumulative_metric_hd]]

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.cumulative_hd is not None:
            self.cumulative_hd.zero_()

        if self.cumulative_metric_dice is not None:
            self.cumulative_metric_dice = None

        if self.cumulative_metric_hd is not None:
            self.cumulative_metric_hd = None

        if self.mat_pre is not None:
            self.mat_pre = None

        if self.mat_tar is not None:
            self.mat_tar = None

        # if self.count is not None:
        #     self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)


@torch.no_grad()
def evaluate_seg(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    dice = DiceCoefficient(num_classes=num_classes, ignore_index=255)
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 500, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            output = output.to(device)

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output.argmax(1).float(), target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
