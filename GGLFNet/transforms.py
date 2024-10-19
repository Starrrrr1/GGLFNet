import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import imgaug as ia
import imgaug.augmenters as iaa

def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size=size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomRotation(object):
    def __init__(self, Rota_Angle):
        self.Rota_Angle = Rota_Angle

    def __call__(self, image, target):
        if random.random()>0.5:
            angle=random.randint(-self.Rota_Angle,self.Rota_Angle)
            image = F.rotate(image,angle)
            target = F.rotate(target,angle)
        return image, target

class ColorJitter(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random()>self.prob:
            image=T.ColorJitter(0.5,0.5,0.5)(image)
        return image, target

class GaussianBlur(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random()>self.prob:
            image=F.gaussian_blur(image,21,1.0)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image.copy())
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask, -1)
    for colour in range(9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
class augment_seg(object):
    def __init__(self):
        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
            # sometimes(
            #     iaa.Superpixels(
            #         p_replace=(0, 1.0),
            #         n_segments=(20, 200)
            #     )
            # ),
            #
            # # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
            # iaa.OneOf([
            #     iaa.GaussianBlur((0, 3.0)),
            #     iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
            #     iaa.MedianBlur(k=(3, 11)),
            # ]),
            #
            # # 锐化处理
            # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            #
            # # 浮雕效果
            # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            #
            # # 边缘检测，将检测到的赋值0或者255然后叠在原图上
            # sometimes(iaa.OneOf([
            #     iaa.EdgeDetect(alpha=(0, 0.7)),
            #     iaa.DirectedEdgeDetect(
            #         alpha=(0, 0.7), direction=(0.0, 1.0)
            #     ),
            # ])),
            #
            # # 加入高斯噪声
            # iaa.AdditiveGaussianNoise(
            #     loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
            # ),
            #
            # # 将1%到10%的像素设置为黑色
            # # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
            # iaa.OneOf([
            #     iaa.Dropout((0.01, 0.1), per_channel=0.5),
            #     iaa.CoarseDropout(
            #         (0.03, 0.15), size_percent=(0.02, 0.05),
            #         per_channel=0.2
            #     ),
            # ]),
            #
            # # 5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
            # iaa.Invert(0.05, per_channel=True),
            #
            # # 每个像素随机加减-10到10之间的数
            # iaa.Add((-10, 10), per_channel=0.5),
            #
            # # 像素乘上0.5或者1.5之间的数字.
            # iaa.Multiply((0.5, 1.5), per_channel=0.5),
            #
            # # 将整个图像的对比度变为原来的一半或者二倍
            # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            #
            # # 将RGB变成灰度图然后乘alpha加在原图上
            # iaa.Grayscale(alpha=(0.0, 1.0)),
            #
            # # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
            # sometimes(
            #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
            # ),
            #
            # # 扭曲图像的局部区域
            # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
        ], random_order=True)
    def __call__(self, img, seg ):
        img=np.array(img)
        seg=np.array(seg)
        seg = mask_to_onehot(seg)
        aug_det = self.img_aug.to_deterministic()
        image_aug = aug_det.augment_image(img)
        segmap = ia.SegmentationMapsOnImage( seg , shape=img.shape )
        segmap_aug = aug_det.augment_segmentation_maps( segmap )
        segmap_aug = segmap_aug.get_arr()
        segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
        return image_aug , segmap_aug

