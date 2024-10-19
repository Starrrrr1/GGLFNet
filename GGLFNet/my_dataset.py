import numpy
import numpy as np
import torch.utils.data as data
import torch
import os

from wandb.old.summary import h5py

import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

# import nibabel as nb



class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, predict=False):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        self.predict=predict

        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # print(self.img_list[idx])
        # print(self.manual[idx])
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        mask = np.clip(manual, a_min=0, a_max=255)


        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        if self.predict==True:
            return img,self.img_list[idx],mask

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

class SynapseDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, predict=False):
        super(SynapseDataset, self).__init__()
        self.flag = "train_npz" if train else "test_vol_h5"
        data_root = os.path.join(root, "Synapse", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        if train==True:
            self.data_list = [os.path.join(data_root, i) for i in os.listdir(data_root)]
            print("Train samples:{}".format(self.data_list.__len__()))
        else:
            self.data_list=[]
            for i in os.listdir(data_root):
                data=h5py.File(os.path.join(data_root, i))
                for j in range(data['image'].shape[0]):
                    a=[os.path.join(data_root, i),j]
                    self.data_list.append(a)
            print("Test samples:{}".format(self.data_list.__len__()))
        self.predict=predict



    def __getitem__(self, idx):
        if self.flag=="train_npz":
            data = np.load(self.data_list[idx])
            img, mask = data['image'], data['label']
            # np.set_printoptions(threshold=np.inf)
            # print(img)

            img=img*255
            # mask=mask*10
            # np.set_printoptions(threshold=np.inf)
            # print(img)
            # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
            mask = Image.fromarray(mask).convert('L')
            img=Image.fromarray(img).convert('RGB')
            # print(img)
            # import matplotlib.pyplot as plt
            # plt.title("img:{}".format(idx))
            # plt.imshow(img)
            # plt.show()
            # plt.title("mask:{}".format(idx))
            # plt.imshow(mask)
            # plt.show()
            # print(idx)

            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            if self.predict==True:
                return img,self.data_list[idx],mask
            else:
                return img, mask
        else:
            data = h5py.File(self.data_list[idx][0])
            n=self.data_list[idx][1]
            img, mask = data['image'][n], data['label'][n]
            img = img * 255

            mask = Image.fromarray(mask).convert('L')
            img = Image.fromarray(img).convert('RGB')
            #print(img)


            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            if self.predict == True:
                return img, self.data_list[idx], mask
            else:
                return img, mask

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


class ACDCDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, predict=False):
        super(ACDCDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, "ACDC_PNG", self.flag,"img")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        if train==True:
            self.data_list = [os.path.join(data_root, i) for i in os.listdir(data_root)]
            print("Train samples:{}".format(self.data_list.__len__()))
        else:
            self.data_list = [os.path.join(data_root, i) for i in os.listdir(data_root)]
            print("Test samples:{}".format(self.data_list.__len__()))
        self.predict=predict



    def __getitem__(self, idx):
        if self.flag=="train":
            img = Image.open(self.data_list[idx]).convert('RGB')
            mask = Image.open(self.data_list[idx].replace("img","gt")).convert('L')


            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            if self.predict==True:
                return img,self.data_list[idx],mask
            else:
                return img, mask
        else:
            img = Image.open(self.data_list[idx]).convert('RGB')
            mask = Image.open(self.data_list[idx].replace("img", "gt")).convert('L')


            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            if self.predict == True:
                return img, self.data_list[idx], mask
            else:
                return img, mask

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

######################################
class PolypDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, predict=False):
        super(PolypDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, "Polyp", self.flag,"image")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        if train==True:
            self.data_list = [os.path.join(data_root, i) for i in os.listdir(data_root)]
            print("Train samples:{}".format(self.data_list.__len__()))
        else:
            self.data_list = [os.path.join(data_root, i) for i in os.listdir(data_root)]
            print("Test samples:{}".format(self.data_list.__len__()))
        self.predict=predict



    def __getitem__(self, idx):
        if self.flag=="train":
            img = Image.open(self.data_list[idx]).convert('RGB')
            mask = Image.open(self.data_list[idx].replace("image","mask")).convert('L')
            # np.set_printoptions(threshold=np.inf)
            # print(np.array(mask))
            mask=np.array(mask)/255
            mask=Image.fromarray(mask)

            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            if self.predict==True:
                return img,self.data_list[idx],mask
            else:
                return img, mask
        else:
            img = Image.open(self.data_list[idx]).convert('RGB')
            mask = Image.open(self.data_list[idx].replace("image", "mask")).convert('L')
            mask = np.array(mask)/255
            mask = Image.fromarray(mask)

            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            if self.predict == True:
                return img, self.data_list[idx], mask
            else:
                return img, mask

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets



if __name__ == '__main__':

    data_transform = {
        "train": T.Compose([T.ToTensor()],),
        }
    # train_dataset = PotsdamSegmentation(voc_root='./dataset_vh_256',transforms=data_transform['train'])
    train_dataset = SynapseDataset(root='./data',train=True, transforms=data_transform['train'])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)

    channels_sum, channel_squared_sum = 0, 0
    num_batches = len(train_loader)

    #计算各个像素的个数
    # num_label=[0,0,0,0,0,0]
    # for data,label in train_loader:
    #     label=label.reshape(-1)
    #     for a in label:
    #         num_label[a]=num_label[a]+1
    # print(num_label)
    #[269014873, 263979133, 180194919, 18768491, 46854081, 337188503]

    #计算损失权重
    # num_label=[269014873, 263979133, 180194919, 18768491, 46854081, 337188503]
    # num=269014873+263979133+180194919+18768491+46854081+337188503
    # for i in num_label:
    #     print(num/float(i)/100)
    # 0.041484695160330404
    # 0.04227606884366879
    # 0.06193293385814059
    # 0.5946135999958654
    # 0.23818629587463255
    # 0.03309721387505315

    #计算均值和方差
    # for data, _ in train_loader:
    #     channels_sum += torch.mean(data, dim=[0, 2, 3])
    #     channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
    #
    # mean = channels_sum / num_batches
    # std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
    # print(mean, std)
    #pot: tensor([0.3412, 0.3637, 0.3378]) tensor([0.1402, 0.1384, 0.1439])
    #vh: tensor([0.4696, 0.3191, 0.3144]) tensor([0.2148, 0.1551, 0.1481])
    #Drive: tensor([0.4974, 0.2706, 0.1624]) tensor([0.3481, 0.1900, 0.1079])
    # 计算各个类别的比例
    class_label=[0,0,0,0,0,0,0,0,0]
    for data, label in train_loader:
        class_label[0]+=torch.sum(label.eq(0))
        class_label[1] += torch.sum(label.eq(1))
        class_label[2] += torch.sum(label.eq(2))
        class_label[3] += torch.sum(label.eq(3))
        class_label[4] += torch.sum(label.eq(4))
        class_label[5] += torch.sum(label.eq(5))
        class_label[6] += torch.sum(label.eq(6))
        class_label[7] += torch.sum(label.eq(7))
        class_label[8] += torch.sum(label.eq(8))
    print(class_label)

    data_transform = {
        "test": T.Compose([T.ToTensor()], ),
    }
    # train_dataset = PotsdamSegmentation(voc_root='./dataset_vh_256',transforms=data_transform['train'])
    test_dataset = SynapseDataset(root='./data', train=False, transforms=data_transform['test'])
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=1,
                              # shuffle=True,
                              pin_memory=True,
                              collate_fn=test_dataset.collate_fn)

    channels_sum, channel_squared_sum = 0, 0
    num_batches = len(test_loader)

    # class_label = [0,0,0,0,0,0,0,0,0]
    for data, label in test_loader:
        class_label[0] += torch.sum(label.eq(0))
        class_label[1] += torch.sum(label.eq(1))
        class_label[2] += torch.sum(label.eq(2))
        class_label[3] += torch.sum(label.eq(3))
        class_label[4] += torch.sum(label.eq(4))
        class_label[5] += torch.sum(label.eq(5))
        class_label[6] += torch.sum(label.eq(6))
        class_label[7] += torch.sum(label.eq(7))
        class_label[8] += torch.sum(label.eq(8))
    sum=np.sum(class_label)-class_label[0]
    print(class_label)
    print(sum)
    proportion=[]
    for i in class_label:
        proportion.append(i/sum)
    print(proportion)
