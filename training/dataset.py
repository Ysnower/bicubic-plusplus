from torch.utils.data import Dataset
import glob
from utils.img_utils import Imread_Modcrop
import random
import numpy as np
from utils.deg import Degrader
from torchvision.transforms import Compose, ToTensor
import cv2
import time


def ImageTransform():
    return Compose([ToTensor(), ])


class SRDataset1(Dataset):
    def __init__(self, conf_data, conf_deg, is_train):
        super(SRDataset1, self).__init__()
        self.hr_path = conf_data['hr_path']
        self.lr_path = conf_data['lr_path']
        self.hr_images = sorted(glob.glob(self.hr_path[0] + '/*'))
        self.lr_images = sorted(glob.glob(self.lr_path[0] + '/*'))
        self.scale = int(conf_data['scale'])
        self.imread_modcrop = Imread_Modcrop(scale=self.scale)
        self.patch_cropsize = conf_data['patch_cropsize']
        self.augment = conf_data['augment']
        self.enable_degradation = conf_deg['blur'] or conf_deg['img_noise']
        self.is_train = is_train
        self.degrader = Degrader(ds_rate=self.scale,
                                 enable_blur=conf_deg['blur'],
                                 enable_img_noise=conf_deg['img_noise'],
                                 enable_kernel_noise=conf_deg['kernel_noise'],
                                 kernel_size=conf_deg['ksize'],
                                 rate_isotropic=conf_deg['rate_iso'],
                                 sig_min=conf_deg['sig_min'],
                                 sig_max=conf_deg['sig_max'],
                                 img_noise_level=conf_deg['img_noise_level'])

    @staticmethod
    def augment_image(img, trans):
        img_aug = img
        if trans == 0:
            img_aug = np.rot90(img_aug, 0)
        elif trans == 1:
            img_aug = np.rot90(img_aug, 1)
        elif trans == 2:
            img_aug = np.rot90(img_aug, 2)
        elif trans == 3:
            img_aug = np.rot90(img_aug, 3)
        elif trans == 4:
            img_aug = np.rot90(img_aug, 0)
            img_aug = np.flip(img_aug, 0)
        elif trans == 5:
            img_aug = np.rot90(img_aug, 0)
            img_aug = np.flip(img_aug, 1)
        elif trans == 6:
            img_aug = np.rot90(img_aug, 1)
            img_aug = np.flip(img_aug, 0)
        elif trans == 7:
            img_aug = np.rot90(img_aug, 1)
            img_aug = np.flip(img_aug, 1)
        return img_aug

    def __getitem__(self, item):
        return_dict = {}
        img_hr, img_lr = self.imread_modcrop(self.hr_images[item], self.lr_images[item], is_train=self.is_train)
        if self.enable_degradation:
            img_lr = self.degrader(img_hr)
        hr_dim = img_hr.shape
        if self.patch_cropsize:
            if hr_dim[0] < self.patch_cropsize:
                img_hr = cv2.copyMakeBorder(img_hr, 0, self.patch_cropsize - hr_dim[0], 0, 0, cv2.BORDER_REFLECT)
                img_lr = cv2.copyMakeBorder(img_lr, 0, (self.patch_cropsize - hr_dim[0])//self.scale, 0, 0,
                                            cv2.BORDER_REFLECT)
            if hr_dim[1] < self.patch_cropsize:
                img_hr = cv2.copyMakeBorder(img_hr, 0,  0, 0, self.patch_cropsize - hr_dim[1], cv2.BORDER_REFLECT)
                img_lr = cv2.copyMakeBorder(img_lr, 0, 0, 0, (self.patch_cropsize - hr_dim[1]) // self.scale,
                                            cv2.BORDER_REFLECT)
            hr_dim = img_hr.shape
            i = int((hr_dim[0] - self.patch_cropsize) * random.random() // self.scale) * self.scale
            j = int((hr_dim[1] - self.patch_cropsize) * random.random() // self.scale) * self.scale
            i_lr = i // self.scale
            j_lr = j // self.scale
            img_hr = (img_hr[i:i + self.patch_cropsize,
                      j:j + self.patch_cropsize,
                      :])
            img_lr = (img_lr[i_lr:i_lr + self.patch_cropsize // self.scale,
                      j_lr:j_lr + self.patch_cropsize // self.scale,
                      :])
        if self.augment:
            t = random.randint(0, 7)
            img_hr = self.augment_image(img_hr, t)
            img_lr = self.augment_image(img_lr, t)
        return_dict['img_lr'] = ImageTransform()(img_lr.copy())
        return_dict['img_hr'] = ImageTransform()(img_hr.copy())
        return return_dict

    def __len__(self):
        return len(self.hr_images)


