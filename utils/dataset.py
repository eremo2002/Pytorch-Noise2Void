"""
"""

import torch
from torch import nn
import torch.nn.functional as F
import glob
import cv2
import numpy as np
import copy

class N2V_dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, patch_size=(64, 64), train_val='train'):
        super(N2V_dataset, self).__init__()
        
        self.img_path_list = [img_file for img_file in glob.glob(img_dir+'/**/*.jpg', recursive=True)]                
        self.patch_size = patch_size
        self.train_val = train_val                
    
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):        
        img_path = self.img_path_list[idx]
        
        source = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        source = self.add_noise(source)

        # normalize [0, 1]
        source = source / 255.   
        
        # add channel axis
        source = np.expand_dims(source, axis=-1)        
        
        # crop patch
        source, bbox = self.random_crop(source, self.patch_size)

        # center pixel masking        
        source, target, blid_pos = self.center_pixel_mask(source)

        # numpy to torch.tensor
        source = torch.tensor(source, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return source.permute(2, 0, 1), target.permute(2, 0, 1), torch.tensor(blid_pos)


    def add_noise(self, img):
        poisson_noise = np.random.poisson(30, (img.shape[0], img.shape[1]))
        gaussian_noise = np.random.normal(0, 25, (img.shape[0], img.shape[1]))        
        
        noisy_img = img + poisson_noise
        noisy_img = noisy_img + gaussian_noise        
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return noisy_img
      

    def random_crop(self, img, patch_size):        
        if type(patch_size) != tuple:
            raise TypeError('patch_size must be tuple')
        
        h, w, _ = img.shape
        
        top = np.random.randint(0, h - patch_size[0])
        left = np.random.randint(0, w - patch_size[1])
        bottom = top + patch_size[0]
        right = left + patch_size[1]
        
        patch = img[top:bottom, left:right, :]
        return patch, (top, left, bottom, right)
      
    
    def center_pixel_mask(self, img):        
        h, w, _ = img.shape
        
        center_pixel_h = h//2
        center_pixel_w = w//2

        random_pixel_h = np.random.randint(0, h)
        random_pixel_w = np.random.randint(0, w)        
        
        while (center_pixel_h == random_pixel_h) and (center_pixel_w == random_pixel_w):
            random_pixel_h = np.random.randint(0, h)
            random_pixel_w = np.random.randint(0, w)
                
        target = copy.deepcopy(img)
        source = copy.deepcopy(img)
        
        source[center_pixel_h, center_pixel_w, :] = img[random_pixel_h, random_pixel_w, :]

        return source, target, [center_pixel_h, center_pixel_w]


