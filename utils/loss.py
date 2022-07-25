"""
References
https://github.com/JohnYKiyo/Noise2Void/blob/master/02_training_test_Noise2Void.ipynb
"""

import torch
import torch.nn.functional as F
import numpy as np

def pixel_mse_loss(predictions, targets, pixel_pos):
    mask = torch.zeros(targets.shape).to(targets.device)
    
    for i,(h,w) in enumerate(pixel_pos):
        mask[i, :, h, w] = 1.

    
    return F.mse_loss(predictions*mask, targets*mask)*10000
