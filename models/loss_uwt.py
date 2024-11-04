# UWT loss
# strongly following  SWT loss
# Training transformer models by wavelet losses improves quantitative and visual performance in single image super-resolution
# modified by SAM J

import torch.nn as nn
# from basicsr.utils.registry import LOSS_REGISTRY
import models.SWT as SWT
import pywt
import numpy as np
from math import exp

# @LOSS_REGISTRY.register()
class UWTLoss(nn.Module):
    def __init__(self, train_weight, loss_weight_ll=0.05, loss_weight_lh=0.025, loss_weight_hl=0.025, loss_weight_hh=0.02, reduction='mean'):
        super(UWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh
        self.loss_weights = [self.loss_weight_ll, self.loss_weight_lh, self.loss_weight_hl, self.loss_weight_hh]
        self.train_weight = train_weight

        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        wavelet = pywt.Wavelet('sym19')
            
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2*np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")

        ## wavelet bands of sr image
        sr_img_y       = 16.0 + (pred[:,0:1,:,:]*65.481 + pred[:,1:2,:,:]*128.553 + pred[:,2:,:,:]*24.966)

        wavelet_sr  = sfm(sr_img_y)[0]

        LL_sr   = wavelet_sr[:,0:1, :, :]
        LH_sr   = wavelet_sr[:,1:2, :, :]
        HL_sr   = wavelet_sr[:,2:3, :, :]
        HH_sr   = wavelet_sr[:,3:, :, :]     

        ## wavelet bands of hr image
        hr_img_y       = 16.0 + (target[:,0:1,:,:]*65.481 + target[:,1:2,:,:]*128.553 + target[:,2:,:,:]*24.966)
     
        wavelet_hr     = sfm(hr_img_y)[0]

        LL_hr   = wavelet_hr[:,0:1, :, :]
        LH_hr   = wavelet_hr[:,1:2, :, :]
        HL_hr   = wavelet_hr[:,2:3, :, :]
        HH_hr   = wavelet_hr[:,3:, :, :]

        sr_allbands = [LL_sr, LH_sr, HL_sr, HH_sr]
        lr_allbands = [LL_hr, LH_hr, HL_hr, HH_hr]
        loss_bands = 0
        for sr_subband, lr_subband, loss_weight  in zip(sr_allbands, lr_allbands, self.loss_weights):
            dis = self.criterion(sr_subband, lr_subband)
            if dis  < 0.5:
                loss_subband = loss_weight*(0.5*dis**2)
            else:
                loss_subband = loss_weight*(dis-0.5)
            loss_bands += loss_subband

        return loss_bands * self.train_weight