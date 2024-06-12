from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
###wo
from model.MSHNet.loss import MSHNet_SLSIoULoss as MSHNet_SoftIoULoss
# from model.ISViMamba.IS_ViMamba import IS_ViMamba
from model import *
from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name

        ###wo:损失函数
        #self.cal_loss = SoftIoULoss()
        if self.model_name == 'MSHNet':
            self.cal_loss = MSHNet_SoftIoULoss()
        else:
            self.cal_loss = SoftIoULoss()


        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()

        ###wo
        elif model_name == 'MSHNet':
            self.model = MSHNet(3)
        elif model_name == 'SCTransNet':
            self.cal_loss = nn.BCELoss(size_average=True)
            config_vit = config.get_SCTrans_config()
            if mode == 'train':
                self.model = SCTransNet(config_vit, mode='train', deepsuper=True)
            else:
                self.model = SCTransNet(config_vit, mode='test', deepsuper=True)
        elif model_name == "IS_ViMamba":
                self.model = IS_ViMamba()
        
    def forward(self, img):
        return self.model(img)

    # def loss(self,pred,labels,warm_epoch,epoch):
    #     loss = self.cal_loss(pred,labels,warm_epoch,epoch)
    #     return loss

    def loss(self, preds, gt_masks):

        ###wo:我加的，为移植SCTransNet
        if self.model_name == 'SCTransNet':
            if isinstance(preds, list):
                loss_total = 0
                for i in range(len(preds)):
                    pred = preds[i]
                    gt_mask = gt_masks[i]
                    loss = self.cal_loss(pred, gt_mask)
                    loss_total = loss_total + loss
                return loss_total / len(preds)

            elif isinstance(preds, tuple):
                a = []
                for i in range(len(preds)):
                    pred = preds[i]
                    loss = self.cal_loss(pred, gt_masks)
                    a.append(loss)
                loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
                return loss_total
        ###
        else:
            loss = self.cal_loss(preds, gt_masks)
        return loss
