from utils import *
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image,ImageOps
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
import torchvision.transforms as transforms
from typing import List
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        ###wo:数据增强
        self.tranform = augumentation()
        self.tranform_size = size_to_same()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//','/'))
            ###wo
            # mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '_pixels0.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//','/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

        ###wo:两种统一输入图片大小的方法
        # img_patch,mask_patch = self.tranform_size(img_patch,mask_patch)
        # img_patch = self.tranform_size(img_patch)
        # mask_patch = self.tranform_size(mask_patch)

        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//','/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:,:,0]
        
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 


class augumentation(object):
    def __call__(self, input, target):
        #wo:推测是左右反转
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        #wo:推测是上下反转
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        #wo:推测是对角线反转
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)

        ###wo:基础的方法统一图片的大小
        # transform = transforms.Compose([
        #     # transforms.ToTensor(),
        #     transforms.Resize((640,640)),
        #     # transforms.ToTensor(),
        #     # transforms.ToPILImage()
        # ])
        # input = transform(input)
        # target = transform(target)

        # input = np.resize(input,(640,640))
        # target = np.resize(target,(640,640))

        # input = Image.fromarray(input)
        # target = Image.fromarray(target)
        # input = input.resize((640,640))
        # target = target.resize((640,640))
        # input = img_to_array(input)
        # target = img_to_array(target)
        # input = np.squeeze(input)
        # target = np.squeeze(target)

        # print(input.shape)
        # print(target.shape)

        return input, target

class size_to_same(object):
    # def __call__(self, input, target):
        ###wo:基础的方法统一图片的大小
        # transform = transforms.Compose([
        #     # transforms.ToTensor(),
        #     transforms.Resize((640, 640)),
        #     # transforms.ToTensor(),
        #     # transforms.ToPILImage()
        # ])
        # input = transform(input)
        # target = transform(target)
        # return input,target
    def __call__(self, batched_inputs:List[torch.Tensor]):
        """
                Args:
                  batch_inputs: 图片张量列表
                Return:
                  padded_images: 填充后的批量图片张量
                  image_sizes_orig: 原始图片尺寸信息
            """
        ## 保留原始图片尺寸
        image_sizes_orig = [[image.shape[-2], image.shape[-1]] for image in batched_inputs]
        ## 找到最大尺寸
        max_size = max([max(image_size[0], image_size[1]) for image_size in image_sizes_orig])

        ## 构造批量形状 (batch_size, channel, max_size, max_size)
        batch_shape = (len(batched_inputs), batched_inputs[0].shape[0], max_size, max_size)

        padded_images = batched_inputs[0].new_full(batch_shape, 0.0)
        for padded_img, img in zip(padded_images, batched_inputs):
            # h, w = img.shape[1:]
            ###wo:
            h, w = img.shape[-2:]

            padded_img[..., :h, :w].copy_(img)

        # return padded_images, np.array(image_sizes_orig)
        return padded_images
