import argparse
import time
import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from torch.optim import Adagrad
from loss import AverageMeter
from tqdm import tqdm
import torch.utils.data as Data
import torchvision.transforms as transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
# parser.add_argument("--model_names", default=['ACM','ALCNet','DNANet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'], type=str,nargs="+",
#                     help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--model_names", default=['IS_ViMamba'], type=str,nargs="+",
                    help="model_name: 'IS_ViMamba','SCTransNet', 'MSHNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--dataset_names", default=['Dataset-mask_256x256'], type=str,nargs='+',
                    help="dataset_name: 'Dataset-mask','NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")

parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='/home/l/ws_dataset/for_PRCV24/', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default='/home/l/ws_running/for_PRCV24/log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")

###wo
parser.add_argument('--warm-epoch',type=int,default=5)
parser.add_argument('--base-size',type=int,default=256)
parser.add_argument('--crop-size',type=int,default=256)

global opt
global net
opt = parser.parse_args()
net = None

seed_pytorch(opt.seed)

def train():
    ###wo
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((640, 640))
    # ])
    if opt.model_name == 'MSHNet':
        trainset = IRSTD_Dataset(opt, mode='train')
        valset = IRSTD_Dataset(opt, mode='val')

        # train_loader = Data.DataLoader(trainset, opt.batchSize, shuffle=True, drop_last=True,collate_fn=PRCV_collate_fn)
        train_loader = Data.DataLoader(trainset, opt.batchSize, shuffle=True, drop_last=True)
        ###wo:为自定义的采样器
        # train_loader = Data.DataLoader(trainset, opt.batchSize, shuffle=False, drop_last=True,transforms=transform)
        val_loader = Data.DataLoader(valset, 1, drop_last=False,collate_fn=PRCV_collate_fn)
    else:
        train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
        train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
        ###wo:为自定义的采样器
        # train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False,collate_fn=PRCV_collate_fn,pin_memory=True)

    net = Net(model_name=opt.model_name, mode='train').cuda()
    net.train()
    
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    
    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
    
    ### Default settings                
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs':400, 'step': [200, 300], 'gamma': 0.1}
    
    ### Default settings of DNANet                
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs':1500, 'min_lr':1e-5}


    opt.nEpochs = opt.scheduler_settings['epochs']
        
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)

    ### wo
    if model_name == 'MSHNet':
        optimizer = Adagrad(filter(lambda p: p.requires_grad, net.model.parameters()), lr=0.05)
        MSHNet_down = nn.MaxPool2d(2,2)
        # losses = AverageMeter()
        # tag = False

    for idx_epoch in range(epoch_state, opt.nEpochs):

        ### wo
        start_time = datetime.datetime.now()
        if model_name == 'MSHNet':
            train_loader = tqdm(train_loader)
            losses = AverageMeter()
            tag = False

        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            ###wo:
            # img, gt_mask = Variable(batch_to_same(img)).cuda(), Variable(batch_to_same(gt_mask)).cuda()
            if img.shape[0] == 1:
                continue

            ###wo:为移植MSHNet
            # pred = net.forward(img)
            if opt.model_name == 'MSHNet':
                if idx_epoch > opt.warm_epoch:
                    tag =True
                masks,pred = net.model(img,tag)
                loss = 0
                loss = loss + net.cal_loss(pred, gt_mask, opt.warm_epoch, idx_epoch)
                for j in range(len(masks)):
                    if j > 0:
                        gt_mask = MSHNet_down(gt_mask)
                    loss = loss + net.cal_loss(masks[j], gt_mask, opt.warm_epoch, idx_epoch)
                loss = loss / (len(masks) + 1)
            ### wo:原来的
            else:
                pred = net.forward(img)
                loss = net.loss(pred, gt_mask)
                total_loss_epoch.append(loss.detach().cpu())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###wo
            if opt.model_name == 'MSHNet':
                losses.update(loss.item(),pred.size(0))
                print(f'Epoch {idx_epoch},loss {losses.avg}')

        scheduler.step()

        ### wo
        end_time = datetime.datetime.now()
        execution_time = end_time - start_time
        print(f"Epoch:{idx_epoch + 1},代码执行时间：{execution_time}")


        if (idx_epoch + 1) % 10 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            
        if (idx_epoch + 1) % 50 == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            ###wo:每50轮
            test(save_pth)
            
        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            test(save_pth)

    ###wo
    params = sum(p.numel() for p in net.model.parameters() if p.requires_grad)
    with open(modelfile, 'wt') as f:
        print(f'模型参数：{params}\n', file=f)
        print(f'\n', file=f)
        print(f'模型结构：\n{net.model}\n', file=f)

def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    ###wo
    # test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False,collate_fn=PRCV_collate_fn)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=True)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        img = Variable(img).cuda()
        pred = net.forward(img)
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    ###wo:写到文件中
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    
def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

###wo:将一个批次的数据大小保持一致
def batch_to_same(batched_inputs:List[torch.Tensor]):
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
    # batch_shape = (len(batched_inputs), batched_inputs[0].shape[0], max_size, max_size)
    ###wo:640
    batch_shape = (len(batched_inputs), batched_inputs[0].shape[0], 640, 640)

    padded_images = batched_inputs[0].new_full(batch_shape, 0.0)
    for padded_img, img in zip(padded_images, batched_inputs):
        # h, w = img.shape[1:]
        ###wo:
        h, w = img.shape[-2:]

        padded_img[..., :h, :w].copy_(img)

    # return padded_images, np.array(image_sizes_orig)
    return padded_images

def PRCV_collate_fn(batch):
    # print(f'批次：{len(batch)}')
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    data = batch_to_same(data)
    label = batch_to_same(label)
    return data,label

if __name__ == '__main__':
    # for dataset_name in opt.dataset_names:

    ###wo
    for _,dataset_name in enumerate(opt.dataset_names):
        opt.dataset_name = dataset_name
        # for model_name in opt.model_names:
        ###wo
        for _,model_name in enumerate(opt.model_names):
            opt.model_name = model_name
            opt.save = '/home/l/ws_running/for_PRCV24/log'
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            # opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            ###wo
            # opt.dataset_name = opt.dataset_name[1][1:-1]
            opt.model_name = opt.model_name
            opt.dataset_name = opt.dataset_name

            note = '调试SCTransNet和MSHNet'
            opt.run_root = opt.save + '/' + opt.dataset_name + '_' + opt.model_name+ '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '_' + note
            opt.save = opt.run_root
            os.mkdir(opt.run_root)
            # opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            logfile = opt.run_root + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_')
            modelfile = opt.run_root + '/' + opt.dataset_name + '_' + opt.model_name
            opt.f = open( logfile + '.txt', 'w')
            # opt.modelfile = open( modelfile + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name+'----train  start')
            train()
            print(opt.dataset_name + '\t' + opt.model_name+'----train  end')
            print('\n')

            opt.f.close()

            opt.run_root = None


