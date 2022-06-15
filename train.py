import sys
sys.path.append('..')
import torch
import torch.nn as nn
import dataset
import ssim
from scfdn import SCFDN as net
import tensorboardX
import argparse
import math
import os
import numpy as np
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
writer = tensorboardX.SummaryWriter('../../f+d_dataset/maxpool/result')
save_path = '../../f+d_dataset/maxpool'
parser = argparse.ArgumentParser(description='SR second idea')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--epoch', default=5500, type=int, help='training epochs')
parser.add_argument('--other_scale_model_path', default='',type=str, help='use this if load model trained in other scale factor as pretrained model')
#parser.add_argument('--other_scale_model_path', default='../../f+d_dataset/net_2x_refine/best.pth',type=str, help='use this if load model trained in other scale factor as pretrained model')
parser.add_argument('--model_path', default='../../f+d_dataset/maxpool/experiment.pth',type=str, help='pretrained model path')
#parser.add_argument('--trainset_data_path', default=['../../../dataset/pic/div2k_train4x'],type=list, help='trainset data path')
#parser.add_argument('--trainset_label_path', default=['../../../dataset/pic/DIV2K_train_HR'],type=list, help='trainset label path')
parser.add_argument('--trainset_data_path', default=['../../../dataset/pic/div2k_train4x','../../../dataset/pic/flickr2k_train4x'],type=list, help='trainset data path')
parser.add_argument('--trainset_label_path', default=['../../../dataset/pic/DIV2K_train_HR','../../../dataset/pic/Flickr2K_HR'],type=list, help='trainset label path')
parser.add_argument('--valset_data_path', default=['../../../dataset/pic/div2k_val4x'],type=list, help='valset data path')
parser.add_argument('--valset_label_path', default=['../../../dataset/pic/DIV2K_valid_HR'],type=list, help='valset label path')
parser.add_argument('--norm', default=True,type=bool, help='norm input to [0,1] or not')
parser.add_argument('--batchsize', default=16, type=int, help='mini-batch size')
parser.add_argument('--train_patchsize', default=128, type=int, help='train patch size of LR image')
parser.add_argument('--lr_max', default=2e-4, type=float, help='max learning rate')
parser.add_argument('--step', default=1000, type=int, help='the step of scheduler')
parser.add_argument('--gamma', default=0.5, type=float, help='the gamma of scheduler')
parser.add_argument('--augment', default=True, type=bool, help='use data augment')
parser.add_argument('--l2', default=False, type=bool, help='use l2 loss')
parser.add_argument('--structure',default=16,type=list,help='the structure of branch')
opt = parser.parse_args()

norm_num=1.
if opt.norm:
    norm_num=255.

def TV_loss_cal(output):
    vec=torch.abs(output[:,:,:-1,:-1]-output[:,:,1:,:-1])
    hor=torch.abs(output[:,:,:-1,:-1]-output[:,:,:-1,1:])
    loss=torch.mean(torch.pow(vec,2)+torch.pow(hor,2))
    return loss

def content_loss(output,label,net):
    output_feature=net(output)
    label_feature=net(label)
    loss=torch.mean(torch.pow(output-label,2))
    return loss

def teature_loss(output,label,net):
    output_feature=net(output)
    label_feature=net(label)
    
    pass

def SSIM_loss_cal(output,label):
    loss=1.-ssim.ms_ssim(output,label,data_range=1.)
    return loss

def setup_seed(seed):
     torch.manual_seed(seed)
     np.random.seed(seed)

def main():
    torch.backends.cudnn.benchmark = True
    setup_seed(1024)
    print('random_seed:1024')
    print('===>Loading dataset')
    if not opt.augment:
        print('not use data augment')
    trainset = dataset.Trainset(opt.trainset_data_path, opt.trainset_label_path, opt.train_patchsize,opt.upscale_factor,opt.augment)
    print(len(trainset))
    train_dataloader = DataLoader(trainset, batch_size=opt.batchsize, num_workers=8, drop_last=True,pin_memory=True, shuffle=True)
    valset = dataset.Valset(opt.valset_data_path,opt.valset_label_path,opt.upscale_factor)
    print(len(valset))
    val_dataloader = DataLoader(valset, batch_size=1, num_workers=0, drop_last=True, pin_memory=True,shuffle=True)

    print('===>Building model')
    model = Net(scale=opt.upscale_factor,resblock=opt.structure)
    start_epoch = 0
    best_PSNR = 0

    model = model.to(device)
    if os.path.exists(opt.other_scale_model_path):
        print('===>loading other scale pretrained model')
        dict = torch.load(opt.other_scale_model_path)['model']
        own_state=model.state_dict()
        for name,param in dict.items():
            if name=='tail.0.weight':
                continue
            else:
                own_state[name].copy_(param)
                
    elif os.path.exists(opt.model_path):
        print('===>Loading pretrained model')
        dict = torch.load(opt.model_path)
        model.load_state_dict(dict['model'])
        start_epoch = dict['epoch'] + 1
        best_PSNR = dict['best']

    criterion = nn.L1Loss()
    if(opt.l2):
        criterion=nn.MSELoss()
        print('use l2 loss')

    criterion = criterion.to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_max)
    for i in range(len(model_optimizer.param_groups)):
        model_optimizer.param_groups[i]['initial_lr']=model_optimizer.param_groups[i]['lr']
    scheduler=torch.optim.lr_scheduler.StepLR(model_optimizer,step_size=opt.step,gamma=opt.gamma)
    for _ in range(start_epoch):
        scheduler.step()
        
    print('===>Training model')
    for epoch in range(start_epoch, opt.epoch):
        print('epoch:{},lr:{}'.format(epoch, model_optimizer.param_groups[0]['lr']))
        loss_avg=train(model, train_dataloader, criterion, model_optimizer)
        writer.add_scalar('train_loss',loss_avg[0],epoch)
        if(epoch%20==0):
            print('epoch:{},MAE_loss:{:.5f}'.format(epoch,loss_avg[0]))
            best_PSNR,eval_PSNR = eval(model, val_dataloader, best_PSNR, epoch)
            writer.add_scalar('best_PSNR',best_PSNR,epoch)
            writer.add_scalar('eval_PSNR',eval_PSNR,epoch)
        scheduler.step()

def train(model, train_dataloader, criterion, model_optimizer):
    MAE_loss_sum=0.
    #TV_loss_sum=0.
    #SSIM_loss_sum=0.
    MAE_weight=1.
    #TV_weight=0.
    #SSIM_weight=1.5
    nums=0  
    for batch in train_dataloader:
        input, label = batch
        input = input.to(device).float()
        input = input / norm_num
        label = label.to(device).float()
        label = label / norm_num

        output = model(input)        
        MAE_loss=criterion(output, label)
        #TV_loss=TV_loss_cal(output)
        #SSIM_loss=SSIM_loss_cal(output,label)
        MAE_loss_sum+=MAE_loss.item()/255.*norm_num
        #TV_loss_sum+=TV_loss.item()
        #SSIM_loss_sum+=SSIM_loss.item()
        nums+=1
        loss=0
        loss+=MAE_weight*MAE_loss
        #loss+=TV_loss
        #loss+=SSIM_weight*SSIM_loss
        #loss=loss/(MAE_weight+SSIM_weight)

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    #return [MAE_loss_sum/nums,TV_loss_sum/nums,SSIM_loss_sum/nums]
    return [MAE_loss_sum/nums]

def eval(model, val_dataloader, best_PSNR, epoch):
    PSNR_sum = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input, label, name = batch
            input = input[:, :, :input.shape[2] // 2 * 2, :input.shape[3] // 2 * 2]
            batchsize, channels, h, w = input.shape;
            input = input.to(device).float()
            input = input / norm_num
            label = label[:, :, :label.shape[2] // (opt.upscale_factor*2) * opt.upscale_factor*2, :label.shape[3] // (opt.upscale_factor*2) * opt.upscale_factor*2]
            label = label.to(device).float()
            label = label / norm_num

            output_full_pic=model(input)
            label=label[:,:,opt.upscale_factor:-opt.upscale_factor,opt.upscale_factor:-opt.upscale_factor]
            output_full_pic=output_full_pic[:,:,opt.upscale_factor:-opt.upscale_factor,opt.upscale_factor:-opt.upscale_factor]
            diff=(label-output_full_pic)/255.*norm_num
            MSE = torch.mean(diff**2)
            PSNR = 10. * torch.log10(1. / MSE)
            PSNR_sum += PSNR.item()

    PSNR_sum /= len(val_dataloader)
    print('PSNR:{:.5f}'.format(PSNR_sum))

    save_model(model, epoch, PSNR_sum, 'experiment.pth')
    print('Model save as experiment.pth')

    if (epoch+20)%opt.step==0:
        save_model(model,epoch,PSNR_sum,'epoch_{}.pth'.format(epoch))

    if best_PSNR < PSNR_sum:
        save_model(model, epoch, PSNR_sum, 'best.pth')
        print('Model save as best.pth')
        return PSNR_sum,PSNR_sum

    return best_PSNR,PSNR_sum


def save_model(model, epoch, best_PSNR, name):
    dict = {'model': model.state_dict(), 'epoch': epoch, 'best': best_PSNR}
    torch.save(dict,os.path.join(save_path, name))


if __name__ == '__main__':
    print(opt)
    main()
