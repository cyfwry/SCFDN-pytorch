import sys
sys.path.append('..')
import torch
import torch.nn as nn
import dataset
from scfdn import SCFDN as net
import argparse
import math
import os
import numpy as np
from torch.utils.data import DataLoader
import time
import tqdm
import cv2
import glob
import tensorboardX
from ssim import SSIM

#writer=tensorboardX.SummaryWriter()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
img_save_path = './test_img/result/'
parser = argparse.ArgumentParser(description='SR second idea')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
#parser.add_argument('--model_path', default='../../f+d_dataset/net_4x_refine/best.pth',type=str, help='pretrained model path')
parser.add_argument('--model_path', default='',type=str, help='pretrained model path')
parser.add_argument('--valset_data_path', default=['../../../dataset/pic/div2k_val4x'],type=list, help='valset data path')
parser.add_argument('--valset_label_path', default=['../../../dataset/pic/DIV2K_valid_HR'],type=list, help='valset label path')
#parser.add_argument('--valset_data_path', default=['../../../dataset/test_img/Set5_4x'],type=list, help='valset data path')
#parser.add_argument('--valset_label_path', default=['../../../dataset/test_img/Set5'],type=list, help='valset label path')
parser.add_argument('--norm', default=True,type=bool, help='norm input to [0,1] or not')
parser.add_argument('--structure',default=16,type=list,help='the structure of branch')
parser.add_argument('--is_y',default=True,type=bool,help='test on Y channel')
parser.add_argument('--pad',default=False,type=bool,help='padding if shape can not be divide by 2')
parser.add_argument('--time',default=False,type=bool,help='eval time')
parser.add_argument('--psnr',default=True,type=bool,help='eval psnr')
parser.add_argument('--img',default=False,type=bool,help='generatre img')
opt = parser.parse_args()

norm_num=1.
if opt.norm:
    norm_num=255.

def main():
    torch.backends.cudnn.benchmark = True
    print('===>Loading dataset')
    valset = dataset.Valset(opt.valset_data_path,opt.valset_label_path,opt.upscale_factor)
    print(len(valset))
    val_dataloader = DataLoader(valset, batch_size=1, num_workers=0, drop_last=True, pin_memory=True,shuffle=False)

    print('===>Building model')
    model = net.Net(scale=opt.upscale_factor,resblock=opt.structure)
    model = model.to(device)

    if os.path.exists(opt.model_path):
        print('===>loading pretrained model')
        dict = torch.load(opt.model_path)
        model.load_state_dict(dict['model'])
    print('===>Evaling model')
    with torch.no_grad():
        if opt.time:
            eval_time(model, val_dataloader)
        if opt.psnr:
            if opt.is_y:
                ssim=SSIM(channel=1)
            else:
                ssim=SSIM()
            eval_psnr(model, val_dataloader, ssim)
        if opt.img:
            val_dataloader=[]
            img_path=glob.glob('./test_img/*.png')
            for i in img_path:
                val_dataloader.append(torch.Tensor(cv2.imread(i).transpose([2,0,1])).unsqueeze(0))

            generate_img(model, val_dataloader)

def eval_time(model, val_dataloader):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for batch in tqdm.tqdm(val_dataloader):
        input, label, _ = batch
        input=input[:,:,:input.shape[2]//2*2,:input.shape[3]//2*2]
        input = input.to(device).float()
        input = input / norm_num

        output = model(input)        
    
    torch.cuda.synchronize()
    sum=0
    for batch in tqdm.tqdm(val_dataloader):
        input, label, _ = batch
        input=input[:,:,:input.shape[2]//2*2,:input.shape[3]//2*2]
        input = input.to(device).float()
        input = input / norm_num
        
        begin.record()
        output = model(input)
        end.record()
        torch.cuda.synchronize()
        sum+=begin.elapsed_time(end)

    print(sum)

def eval_psnr(model, val_dataloader, ssim):
    sum=0
    sum_ssim=0
    length=len(val_dataloader)
    for batch in tqdm.tqdm(val_dataloader):
        input, label, name = batch
        if opt.pad:
            new_shape=input.shape
            new_shape_h=(new_shape[2]+1)//2*2
            new_shape_w=(new_shape[3]+1)//2*2
            new_input=torch.zeros([new_shape[0],new_shape[1],new_shape_h,new_shape_w])
            new_input[:,:,:input.shape[2],:input.shape[3]]=input
            for i in range(new_shape_h):
                new_input[:,:,i,new_shape_w-1]=new_input[:,:,i,new_shape_w-2]
            for i in range(new_shape_w):
                new_input[:,:,new_shape_h-1,i]=new_input[:,:,new_shape_h-2,i]
            new_input[:,:,new_shape_h-1,new_shape_w-1]=new_input[:,:,new_shape_h-2,new_shape_w-2]
            input=new_input
        else:
            input=input[:,:,:input.shape[2]//2*2,:input.shape[3]//2*2]
        input = input.to(device).float()
        input = input / norm_num
        if opt.pad:
            label=label[:,:,:label.shape[2]//opt.upscale_factor*opt.upscale_factor,:label.shape[3]//opt.upscale_factor*opt.upscale_factor]
        else:
            label=label[:,:,:label.shape[2]//2//opt.upscale_factor*2*opt.upscale_factor,:label.shape[3]//2//opt.upscale_factor*2*opt.upscale_factor]
        label=label.to(device).float()
        label=label / norm_num
        output = model(input)

        if opt.pad:
            output=output[:,:,:label.shape[2],:label.shape[3]]
        label=label[:,:,opt.upscale_factor:-opt.upscale_factor,opt.upscale_factor:-opt.upscale_factor]
        output=output[:,:,opt.upscale_factor:-opt.upscale_factor,opt.upscale_factor:-opt.upscale_factor]
        label=torch.clamp(label,0.,255./norm_num)
        output=torch.clamp(output,0.,255./norm_num)
        if opt.is_y:
            label=label[:,0]*0.098+label[:,1]*0.504+label[:,2]*0.257+16./norm_num
            output=output[:,0]*0.098+output[:,1]*0.504+output[:,2]*0.257+16./norm_num
            label=label.unsqueeze(1)
            output=output.unsqueeze(1)

        diff=(output-label)/255.*norm_num
        MSE=torch.mean(diff**2)
        psnr=10.*torch.log10(1./MSE)
        sum+=psnr.item()
        ssim_val=ssim(output.cpu()*norm_num,label.cpu()*norm_num)
        sum_ssim+=ssim_val
        save_path='./result'

    print(sum/length)
    print(sum_ssim/length)

def generate_img(model, val_dataloader):
    i=0
    if len(val_dataloader[0])==2:
        for batch in tqdm.tqdm(val_dataloader):
            input, label,name = batch
            input=input[:,:,:input.shape[2]//2*2,:input.shape[3]//2*2]
            input = input.to(device).float()
            input = input / norm_num
            label=label[:,:,:label.shape[2]//2//opt.upscale_factor*2*opt.upscale_factor,:label.shape[3]//2//opt.upscale_factor*2*opt.upscale_factor]
            label=label.to(device).float()
            label=label / norm_num
            output = model(input)
            output=output.clamp(0,255./norm_num).cpu().squeeze().numpy()
            output=output*norm_num
            output=output.astype(np.uint8)
            output=np.transpose(output,[1,2,0])
            cv2.imwrite(img_save_path+str(i)+'.png',output)
            label=label.clamp(0,255./norm_num).cpu().squeeze().numpy()
            label=label*norm_num
            label=label.astype(np.uint8)
            label=np.transpose(label,[1,2,0])
            cv2.imwrite(img_save_path+str(i)+'ori.png',label)
            i+=1
    else:
        for batch in tqdm.tqdm(val_dataloader):
            input = batch
            input=input[:,:,:input.shape[2]//2*2,:input.shape[3]//2*2]
            input = input.to(device).float()
            input = input / norm_num
            output = model(input)
            output=output.clamp(0,255./norm_num).cpu().squeeze().numpy()
            output=output*norm_num
            output=output.astype(np.uint8)
            output=np.transpose(output,[1,2,0])
            cv2.imwrite(img_save_path+str(i)+'.png',output)
            i+=1


if __name__ == '__main__':
    print(opt)
    main()
