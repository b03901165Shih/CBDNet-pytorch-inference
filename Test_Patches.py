
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from CBDNet_model import CBDNet_pretrain

class noisy_Dataset(Dataset):
    def __init__(self, img_path_head):
        self.img_path_head      = img_path_head
        self.imagenames         = np.sort([x for x in os.listdir(img_path_head)])
        self.numOfImages        = len(self.imagenames)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    #to [-1, 1]
        ])
    def __len__(self):
        return self.numOfImages
    def __getitem__(self, idx):
        img = Image.open(self.img_path_head+self.imagenames[idx]).convert('RGB')
        c_img = self.transform(img.copy())
        img.close()
        x_pad = 0; y_pad = 0;
        if(c_img.size(1)%4!=0):
            x_pad = (4-c_img.size(1)%4)
        if(c_img.size(2)%4!=0):
            y_pad = (4-c_img.size(2)%4)
        c_img = F.pad(c_img[None], (0, x_pad, 0, y_pad), 'reflect').squeeze()
        return c_img, x_pad, y_pad

def vis_img(img):
    img = img.data.cpu().numpy()
    img = np.clip(img, 0, 1)
    img = np.moveaxis(img, 0, 2)
    return img

        
if __name__=='__main__':
    folderTest  = '../testsets/';
    imageSets   = ['DND_patches/','Nam_patches/','NC12/'];
    
    setTestCur  = imageSets[0];     # current testing dataset
    batch_size  = 1
    showResult  = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loader = DataLoader(noisy_Dataset(img_path_head = folderTest+setTestCur), 
                          batch_size=batch_size, shuffle=False)

    if(setTestCur=='DND_patches/' or setTestCur=='NC12/'):
        model = CBDNet_pretrain('model/CBDNet.pth').to(device)
    else:
        model = CBDNet_pretrain('model/CBDNet_JPEG.pth').to(device)
    model.eval()

    for param in list(model.parameters()):
        param.requires_grad=False
    
    
    for i, (t_data, x_pad, y_pad) in enumerate(test_loader):
        target_img  = Variable(t_data.type(torch.FloatTensor)).to(device)
        restored_img = model(target_img)
        restored_img = restored_img.cpu()
        if(int(x_pad[0].cpu().numpy()) != 0):
            restored_img = restored_img[:,:,:-int(x_pad[0].cpu().numpy()),:]
            t_data = t_data[:,:,:-int(x_pad[0].cpu().numpy()),:]
        if(int(y_pad[0].cpu().numpy()) != 0):
            restored_img = restored_img[:,:,:,:-int(y_pad[0].cpu().numpy())]
            t_data = t_data[:,:,:,:-int(y_pad[0].cpu().numpy())]
        
        display = np.concatenate([vis_img(t_data[0]), vis_img(restored_img[0])], axis=1)
        diff = (abs(t_data[0]-restored_img[0]).sum()).data.numpy()
        print('Difference:', diff)
        if(showResult):
            plt.imshow(display); plt.show()
    
