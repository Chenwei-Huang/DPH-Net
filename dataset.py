import os
import numpy as np
import cv2
import math
import random
import torch
import torch.nn as nn
import flow_transforms
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils as utils

# default_depthlable = 'E:/datasets/MVR_source/test/test_depthmap/'
# default_src = 'E:/datasets/MVR_source/test/test_source/'

# default_depthlable = 'E:/datasets/MVR_source/train/train_depthmap/'
# default_src = 'E:/datasets/MVR_source/train/train_source/'

default_depthlable = '../../../MVR/train/train_depthmap/'
default_src = '../../../MVR/train/train_source/'


# 定义读取文件的格式
def default_loader(path):
    # return Image.open(path).convert('RGB')
    return cv2.imread(path, 1)      # 0: grayscale       1: color

def default_loader2(path):
    # return Image.open(path).convert('RGB')
    return cv2.imread(path, 0)      # 0: grayscale       1: color

class MyDataset(Dataset):
    def __init__(self, src_path=default_src, dep_path=default_depthlable,
                 loader=default_loader, loader2=default_loader2):

        self.loader = loader
        self.loader2= loader2

        self.src_path = src_path
        self.source_path = []

        self.dep_path = dep_path
        self.depth_path = []


        scr_files = os.listdir(src_path)
        scr_files.sort(key=lambda x: int(x[:-4][12:]))  #int(x[5:])
        for file in scr_files:
            self.source_path.append(file)
        print("source_image_numbers: ", len(self.source_path))
        # print(self.image_path)


        dep_files = os.listdir(dep_path)
        dep_files.sort(key=lambda x: int(x[:-4][12:]))  #int(x[5:])
        for file in dep_files:
            self.depth_path.append(file)
        print("lable_image_numbers: ", len(self.depth_path))
        # print(self.depth_path)



    def __getitem__(self,index):
        src_img  = self.loader(self.src_path + '/' + str(self.source_path[index]))
        dep  = self.loader2(self.dep_path + '/' + str(self.depth_path[index]))
        # print(self.source_path[index])
        # print(self.depth_path[index])
        # cv2.imshow("source image", src_img)
        # cv2.waitKey(0)
        depthmap = torch.tensor(dep, dtype=torch.float32)
        # print("min",torch.min(depthmap))
        src_tensor = torch.from_numpy(src_img).permute(2,0,1)
        C, H, W = src_tensor.size()
        hh = 100
        ww = 356


        # mesh grid
        xx = torch.range(0, W-1).view(1, -1).repeat(H, 1)
        yy = torch.range(0, H-1).view(-1, 1).repeat(1, W)
        Xa = xx.view(H, W).repeat(1, 1)  # H*W 1-620
        Ya = yy.view(H, W).repeat(1, 1)  # H*W 1-460


        # set translation/rotation parameters
        ta = 20    # Xa  1-620
        tb = 15    # Ya  1-460
        tta = (torch.min(depthmap[hh:ww,hh:ww]) * ta).int().item()
        ttb = (torch.min(depthmap[hh:ww,hh:ww]) * tb).int().item()
        a  = torch.randint(-tta,tta,(1, ))[0].float()
        b  = torch.randint(-ttb,ttb,(1, ))[0].float()
        # a = torch.tensor(18*30)
        # b = torch.tensor(0)
        # c = torch.tensor(0)
        # T = torch.tensor([ta,tb])
        T = torch.tensor([a/torch.min(depthmap[hh:ww,hh:ww]), b/torch.min(depthmap[hh:ww,hh:ww])])

        alpha_n = random.randint(90,120)
        beta_n  = random.randint(480,640)
        gamma_n = random.randint(30,120)
        alpha = math.pi / alpha_n
        beta  = math.pi / beta_n
        gamma = math.pi / gamma_n


        # image coordinate transform
        Ra = torch.tensor([[1,       0,        0],
                           [0, math.cos(alpha),math.sin(alpha)],
                           [0,-math.sin(alpha),math.cos(alpha)]])
        Rb = torch.tensor([[math.cos(beta), 0, -math.sin(beta)],
                           [0, 1, 0],
                           [math.sin(beta), 0, math.cos(beta)]])
        Rc = torch.tensor([[math.cos(gamma), math.sin(gamma), 0],
                           [-math.sin(gamma),math.cos(gamma), 0],
                           [0, 0, 1]])

        R = torch.mm(Rc,torch.mm(Rb,Ra))
        # R = torch.inverse(R)
        # print(R)
        # print(T)

        Xb1 = R[0][0]*Xa + R[0][1]*Ya + R[0][2]+ a*torch.ones(H,W)/depthmap
        Yb1 = R[1][0]*Xa + R[1][1]*Ya + R[1][2]+ b*torch.ones(H,W)/depthmap
        # Zb  = R[2][0]*Xa + R[2][1]*Ya + R[2][2]+ b*torch.ones(H,W)/depthmap
        # Xb1 = Xb1/Zb
        # Yb1 = Yb1/Zb

        # Xb = torch.round(Xb1).type(torch.long)
        # Yb = torch.round(Yb1).type(torch.long)
        delta_x = Xb1 - Xa#.long()
        delta_y = Yb1 - Ya#.long()
        delta_coordinate = torch.stack([delta_x,delta_y])

        ######################################################################
        # visualization
        ######################################################################
        vgrid = torch.stack([Xb1, Yb1]).float()
        vgrid = vgrid.unsqueeze(0)

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        # print(vgrid.requires_grad)

        src = src_tensor.float().unsqueeze(0)
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(src, vgrid)
        mask = torch.ones(src.size())
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        img2_result = output * mask
        # print(img2_result.requires_grad)

        tar = img2_result.squeeze()
        tar_img = tar.permute(1, 2, 0).detach().numpy().astype(np.uint8)
        tar_img_crop = tar_img[hh:ww, hh:ww]
        src_img = src_tensor.permute(1, 2, 0).detach().numpy()
        src_img_crop = src_img[hh:ww, hh:ww]
        # cv2.imshow("img_source", src_img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("img_target", tar_img)
        # cv2.waitKey(0)

        # cv2.imshow("src_img_crop", src_img_crop)
        # cv2.waitKey(0)
        #
        # cv2.imshow("tar_img_crop", tar_img_crop)
        # cv2.waitKey(0)

        # delta_coordinate = delta_coordinate.permute(1,2,0).numpy()
        delta_coordinate = delta_coordinate[:, hh:ww, hh:ww]
        depthmap_crop = dep[hh:ww,hh:ww]


        # Data loading code
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        ])

        # target_transform = transforms.Compose([
        #     flow_transforms.ArrayToTensor(),
        #     # transforms.Normalize(mean=[0, 0], std=[100, 100])
        # ])
        img1 = input_transform(src_img_crop)
        img2 = input_transform(tar_img_crop)
        inputs = torch.cat([img1,img2],dim=0)
        depthmap_crop = torch.from_numpy(depthmap_crop)
        # delta_coordinate = target_transform(delta_coordinate)


        return inputs, delta_coordinate, R, T, depthmap_crop


    def __len__(self):
        return len(self.source_path)



######## test #########
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # warp_layer = Spatial_Transform_Layer().to(device)
# train_data = MyDataset()
# train_loader = utils.data.DataLoader(dataset=train_data, batch_size=2)
#
# for index, (inputs, delta_coord, R, T, depthmap) in enumerate(train_loader):
#     # inputs = inputs.to(device)
#
#     print(index)
#
#     # outputs = warp_layer.forward(parameter, inputs, depth)
#     # cv2.imshow('transformed_image',outputs)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

