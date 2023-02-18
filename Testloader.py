import os
import numpy as np
import cv2
import torch
import flow_transforms
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils as utils


default_depthlable = 'IRDdataset/PDMV_v2/testset/test_depthmap/'
default_src        = 'IRDdataset/PDMV_v2/testset/test_source/'
default_tar        = 'IRDdataset/PDMV_v2/testset/test_target/'
default_src_crop   = 'IRDdataset/PDMV_v2/testset/source_crop/'
default_tar_crop   = 'IRDdataset/PDMV_v2/testset/target_crop/'
default_R          = 'IRDdataset/PDMV_v2/testset/R_matrix/'
default_T          = 'IRDdataset/PDMV_v2/testset/T_matrix/'
default_delta_xy   = 'IRDdataset/PDMV_v2/testset/delta_xy/'


# 定义读取文件的格式
def default_loader(path):
    # return Image.open(path).convert('RGB')
    return cv2.imread(path, 1)      # 0: grayscale       1: color

def default_loader2(path):
    # return Image.open(path).convert('RGB')
    return cv2.imread(path, 0)      # 0: grayscale       1: color

class MyDataset(Dataset):
    def __init__(self, dep_path=default_depthlable, src_path=default_src, tar_path=default_tar, src_crop_path =default_src_crop,
                 tar_crop_path=default_tar_crop, R_path=default_R, T_path=default_T, delta_xy_path=default_delta_xy):

        self.dep_path      = dep_path
        self.depth_path    = []

        self.src_path = src_path
        self.source_path = []

        self.tar_path      = tar_path
        self.target_path   = []

        self.src_crop_path = src_crop_path
        self.source_crop_path = []

        self.tar_crop_path = tar_crop_path
        self.target_crop_path = []

        self.R_path        = R_path
        self.R_matrix_path = []

        self.T_path        = T_path
        self.T_matrix_path = []

        self.delta_xy_path = delta_xy_path
        self.delta_coord_path = []


        dep_files = os.listdir(dep_path)
        dep_files.sort(key=lambda x: int(x[:-4][5:]))
        for file in dep_files:
            self.depth_path.append(file)
        print("delpthlable_image_numbers: ", len(self.depth_path))

        scr_files = os.listdir(src_path)
        scr_files.sort(key=lambda x: int(x[:-4][5:]))
        for file in scr_files:
            self.source_path.append(file)
        print("source_image_numbers: ", len(self.source_path))

        tar_files = os.listdir(tar_path)
        tar_files.sort(key=lambda x: int(x[:-4][4:]))
        for file in tar_files:
            self.target_path.append(file)
        print("target_image_numbers: ", len(self.target_path))

        scr_crop_files = os.listdir(src_crop_path)
        scr_crop_files.sort(key=lambda x: int(x[:-4][9:]))
        for file in scr_crop_files:
            self.source_crop_path.append(file)
        print("src_crop_numbers: ", len(self.source_crop_path))

        tar_crop_files = os.listdir(tar_crop_path)
        tar_crop_files.sort(key=lambda x: int(x[:-4][9:]))
        for file in tar_crop_files:
            self.target_crop_path.append(file)
        print("tar_crop_numbers: ", len(self.target_crop_path))

        T_files = os.listdir(T_path)
        T_files.sort(key=lambda x: int(x[:-3][2:]))
        for file in T_files:
            self.T_matrix_path.append(file)
        print("T_numbers: ", len(self.T_matrix_path))

        R_files = os.listdir(R_path)
        R_files.sort(key=lambda x: int(x[:-3][2:]))
        for file in R_files:
            self.R_matrix_path.append(file)
        print("R_numbers: ", len(self.R_matrix_path))

        xy_files = os.listdir(delta_xy_path)
        xy_files.sort(key=lambda x: int(x[:-3][9:]))
        for file in xy_files:
            self.delta_coord_path.append(file)
        print("delta_xy_numbers: ", len(self.delta_coord_path))



    def __getitem__(self,index):
        dep       =    cv2.imread(self.dep_path + self.depth_path[index], 0)
        src_img   =    cv2.imread(self.src_path + self.source_path[index], 1)
        tar_img   =    cv2.imread(self.tar_path + self.target_path[index], 1)
        src_crop  =    cv2.imread(self.src_crop_path + self.source_crop_path[index], 1)
        tar_crop  =    cv2.imread(self.tar_crop_path + self.target_crop_path[index], 1)
        T         =    torch.load(self.T_path + self.T_matrix_path[index])
        R         =    torch.load(self.R_path + self.R_matrix_path[index])
        delta_coordinate = torch.load(self.delta_xy_path + self.delta_coord_path[index])

        # print(self.source_path[index])
        # print(self.depth_path[index])
        # print(self.target_path[index])
        # print(self.source_crop_path[index])
        # print(self.target_crop_path[index])
        # print(self.T_matrix_path[index])
        # print(self.R_matrix_path[index])
        # print(self.delta_coord_path[index])


        depthmap = torch.tensor(dep, dtype=torch.float32)
        src_tensor = torch.from_numpy(src_img).permute(2,0,1)
        tar_tensor = torch.from_numpy(tar_img).permute(2,0,1)


        # Data processing
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        ])
        img1 = input_transform(src_crop)
        img2 = input_transform(tar_crop)
        inputs = torch.cat([img1,img2],dim=0)
        # depthmap_crop = torch.from_numpy(depthmap_crop)


        return inputs, delta_coordinate, R, T, depthmap, src_tensor, tar_tensor


    def __len__(self):
        return len(self.source_path)



######## test #########
# import evaluate
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # warp_layer = Spatial_Transform_Layer().to(device)
# train_data = MyDataset()
# train_loader = utils.data.DataLoader(dataset=train_data, batch_size=1)
#
# for index, (images, delta_coordinate, R, T, D, src, tar) in enumerate(train_loader):
#     input = images.to(device)
#     epe,ssim  = evaluate.eval(input, delta_coordinate, R, T, D, src, tar)
#     # print('Exam {} --EPE: {:.5f}  --SSIM: {:.5f}'.format(index, epe,ssim))
#     # print(torch.log(R))
#     print(R)
#     print(index)


