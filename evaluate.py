import torch
import torch.nn as nn
import pytorch_ssim
import numpy as np
import cv2
import os


def eval(inputs, delta_coord, out_r, out_t, depthmap, source, target, index):

    # inputs = (inputs+0.5)*255
    R = out_r[0]
    T = out_t[0]
    hh = 100
    ww = 356
    depth = depthmap[0].float()
    # N, C, H, W = inputs.size()
    W = 620
    H = 460
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    Xa = xx.view(H, W).repeat(1, 1)  # W 0-619
    Ya = yy.view(H, W).repeat(1, 1)  # W 0-459
    # print(Xa)
    # print(Ya)


    # set translation/rotation parameters
    ta = T[0] # Xa
    tb = T[1] # Ya
    a = (torch.min(depth[hh:ww, hh:ww]) * ta).float().item()
    b = (torch.min(depth[hh:ww, hh:ww]) * tb).float().item()
    # print('a2:',a)
    # print('b2:',b)
    # print(R)
    # print(depth)


    # image coordinate transform
    Xb1 = R[0][0] * Xa + R[0][1] * Ya + R[0][2] + a * torch.ones(H, W) / depth
    Yb1 = R[1][0] * Xa + R[1][1] * Ya + R[1][2] + b * torch.ones(H, W) / depth
    # Zb  = R[2][0]*Xa + R[2][1]*Ya + R[2][2]+ b*torch.ones(H,W)/depthmap
    # Xb1 = Xb1/Zb
    # Yb1 = Yb1/Zb



    # Xb1 = torch.round(Xb1).type(torch.long)
    # Yb1 = torch.round(Yb1).type(torch.long)
    delta_x = Xb1 - Xa#.long()
    delta_y = Yb1 - Ya#.long()
    # out_displacement = torch.stack([delta_x, delta_y])
    # print('out: ', out_displacement)
    # EPE = torch.dist(out_displacement, delta_coord, p=2)/(H*W)

    # Euclidean distance
    delta_x  = delta_x[hh:ww, hh:ww].float()
    delta_y  = delta_y[hh:ww, hh:ww].float()
    label_Xb = delta_coord[0][0,:,:]
    label_Yb = delta_coord[0][1,:,:]
    # print(delta_x-label_Xb)
    # print(delta_y-label_Yb)
    error_matrix = ((delta_x - label_Xb) ** 2 + (delta_y - label_Yb) ** 2) ** 0.5
    EPE = torch.sum(error_matrix) / (256*256)

######################################################################
    # visualization
######################################################################
    src_tensor = source[0]
    tar_tensor = target[0]
    vgrid = torch.stack([Xb1, Yb1]).float()
    vgrid = vgrid.unsqueeze(0)
    if src_tensor.is_cuda:
        vgrid = vgrid.cuda()

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    # print(vgrid.requires_grad)

    src = src_tensor.float().unsqueeze(0)
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(src, vgrid)
    mask = torch.ones(src.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    img2_result = output * mask
    # print(img2_result.requires_grad)

    tar = img2_result.squeeze()
    # tar_img_t = tar[:,hh:ww, hh:ww]
    # tar_img   = tar_tensor[:,hh:ww, hh:ww]


    tar_img_t = tar.permute(1, 2, 0).cpu().detach().numpy()[hh:ww, hh:ww]
    tar_img_t = tar_img_t.astype(np.uint8)
    src_img   = src_tensor.permute(1, 2, 0).cpu().detach().numpy()[hh:ww, hh:ww]
    tar_img   = tar_tensor.permute(1, 2, 0).cpu().detach().numpy()[hh:ww, hh:ww]

######################Save single image######################
#    if not os.path.isdir('result_epoch_84_addmse/src'):
#        os.mkdir('result_epoch_84_addmse/src')
#    cv2.imwrite("result_epoch_84_addmse/src/src_"+str(index)+".jpg", src_img)

#    if not os.path.isdir('result_epoch_84_addmse/tar_pred'):
#        os.mkdir('result_epoch_84_addmse/tar_pred')
#    cv2.imwrite("result_epoch_84_addmse/tar_pred/tar_pred_"+str(index)+".jpg",tar_img_t)

#    if not os.path.isdir('result_epoch_84_addmse/tar'):
#        os.mkdir('result_epoch_84_addmse/tar')
#    cv2.imwrite("result_epoch_84_addmse/tar/tar_"+str(index)+".jpg",tar_img)


#####################Save addimage######################
    if not os.path.isdir('result_epoch_84_addmse/addimage'):
        os.mkdir('result_epoch_84_addmse/addimage')
    cv2.imwrite("result_epoch_84_addmse/addimage/regis_"+str(index)+".jpg",tar_img//2+tar_img_t//2)

    # ######################Save addimage######################
    # if not os.path.isdir('result_epoch_84_addmse/addimage_without_onlineupdating'):
    #     os.mkdir('result_epoch_84_addmse/addimage_without_onlineupdating')
    # cv2.imwrite("result_epoch_84_addmse/addimage_without_onlineupdating/regis_" + str(index) + ".jpg", tar_img // 2 + tar_img_t // 2)


    # s = tar_img_t.float().unsqueeze(0)
    # t = tar_img.float().unsqueeze(0)
    s = torch.tensor(tar_img).float().unsqueeze(0)
    t = torch.tensor(tar_img_t).float().unsqueeze(0)
    SSIM = pytorch_ssim.ssim(s, t)

    RMSE = np.sqrt(np.mean((tar_img - tar_img_t) ** 2))

    return EPE, SSIM, RMSE


