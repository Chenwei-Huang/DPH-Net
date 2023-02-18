import torch
import torch.nn as nn
import numpy as np
import cv2



def spatial_transform(out_r, out_t, depthmap, source, target):

    # inputs = (inputs+0.5)*255
    R = out_r[0]
    T = out_t[0]
    # print(out_r.grad)
    # print(out_t.grad)
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
    # print(Xb1.requires_grad) #True

    # Xb1 = torch.round(Xb1).type(torch.long)
    # Yb1 = torch.round(Yb1).type(torch.long)
    # print(Xb1.requires_grad) #False


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
    tar_img_t = tar[:,hh:ww, hh:ww]
    tar_img   = tar_tensor[:,hh:ww, hh:ww]


    # tar_img_t = tar.permute(1, 2, 0).numpy()[hh:ww, hh:ww]
    # tar_img_t = tar_img_t.astype(np.uint8)
    # src_img   = src_tensor.permute(1, 2, 0).numpy()[hh:ww, hh:ww]
    # tar_img   = tar_tensor.permute(1, 2, 0).numpy()[hh:ww, hh:ww]
    # cv2.imshow("img_source", src_img)
    # cv2.waitKey(0)
    # cv2.imwrite("src_img.jpg", src_img)
    # cv2.imshow("img_target_t", tar_img_t)
    # cv2.waitKey(0)
    # cv2.imwrite("tar_img_t.jpg",tar_img_t)
    # cv2.imshow("img_target", tar_img)
    # cv2.waitKey(0)
    # cv2.imwrite("tar_img.jpg",tar_img)


    s = tar_img_t.float().unsqueeze(0)
    t = tar_img.float().unsqueeze(0)
    # s = torch.tensor(tar_img).float().unsqueeze(0)
    # t = torch.tensor(tar_img_t).float().unsqueeze(0)
    # SSIM = pytorch_ssim.ssim(s, t)


    return t, s


