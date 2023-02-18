from Testloader import MyDataset
from model import Parameter_transformNet
import evaluate
import model_finetune
import pytorch_ssim

import torch.optim as optim
import torch.nn as nn
import torch.utils as utils
import torch
from torch.utils import data
import numpy as np

import time
import logging


def Save_errlist(list1,filename):
    file = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        file.write(str(list1[i]))              # write函数不能写int类型的参数，所以使用str()转化                      # 相当于Tab一下，换一个单元格
        file.write('\n')                       # 写完一个元素立马换行
    file.close()

def Save_paralist(list1,filename):
    file = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        file.write(str(list1[i]))              # write函数不能写int类型的参数，所以使用str()转化                      # 相当于Tab一下，换一个单元格
        file.write(' ')
    file.close()

def test_model(TestLoader, criterion1, criterion2, criterion3):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Start Testing")
    counter= 0
    epeadd = 0
    ssimadd= 0
    rmseadd= 0


    start_time = time.time()
    for index, (inputs, delta_coord, R, T, D, src, tar) in enumerate(TestLoader):
        counter += 1
        torch.cuda.empty_cache()
        inputs = inputs.to(device)
        src = src.to(device)
        tar = tar.to(device)
        # model  = copy.deepcopy(initial_model) # gradient miss
        # model = initial_model                 # shallow copy
        learning_rate = 0.0001
        model = Parameter_transformNet()
        model.load_state_dict(torch.load('CHECKPOINT_FILE_with_epoch_84')['model_state_dict'])  # GPU use
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        time1 = time.time()


        i = 0
        flag = 0
        ssim_loss = 1

        while i < 40 and ssim_loss > 0.15:
            optimizer.zero_grad()
            if i == 20:
                learning_rate = learning_rate*0.1
            if i == 15 and ssim_loss>0.9:
                break
            # if i == 39 and ssim_loss > 0.4:
            #     i = 19
            #     flag = 1
            # if i == 39 and flag == 1:
            #     break

            out_r, out_t  = model.forward(inputs)
            out_r = out_r.cpu()
            out_t = out_t.cpu()
            out_img, tar_img = model_finetune.spatial_transform(out_r, out_t, D, src, tar)
            ssim_loss = 1-criterion3(out_img, tar_img)
            # print('{:.5f}'.format(ssim_loss.item()))
            print('Test Example {} :  --ssim_loss = {:.3f}'.format(index, ssim_loss.item()))
            ssim_loss.backward()
            optimizer.step()
            i += 1



        out_r, out_t = model.forward(inputs)
        out_r = out_r.cpu()
        out_t = out_t.cpu()
        inputs = inputs.cpu()
        epe, ssim, rmse = evaluate.eval(inputs, delta_coord, out_r, out_t, D, src, tar, index)

        time2 = time.time()

        L2_parm_R = criterion1(out_r, R).item()**0.5
        L2_parm_T = criterion2(out_t, T).item()**0.5

        print('Test Example {} : --L2_parm_R = {:.3f}  --L2_parm_T = {:.3f}  --RMSE = {:.3f}  --EPE = {:.3f}  --SSIM = {:.3f} --time = {:.2f}'.format(
            index, L2_parm_R, L2_parm_T, rmse, epe, ssim, time2-time1))
        epeadd  += epe.item()
        ssimadd += ssim.item()
        rmseadd += rmse.item()
        del model
        del inputs
        del delta_coord
        del T, D, R, src, tar
        del optimizer


    average_epe  = epeadd  / counter
    average_ssim = ssimadd / counter
    average_mse  = rmseadd  / counter
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("************************")
    logger.info("Average EPE:  " + str(average_epe))
    logger.info("Average SSIM: " + str(average_ssim))
    logger.info("Average RMSE: " + str(average_mse))
    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: " + str(elapsed_time))
    logger.info("************************")

    # Save_errlist(EPE, 'Average_EPE')


def main():
    # model = Parameter_transformNet()
    #print(model)
    # model.load_state_dict(torch.load('changelabeldepthsize_batch20_lr1e-3/CHECKPOINT_FILE_with_epoch_16')['model_state_dict']) #GPU use
    #setmodecpu
    # model.load_state_dict(torch.load('changesize_doubleloss_checkpoint_batch20_lr1e-3/CHECKPOINT_FILE_with_epoch_3',map_location='cpu')['model_state_dict'])
    # print parameters of some layers
    # for name in model.state_dict():
    #     print(name)
    # print(model.state_dict()['conv_layer3.weight'])
    # model.load_state_dict(torch.load('model_final_Gaussian',map_location='cpu'))
    # print(model.state_dict()['conv_layer3.weight'])

    #GPU use
    # model = model.to(device)

    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    criterion3 = pytorch_ssim.SSIM()

    TestData = MyDataset()
    TestLoader = utils.data.DataLoader(TestData, 1)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    test_model(TestLoader, criterion1, criterion2, criterion3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
main()
