from dataset import MyDataset
from model import Parameter_transformNet

import torch
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
from torch.utils import data

import argparse
import time
import os
import logging

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs to train', default=1000, type=int)
    parser.add_argument('--lr', help='learning rate', default='0.001', type=float)
    parser.add_argument('--momentum', default='0.9', type=float)
    parser.add_argument('--batch_size', help='batch size', default='20',type=int)
    parser.add_argument('--dataset', help='experiment dataset . Default is "mydataset"', default="mydataset", type=str)
    parser.add_argument('--save_path', help='checkpoint save path', default="../Loss10R1T_batch20_lr1e-3", type=str)
    parser.add_argument('--RESUME', help='Resume checkpoints or not', default=False, type=bool)
    parser.add_argument('--stepoch', help='start epoch(0 or checkpoints.epoch)', default="0", type=int)
    args = parser.parse_args()
    return args



def train_model(args, model, criterion1, criterion2, optimizer, TrainLoader):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting Training")

    for epoch in range(args.stepoch, args.epochs):
        torch.cuda.empty_cache()
            
        start_time = time.time()
        testError1 = 0
        testError2 = 0
        for index, (inputs, delta_coord, R, T, D) in enumerate(TrainLoader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            R = R.to(device)
            T = T.to(device)
            out_r, out_t = model.forward(inputs)
            # out_r = model.forward(inputs)

            loss_R = criterion1(out_r, R)**0.5
            loss_T = criterion2(out_t, T)**0.5
            testError1 += loss_R.item()
            testError2 += loss_T.item()

            if index%10 == 0:
                logger.info('iteration={}/{}, loss_R={:.5f}, loss_T={:.5f}'.format(index,len(TrainLoader),loss_R.item()
                                                                                   ,loss_T.item()))

            loss_R.backward(retain_graph=True)
            loss_T.backward()
            optimizer.step()

            # del loss_R
            # del outputs
            # del inputs
            # del target

        AvgError_r = testError1 /len(TrainLoader)
        AvgError_t = testError2 / len(TrainLoader)
        elapsed_time = time.time() - start_time
        logger.info('*****************************************')
        logger.info('Epoch Number: [{}/{}] | Average Error_R: {:.5f} | Average Error_T: {:.5f}'.format(epoch+1, args.epochs,
                                                                                                       AvgError_r, AvgError_t))
        logger.info('Time elapsed: '+str(elapsed_time)+' seconds')
        logger.info('*****************************************\n')

        checkpoint = {
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        }
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        torch.save(checkpoint, args.save_path+"/CHECKPOINT_FILE_with_epoch_"+str(epoch+1))

    logger.info("Finish")
    del TrainLoader


def main():
    args = parseArgs()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if args.RESUME:
        model = Parameter_transformNet()
        # print(model)
        model.load_state_dict(torch.load(args.save_path+'/CHECKPOINT_FILE_with_epoch_'+args.stepoch)['model_state_dict'])
    else:
        model = Parameter_transformNet()
    model = model.to(device)

    logger.info("Arguments are:")
    logger.info(args)
    logger.info("------------------")

    trainData = MyDataset()
    TrainLoader = utils.data.DataLoader(trainData, args.batch_size, shuffle=True)
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_model(args, model, criterion1, criterion2, optimizer, TrainLoader)
    # torch.save(model.state_dict(), "finalmodel")


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main()
