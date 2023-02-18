import torch
import torch.nn as nn
import torch.nn.functional as F
from correlation import correlation


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def conv_(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

class Parameter_transformNet(nn.Module):
    def __init__(self,batchNorm=True):
        super(Parameter_transformNet, self).__init__()

        self.batchNorm  = batchNorm
        self.conv1      = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2      = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3      = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)


        self.conv4a     = conv(self.batchNorm, 593,  256, stride=2)
        self.conv5a     = conv(self.batchNorm, 256,  128, stride=2)
        self.conv6a     = conv(self.batchNorm, 128,   64, stride=2)                   # n*64*4*4
        self.conv7a     = conv_(self.batchNorm, 64,    8, kernel_size=2, stride=1)    # n*8*3*3
        self.conv8a     = conv_(self.batchNorm,  8,    1, kernel_size=1, stride=1)


        self.conv4b = conv(self.batchNorm, 593, 256, stride=2)
        self.conv5b = conv(self.batchNorm, 256, 128, stride=2)
        self.conv6b = conv(self.batchNorm, 128, 64, stride=2)  # n*64*4*4
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1b = nn.Linear(64*4*4, 64)
        self.relu = nn.ReLU()
        self.fc2b = nn.Linear(64, 2)


    def forward(self, x):

        x1 = x[:,:3,:,:]
        x2 = x[:,3:,:,:]
        # print('x1.shape:{} , x2.shape:{}'.format(x1.size(),x2.size()))

        # Encoder
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Correlation Layer
        # out_conv_redir = self.conv_redir(out_conv3a)
        corr = correlation.FunctionCorrelation(tenFirst=out_conv3a, tenSecond=out_conv3b)
        corr = self.corr_activation(corr)
        out_correlation = torch.cat([out_conv3a, out_conv3b, corr], dim=1)


        # Decoder-R
        out_conv4a = self.conv4a(out_correlation)
        out_conv5a = self.conv5a(out_conv4a)
        out_conv6a = self.conv6a(out_conv5a)
        out_conv7a = self.conv7a(out_conv6a)
        out_conv8a = self.conv8a(out_conv7a)
        R_matrix  = out_conv8a[:,0,:,:]

        # # Decoder-T
        out_conv4b = self.conv4b(out_correlation)
        out_conv5b = self.conv5b(out_conv4b)
        out_conv6b = self.conv6b(out_conv5b)
        out = torch.flatten(out_conv6b, 1)
        out_fc1b   = self.fc1b(out)
        out_fc1b   = self.relu(out_fc1b)
        T_matrix   = self.fc2b(out_fc1b)


        # print('out_conv3a: {}'.format(out_conv3a.size()))
        # print('out_conv3b: {}'.format(out_conv3b.size()))
        # print('out_correlation {}'.format(out_correlation.size()))
        # print('out_conv4a: ', out_conv4a.size())
        # print('out_conv5a: ', out_conv5a.size())
        # print('out_conv6a: ', out_conv6a.size())
        # print('out_conv7a: ', out_conv7a.size())
        # print('R-matrix: ', R_matrix.size())
        # print('T-matrix: ', T-matrix.size())

        return R_matrix, T_matrix
