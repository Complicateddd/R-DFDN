
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class resblock(nn.Module):

    def __init__(self, depth, channels, stride=1, bn='', nresblocks=1.,affine=True, kernel_size=3, bias=True):
        self.depth = depth
        self. channels = channels
        
        super(resblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(depth,affine=affine) if bn else nn.Sequential()
        self.conv2 = (nn.Conv2d(depth, channels, kernel_size=kernel_size, stride=stride, padding=1, bias=bias))
        self.bn2 = nn.BatchNorm2d(channels, affine=affine) if bn else nn.Sequential()

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)

        self.shortcut = nn.Sequential()
        if stride > 1 or depth!=channels:
            layers = []
            conv_layer = nn.Conv2d(depth, channels, kernel_size=1, stride=stride, padding=0, bias=bias)
            layers += [conv_layer, nn.BatchNorm2d(channels,affine=affine) if bn else nn.Sequential()]
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        out = ACT(self.bn1(x))
        out = ACT(self.bn2(self.conv2(out)))
        out = (self.conv3(out))
        short = self.shortcut(x)
        out += 1.*short
        return out


class ResNet(nn.Module):
    def __init__(self, depth=56, nb_filters=32, num_classes=10, bn=False, affine=True, kernel_size=3, inp_channels=3, k=1, bias=False, inp_noise=0): # n=9->Resnet-56
        super(ResNet, self).__init__()
        self.inp_noise = inp_noise
        nstage = 3 
        
        self.pre_clf=[]

        assert ((depth-2)%6 ==0), 'resnet depth should be 6n+2'
        n = int((depth-2)/6)
        
        nfilters = [nb_filters, nb_filters*k, 2* nb_filters*k, 4* nb_filters*k, num_classes]
        self.nfilters = nfilters
        self.num_classes = num_classes
        self.conv1 = (nn.Conv2d(inp_channels, nfilters[0], kernel_size=kernel_size, stride=1, padding=0, bias=bias))
        self.bn1 = nn.BatchNorm2d(nfilters[0], affine=affine) if bn else nn.Sequential()
        nb_filters_prev = nb_filters_cur = nfilters[0]
        for stage in range(nstage):
            nb_filters_cur =  nfilters[stage+1]
            for i in range(n):
                subsample = 1 if (i > 0 or stage == 0) else 2
                layer = resblock(nb_filters_prev, nb_filters_cur, subsample, bn=bn, nresblocks = nstage*n, affine=affine, kernel_size=3, bias=bias)
                self.pre_clf.append(layer)
                nb_filters_prev = nb_filters_cur

        self.pre_clf_1 = nn.Sequential(*self.pre_clf[:n])
        self.pre_clf_2 = nn.Sequential(*self.pre_clf[n:2*n])
        self.pre_clf_3 = nn.Sequential(*self.pre_clf[2*n:])
    def forward(self, x, ret_hid=False, train=True):
        if x.size()[1]==1: # if MNIST is given, replicate 1 channel to make input have 3 channel
            out = torch.ones(x.size(0), 3, x.size(2), x.size(3)).type('torch.cuda.FloatTensor')
            out = out*x
        else:
            out = x
        if self.inp_noise>0 and train:
            out = out + self.inp_noise*torch.randn_like(out)
        hid = self.conv1(out)
        out = self.bn1(hid)
        out1 = self.pre_clf_1(out)
        out2 = self.pre_clf_2(out1)
        class_feature = self.pre_clf_3(out2)
        return hid,class_feature

if __name__ == '__main__':
    pass