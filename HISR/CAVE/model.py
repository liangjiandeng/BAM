import torch
import torch.nn as nn
from torch.nn import functional as F


class BRRes_Block(nn.Module):
    def __init__(self,channels):
        super(BRRes_Block, self).__init__()
        self.conv1=nn.Conv2d(channels,channels,3,1,1)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(channels,channels//2,3,1,1)
        self.conv3=nn.Conv2d(channels,channels//2,3,1,1)

    def forward(self, x):
        res=self.conv1(x)
        res1=self.relu(res)
        res2=self.relu(-res)
        res1=self.conv2(res1)
        res2=self.conv3(res2)
        res=torch.cat([res1,res2],dim=1)
        x=res+x
        return x


class BRResNet(nn.Module):
    def __init__(self):
        super(BRResNet, self).__init__()
        self.ups=nn.UpsamplingBilinear2d(scale_factor=4)
        self.head_conv=nn.Conv2d(34,64,3,1,1)

        self.BRRB1 = BRRes_Block(64)
        self.BRRB2 = BRRes_Block(64)
        self.BRRB3 = BRRes_Block(64)
        self.BRRB4 = BRRes_Block(64)
        self.BRRB5 = BRRes_Block(64)

        self.tail_conv=nn.Conv2d(64,31,3,1,1)

    def forward(self, rgb, ms):
        lms=self.ups(ms)
        x=torch.cat([rgb,lms],dim=1)
        x=self.head_conv(x)
        x = self.BRRB1(x)
        x = self.BRRB2(x)
        x = self.BRRB3(x)
        x = self.BRRB4(x)
        x = self.BRRB5(x)

        x=self.tail_conv(x)
        sr=x+lms

        return sr

if __name__ == '__main__':
    from torchsummary import summary
    summary(BRResNet(),[(3,64,64),(31,16,16)],device='cpu')


