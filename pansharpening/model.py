import torch
from torch import nn



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
        self.head_conv=nn.Conv2d(5,32,3,1,1)

        self.BRRB1 = BRRes_Block(32)
        self.BRRB2 = BRRes_Block(32)
        self.BRRB3 = BRRes_Block(32)
        self.BRRB4 = BRRes_Block(32)
        self.BRRB5 = BRRes_Block(32)

        self.tail_conv=nn.Conv2d(32,4,3,1,1)

    def forward(self, pan, lms):
        x=torch.cat([pan,lms],dim=1)
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
    summary(BRResNet(),[(1,64,64),(4,64,64)],device='cpu')




