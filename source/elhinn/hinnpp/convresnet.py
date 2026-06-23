import torch

__all__ = ['ConvResNet']

class ConvResBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,height=90,width=120):
        super(ConvResBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels,out_channels,7,1,3)
        self.activation = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,7,1,3)
        self.norm = torch.nn.LayerNorm([in_channels]+[height,width])

    def forward(self,input):
        residual = self.norm(self.conv2(self.activation(self.conv1(input))))
        output = residual+input
        return output

class ConvResNet(torch.nn.Module):
    def __init__(self,in_channels,out_channels,height=360,width=480):
        super(ConvResNet,self).__init__()
        self.cw = torch.nn.Parameter(torch.randn(1,2,1,width))
        self.rw = torch.nn.Parameter(torch.randn(1,2,height,1))
        self.conv1 = torch.nn.Conv2d(in_channels+2,24,6,2,2)
        self.activation1 = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(24,96,6,2,2)
        self.activation2 = torch.nn.Tanh()
        self.convresblocks = torch.nn.Sequential(ConvResBlock(96,96),
                                                 ConvResBlock(96,96),
                                                 ConvResBlock(96,96),
                                                 ConvResBlock(96,96))
        self.pixelshuffle = torch.nn.PixelShuffle(4)
        self.conv = torch.nn.Conv2d(6,out_channels,5,1,2)

    def forward(self,input):
        input = torch.cat([input,(self.rw@self.cw).repeat(input.shape[0],1,1,1)],dim=1)
        output = self.activation2(self.conv2(self.activation1(self.conv1(input))))
        output = self.conv(self.pixelshuffle(self.convresblocks(output)))
        return output
