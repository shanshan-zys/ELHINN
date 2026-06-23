import torch
import torchvision

__all__ = ['Inception','FIDInception']

class Inception(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Inception,self).__init__()
        self.baseline = torchvision.models.inception.inception_v3(weights=None,init_weights=True)
        self.baseline.Conv2d_1a_3x3.conv = torch.nn.Conv2d(in_channels,32,3,2,1,bias=False)
        self.baseline.fc = torch.nn.Linear(2048,out_channels)

    def forward(self,input):
        return self.baseline(input)

class InceptionA(torchvision.models.inception.InceptionA):
    def __init__(self,in_channels,pool_features):
        super(InceptionA,self).__init__(in_channels,pool_features)

    def forward(self,input):
        branch1x1 = self.branch1x1(input)
        branch5x5 = self.branch5x5_1(input)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(input)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = torch.nn.functional.avg_pool2d(input,3,1,1,count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch5x5,branch3x3dbl,branch_pool]
        return torch.cat(outputs,1)

class InceptionC(torchvision.models.inception.InceptionC):
    def __init__(self,in_channels,channels_7x7):
        super(InceptionC,self).__init__(in_channels,channels_7x7)

    def forward(self,input):
        branch1x1 = self.branch1x1(input)
        branch7x7 = self.branch7x7_1(input)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(input)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = torch.nn.functional.avg_pool2d(input,3,1,1,count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch7x7,branch7x7dbl,branch_pool]
        return torch.cat(outputs,1)

class InceptionE1(torchvision.models.inception.InceptionE):
    def __init__(self,in_channels):
        super(InceptionE1,self).__init__(in_channels)

    def forward(self,input):
        branch1x1 = self.branch1x1(input)
        branch3x3 = self.branch3x3_1(input)
        branch3x3 = [self.branch3x3_2a(branch3x3),self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3,1)
        branch3x3dbl = self.branch3x3dbl_1(input)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl),self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl,1)
        branch_pool = torch.nn.functional.avg_pool2d(input,3,1,1,count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch3x3,branch3x3dbl,branch_pool]
        return torch.cat(outputs,1)

class InceptionE2(torchvision.models.inception.InceptionE):
    def __init__(self,in_channels):
        super(InceptionE2,self).__init__(in_channels)

    def forward(self,input):
        branch1x1 = self.branch1x1(input)
        branch3x3 = self.branch3x3_1(input)
        branch3x3 = [self.branch3x3_2a(branch3x3),self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3,1)
        branch3x3dbl = self.branch3x3dbl_1(input)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl),self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl,1)
        branch_pool = torch.nn.functional.max_pool2d(input,3,1,1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch3x3,branch3x3dbl,branch_pool]
        return torch.cat(outputs,1)

class FIDInception(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FIDInception,self).__init__()
        self.baseline = torchvision.models.inception.inception_v3(weights=None,init_weights=True)
        self.baseline.Conv2d_1a_3x3.conv = torch.nn.Conv2d(in_channels,32,3,2,1,bias=False)
        self.baseline.Mixed_5b = InceptionA(192,32)
        self.baseline.Mixed_5c = InceptionA(256,64)
        self.baseline.Mixed_5d = InceptionA(288,64)
        self.baseline.Mixed_6b = InceptionC(768,128)
        self.baseline.Mixed_6c = InceptionC(768,160)
        self.baseline.Mixed_6d = InceptionC(768,160)
        self.baseline.Mixed_6e = InceptionC(768,192)
        self.baseline.Mixed_7b = InceptionE1(1280)
        self.baseline.Mixed_7c = InceptionE2(2048)
        self.baseline.fc = torch.nn.Linear(2048,out_channels)

    def forward(self,input):
        return self.baseline(input)
