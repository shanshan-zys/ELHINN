import torch
from him import HIM
from convresnet import ConvResNet

__all__ = ['HINNPP']

class HINNPP(torch.nn.Module):
    def __init__(self):
        super(HINNPP,self).__init__()
        self.him = HIM()
        self.convresnet = ConvResNet(7,2)

    def forward(self,area,current_velocity):
        velocity_us,velocity_bou = self.him(area,current_velocity)
        next_velocity = self.convresnet(torch.cat([area,current_velocity,velocity_us,velocity_bou],dim=1))
        return next_velocity
