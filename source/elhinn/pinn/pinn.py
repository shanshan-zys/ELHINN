import torch
from utils import normalize_location,denormalize_location,resample

__all__ = ['PINN']

class PINN(torch.nn.Module):
    def __init__(self,in_channels=4,hidden_channels=64,out_channels=2):
        super(PINN,self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels,hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels,hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels,out_channels)
        )

    def forward(self,area,velocity,velocity_sam,source,mask):
        B,N,_ = source.shape
        H,W = area.shape[-2:]
        consistency = torch.ones(B,N,dtype=torch.bool,device=area.device)
        source_norm = normalize_location(source)
        displacement = torch.zeros_like(source_norm)
        for b in range(B):
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel()>0:
                inputs = torch.cat([source_norm[b,valid_idx],velocity_sam[b,valid_idx]],dim=1)
                disp = self.mlp(inputs)
                displacement[b,valid_idx] = disp
        source_denorm = denormalize_location(source_norm)
        destination = source_denorm+displacement
        x = destination[:,:,1].round().long().clamp(0,W-1)
        y = destination[:,:,0].round().long().clamp(0,H-1)
        valid = torch.zeros_like(mask,dtype=torch.bool)
        for b in range(B):
            valid[b,:] = (area[b,y[b],x[b]]>0)&mask[b]
        for b in range(B):
            invalid_idx = (~valid[b]).nonzero(as_tuple=False).squeeze(1)
            if invalid_idx.numel()>0:
                destination[b,invalid_idx] = resample(area[b],velocity[b],source[b].detach(),invalid_idx,radius=10)
                valid[b,invalid_idx] = True
                consistency[b,invalid_idx] = False
        return destination,consistency&valid
