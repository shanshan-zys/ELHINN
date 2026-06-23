import torch
from loki import Loki

__all__ = ['HIM']

class HIM(torch.nn.Module):
    def __init__(self,offset=10,threshold=0.2):
        super(HIM,self).__init__()
        self.lamda1 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.lamda2 = torch.nn.Parameter(torch.tensor(0.2),requires_grad=True)
        self.lamda3 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.lamda4 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.lamda5 = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.offset = offset
        self.threshold = threshold
        self.register_buffer('dx',torch.tensor([[-1,0,1],
                                                [-2,0,2],
                                                [-1,0,1]],dtype=torch.float).view(1,1,3,3))
        self.register_buffer('dy',torch.tensor([[-1,-2,-1],
                                                [0,0,0],
                                                [1,2,1]],dtype=torch.float).view(1,1,3,3))
        self.register_buffer('lap',torch.tensor([[0,1,0],
                                                 [1,-4,1],
                                                 [0,1,0]],dtype=torch.float).view(1,1,3,3))
        self.register_buffer('avg',torch.tensor([[0.125,0.125,0.125],
                                                 [0.125,0,0.125],
                                                 [0.125,0.125,0.125]],dtype=torch.float).view(1,1,3,3))
        self.register_buffer('grid',None)
        self.loki_u = Loki(in_channels=1,hidden_channels=3,out_channels=1,residual=True)
        self.loki_v = Loki(in_channels=1,hidden_channels=3,out_channels=1,residual=True)

    def convection(self,u,v):
        u_x = torch.nn.functional.conv2d(u,self.dx,padding=1)
        u_y = torch.nn.functional.conv2d(u,self.dy,padding=1)
        v_x = torch.nn.functional.conv2d(v,self.dx,padding=1)
        v_y = torch.nn.functional.conv2d(v,self.dy,padding=1)
        u_con = u*u_x+v*u_y
        v_con = u*v_x+v*v_y
        return u_con,v_con

    def viscosity(self,u,v):
        u_vis = torch.nn.functional.conv2d(u,self.lap,padding=1)
        v_vis = torch.nn.functional.conv2d(v,self.lap,padding=1)
        return u_vis,v_vis

    def alignment(self,u,v):
        u_ali = torch.nn.functional.conv2d(u,self.avg,padding=1)-u
        v_ali = torch.nn.functional.conv2d(v,self.avg,padding=1)-v
        return u_ali,v_ali

    def navigation(self,u,v):
        device = u.device
        B,_,H,W = u.shape
        if self.grid is None or self.grid.shape[0]!=B or self.grid.shape[1:3]!=(H,W):
            grid_x,grid_y = torch.meshgrid(torch.arange(W,device=device),torch.arange(H,device=device),indexing='xy')
            self.grid = torch.stack((grid_x,grid_y),dim=-1).float().unsqueeze(0).repeat(B,1,1,1)
        displacement = torch.stack([torch.sign(u.squeeze(1))*self.offset,torch.sign(v.squeeze(1))*self.offset],dim=-1)
        new_grid = self.grid+displacement
        new_grid[...,0] = 2.0*new_grid[...,0]/(W-1)-1.0
        new_grid[...,1] = 2.0*new_grid[...,1]/(H-1)-1.0
        u_next = torch.nn.functional.grid_sample(u,new_grid,mode='nearest',padding_mode='border',align_corners=True)
        v_next = torch.nn.functional.grid_sample(v,new_grid,mode='nearest',padding_mode='border',align_corners=True)
        u_nav = u_next-u
        v_nav = v_next-v
        return u_nav,v_nav

    def cohesion(self,u,v):
        device = u.device
        B,_,H,W = u.shape
        mask = torch.logical_or(u>self.threshold,v>self.threshold).float()
        boundary = torch.nn.functional.conv2d(mask,self.lap,padding=1)
        boundary = (torch.abs(boundary)>1e-6).float()
        if self.grid is None or self.grid.shape[1:3]!=(H,W):
            grid_x,grid_y = torch.meshgrid(torch.arange(W,device=device),torch.arange(H,device=device),indexing='xy')
            self.grid = torch.stack((grid_x,grid_y),dim=-1).float().unsqueeze(0).repeat(B,1,1,1)
        velocity_mag = torch.sqrt(u**2+v**2)
        new_grid_x1 = torch.stack([2.0*(self.grid[...,0]+self.offset)/(W-1)-1.0,2.0*self.grid[...,1]/(H-1)-1.0],dim=-1)
        new_grid_x2 = torch.stack([2.0*(self.grid[...,0]-self.offset)/(W-1)-1.0,2.0*self.grid[...,1]/(H-1)-1.0],dim=-1)
        new_grid_y1 = torch.stack([2.0*self.grid[...,0]/(W-1)-1.0,2.0*(self.grid[...,1]+self.offset)/(H-1)-1.0],dim=-1)
        new_grid_y2 = torch.stack([2.0*self.grid[...,0]/(W-1)-1.0,2.0*(self.grid[...,1]-self.offset)/(H-1)-1.0],dim=-1)
        velocity_x1 = torch.nn.functional.grid_sample(velocity_mag,new_grid_x1,mode='nearest',padding_mode='border',align_corners=True)
        velocity_x2 = torch.nn.functional.grid_sample(velocity_mag,new_grid_x2,mode='nearest',padding_mode='border',align_corners=True)
        velocity_y1 = torch.nn.functional.grid_sample(velocity_mag,new_grid_y1,mode='nearest',padding_mode='border',align_corners=True)
        velocity_y2 = torch.nn.functional.grid_sample(velocity_mag,new_grid_y2,mode='nearest',padding_mode='border',align_corners=True)
        u_sign = torch.where(velocity_x1>velocity_x2,1.0,-1.0)
        v_sign = torch.where(velocity_y1>velocity_y2,1.0,-1.0)
        cohesion = (velocity_mag>self.threshold).float()
        u_coh = cohesion*boundary*u_sign
        v_coh = cohesion*boundary*v_sign
        return u_coh,v_coh

    def NavierStokesEquation(self,velocity_ds):
        u = velocity_ds[:,0,:,:].unsqueeze(1)
        v = velocity_ds[:,1,:,:].unsqueeze(1)
        u_con,v_con = self.convection(u,v)
        u_vis,v_vis = self.viscosity(u,v)
        u_ali,v_ali = self.alignment(u,v)
        u_nav,v_nav = self.navigation(u,v)
        u_coh,v_coh = self.cohesion(u,v)
        u_loki = self.loki_u(u)
        v_loki = self.loki_v(v)
        u_nse = self.lamda1*u_con-self.lamda2*u_vis+self.lamda3*u_ali+self.lamda4*u_nav+self.lamda5*u_coh+u_loki
        v_nse = self.lamda1*v_con-self.lamda2*v_vis+self.lamda3*v_ali+self.lamda4*v_nav+self.lamda5*v_coh+v_loki
        velocity_nse = torch.cat([u_nse,v_nse],dim=1)
        return velocity_nse

    def CameraOffset(self,area,velocity):
        B,C,_,_ = velocity.shape
        unwalkable_area = (area==0).view(B,1,-1)
        offset = velocity.view(B,C,-1)*unwalkable_area
        offset = offset.sum(dim=2,keepdim=True)/(unwalkable_area.sum(dim=2,keepdim=True)+1e-6)
        velocity_cam = (velocity-offset.view(B,C,1,1))*area
        return velocity_cam

    def BoundaryCondition(self,area,velocity):
        u = velocity[:,0,:,:].unsqueeze(1)
        v = velocity[:,1,:,:].unsqueeze(1)
        boundary = torch.nn.functional.conv2d(area,self.lap,padding=1)
        boundary = (torch.abs(boundary)>1e-6).float()
        u_bou = -u*boundary
        v_bou = -v*boundary
        velocity_bou = torch.cat([u_bou,v_bou],dim=1)
        return velocity_bou

    def forward(self,area,velocity):
        new_height,new_width = int(velocity.shape[2]/5),int(velocity.shape[3]/5)
        grid_height,grid_width = torch.meshgrid(torch.linspace(-1,1,new_height),torch.linspace(-1,1,new_width),indexing='ij')
        grid = torch.stack((grid_width,grid_height),dim=-1).float().unsqueeze(0).repeat(velocity.shape[0],1,1,1).to(velocity.device)
        velocity_ds = torch.nn.functional.grid_sample(velocity,grid,mode='bilinear',padding_mode='border',align_corners=True)
        velocity_nse = self.NavierStokesEquation(velocity_ds)
        velocity_us = torch.nn.functional.interpolate(velocity_nse,size=(velocity.shape[2],velocity.shape[3]),mode='bicubic',align_corners=True)
        velocity_us = self.CameraOffset(area,velocity_us)
        velocity_bou = self.BoundaryCondition(area,velocity)
        return velocity_us,velocity_bou
