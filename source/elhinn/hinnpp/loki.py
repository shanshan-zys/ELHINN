import math
import torch

__all__ = ['Loki']

class KANLinear(torch.nn.Module):
    def __init__(self,in_features,out_features,grid_size=5,spline_order=3,scale_noise=0.1,scale_base=1.0,scale_spline=1.0,
                 enable_standalone_scale_spline=True,base_activation=torch.nn.SiLU,grid_eps=0.02,grid_range=[-1, 1],):
        super(KANLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1]-grid_range[0])/grid_size
        grid = ((torch.arange(-spline_order,grid_size+spline_order+1)*h+grid_range[0]).expand(in_features,-1).contiguous())
        self.register_buffer("grid",grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features,in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features,in_features,grid_size+spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features,in_features))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight,a=math.sqrt(5)*self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size+1,self.in_features,self.out_features)-1/2)*self.scale_noise/self.grid_size)
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0)*self.curve2coeff(
                self.grid.T[self.spline_order:-self.spline_order],noise,))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler,a=math.sqrt(5)*self.scale_spline)

    def b_splines(self,x:torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = (self.grid)
        x = x.unsqueeze(-1)
        bases = ((x>=grid[:,:-1])&(x<grid[:,1:])).to(x.dtype)
        for k in range(1,self.spline_order+1):
            bases = ((x-grid[:,:-(k+1)])/(grid[:,k:-1]-grid[:,:-(k+1)])*bases[:,:,:-1])+((grid[:,k+1:]-x)/(grid[:,k+1:]-grid[:,1:(-k)])*bases[:,:,1:])
        assert bases.size()==(x.size(0),self.in_features,self.grid_size+self.spline_order,)
        return bases.contiguous()

    def curve2coeff(self,x:torch.Tensor,y:torch.Tensor):
        assert x.dim()==2 and x.size(1)==self.in_features
        assert y.size()==(x.size(0),self.in_features,self.out_features)
        A = self.b_splines(x).transpose(0,1)
        B = y.transpose(0,1)
        solution = torch.linalg.lstsq(A,B).solution
        result = solution.permute(2,0,1)
        assert result.size()==(self.out_features,self.in_features,self.grid_size+self.spline_order,)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight*(self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self,x:torch.Tensor):
        assert x.dim()==2 and x.size(1)==self.in_features
        base_output = torch.nn.functional.linear(self.base_activation(x),self.base_weight)
        spline_output = torch.nn.functional.linear(self.b_splines(x).view(x.size(0),-1),self.scaled_spline_weight.view(self.out_features, -1),)
        return base_output+spline_output

    @torch.no_grad()
    def update_grid(self,x:torch.Tensor,margin=0.01):
        assert x.dim()==2 and x.size(1)==self.in_features
        batch = x.size(0)
        splines = self.b_splines(x)
        splines = splines.permute(1,0,2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1,2,0)
        unreduced_spline_output = torch.bmm(splines,orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1,0,2)
        x_sorted = torch.sort(x,dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0,batch-1,self.grid_size+1,dtype=torch.int64,device=x.device)]
        uniform_step = (x_sorted[-1]-x_sorted[0]+2*margin)/self.grid_size
        grid_uniform = (torch.arange(self.grid_size+1,dtype=torch.float32,device=x.device).unsqueeze(1)*uniform_step+x_sorted[0]-margin)
        grid = self.grid_eps*grid_uniform+(1-self.grid_eps)*grid_adaptive
        grid = torch.concatenate([grid[:1]-uniform_step*torch.arange(self.spline_order,0,-1,device=x.device).unsqueeze(1),grid,grid[-1:]
                                  +uniform_step*torch.arange(1,self.spline_order+1,device=x.device).unsqueeze(1),],dim=0,)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x,unreduced_spline_output))

    def regularization_loss(self,regularize_activation=1.0,regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake/regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p*p.log())
        return (regularize_activation*regularization_loss_activation+regularize_entropy*regularization_loss_entropy)

class KAN(torch.nn.Module):
    def __init__(self,layers_hidden,grid_size=5,spline_order=3,scale_noise=0.1,scale_base=1.0,scale_spline=1.0,
                 base_activation=torch.nn.SiLU,grid_eps=0.02,grid_range=[-1,1],):
        super(KAN,self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features,out_features in zip(layers_hidden,layers_hidden[1:]):
            self.layers.append(KANLinear(in_features,out_features,grid_size=grid_size,spline_order=spline_order,
                                         scale_noise=scale_noise,scale_base=scale_base,scale_spline=scale_spline,
                                         base_activation=base_activation,grid_eps=grid_eps,grid_range=grid_range,))

    def forward(self,x:torch.Tensor,update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self,regularize_activation=1.0,regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation,regularize_entropy) for layer in self.layers)

class Loki(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,residual=True):
        super(Loki,self).__init__()
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.proj1 = torch.nn.Linear(in_channels,hidden_channels)
        self.kan = KAN([hidden_channels,2*hidden_channels+1,hidden_channels])
        self.proj2 = torch.nn.Linear(hidden_channels,out_channels)

    def forward(self,input):
        p1 = self.proj1(input.permute(0,2,3,1))
        k = self.kan(p1.reshape(-1,self.hidden_channels))
        p2 = self.proj2(k.reshape(input.shape[0],input.shape[2],input.shape[3],-1)).permute(0,3,1,2)
        output = (input+p2) if self.residual else p2
        return output
