import torch

def normalize_location(location,height=360,width=480):
    location_norm = location.clone()
    location_norm[...,0] = location[...,0]/(height-1)*2-1
    location_norm[...,1] = location[...,1]/(width-1)*2-1
    return location_norm

def denormalize_location(location_norm,height=360,width=480):
    location = location_norm.clone()
    location[...,0] = (location_norm[...,0]+1)/2*(height-1)
    location[...,1] = (location_norm[...,1]+1)/2*(width-1)
    return location

def normalize_velocity(velocity,height=360,width=480):
    velocity_norm = velocity.clone()
    velocity_norm[...,0] = velocity[...,0]/(height-1)*2
    velocity_norm[...,1] = velocity[...,1]/(width-1)*2
    return velocity_norm

def denormalize_velocity(velocity_norm,height=360,width=480):
    velocity = velocity_norm.clone()
    velocity[...,0] = velocity_norm[...,0]/2*(height-1)
    velocity[...,1] = velocity_norm[...,1]/2*(width-1)
    return velocity

def sample_random(area,N):
    device = area.device
    walkable = area.nonzero(as_tuple=False)
    indices = torch.randperm(walkable.shape[0],device=device)[:N]
    return walkable[indices].float()

def sample_dynamic(area,velocity,N,threshold=0.2):
    device = area.device
    velocity_mag = torch.sqrt(velocity[0]**2+velocity[1]**2)
    walkable = (area>0)&(velocity_mag>threshold)
    positions = walkable.nonzero(as_tuple=False)
    if positions.shape[0]<N:
        return sample_random(area,N)
    indices = torch.randperm(positions.shape[0],device=device)[:N]
    return positions[indices].float()

def sample(area,velocity,N,threshold=0.2,ratio=0.7):
    N_dynamic = int(N*ratio)
    N_random = N-N_dynamic
    dynamic = sample_dynamic(area,velocity,N_dynamic,threshold)
    random_samples = sample_random(area,N_random)
    return torch.cat([dynamic,random_samples],dim=0)

def find_boundary(area):
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float,device=area.device).view(1,1,3,3)
    area_4d = area.unsqueeze(0).unsqueeze(0) if area.dim()==2 else area.unsqueeze(0)
    boundary = torch.nn.functional.conv2d(area_4d,lap,padding=1)
    boundary = (torch.abs(boundary.squeeze())>1e-6).float()
    return boundary

def compute_normals(area):
    device = area.device
    dx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float,device=device).view(1,1,3,3)
    dy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float,device=device).view(1,1,3,3)
    area_4d = area.unsqueeze(0).unsqueeze(0) if area.dim()==2 else area.unsqueeze(0)
    grad_x = torch.nn.functional.conv2d(area_4d,dx,padding=1).squeeze()
    grad_y = torch.nn.functional.conv2d(area_4d,dy,padding=1).squeeze()
    mag = torch.sqrt(grad_x**2+grad_y**2)+1e-8
    normal_x = grad_x/mag
    normal_y = grad_y/mag
    return normal_x,normal_y

def compute_density_map(area,location,H,W,radius=10):
    device = area.device
    density = torch.zeros(H,W,device=device)
    x = location[:,1].round().long().clamp(0,W-1)
    y = location[:,0].round().long().clamp(0,H-1)
    indices = y*W+x
    density_flat = density.view(-1)
    density_flat.scatter_add_(0,indices,torch.ones_like(indices,dtype=torch.float))
    density = density_flat.view(H,W)
    kernel_size = 2*radius+1
    kernel = torch.ones(1,1,kernel_size,kernel_size,device=device)/(kernel_size**2)
    density = torch.nn.functional.conv2d(density.unsqueeze(0).unsqueeze(0),kernel,padding=radius).squeeze()
    return density

def resample(area,velocity,location,indices,radius=10):
    device = area.device
    H,W = area.shape[-2:]
    boundary = find_boundary(area)
    boundary_positions = boundary.nonzero(as_tuple=False).float()
    if boundary_positions.shape[0]==0:
        return location[indices]
    normal_x,normal_y = compute_normals(area)
    bx = boundary_positions[:,1].long()
    by = boundary_positions[:,0].long()
    velocity_u = velocity[0] if velocity.dim()==3 else velocity[0,0]
    velocity_v = velocity[1] if velocity.dim()==3 else velocity[0,1]
    speed = torch.sqrt(velocity_u[by,bx]**2+velocity_v[by,bx]**2)
    nx = normal_x[by,bx]
    ny = normal_y[by,bx]
    vx = velocity_u[by,bx]
    vy = velocity_v[by,bx]
    inflow = (vx*nx+vy*ny)
    inflow_mask = inflow>0
    if not inflow_mask.any():
        rand_idx = torch.randperm(boundary_positions.shape[0],device=device)[:indices.shape[0]]
        return boundary_positions[rand_idx]
    inflow_positions = boundary_positions[inflow_mask]
    inflow_speed = speed[inflow_mask]
    density_map = compute_density_map(area,location,H,W,radius)
    dx = inflow_positions[:,1].long()
    dy = inflow_positions[:,0].long()
    density_at_boundary = density_map[dy,dx]
    weights = inflow_speed/(density_at_boundary+1e-6)
    weights = weights/weights.sum()
    sampled_idx = torch.multinomial(weights,indices.shape[0],replacement=True)
    return inflow_positions[sampled_idx]

def dead_zone_jitter(area,velocity,location,threshold=0.1,jitter_scale=0.5):
    velocity_mag = torch.sqrt(velocity[0]**2+velocity[1]**2) if velocity.dim()==3 else torch.sqrt(velocity[0,0]**2+velocity[0,1]**2)
    x = location[:,1].round().long().clamp(0,velocity_mag.shape[-1]-1)
    y = location[:,0].round().long().clamp(0,velocity_mag.shape[-2]-1)
    speed = velocity_mag[y,x]
    stagnant = speed<threshold
    if stagnant.any():
        jitter = torch.randn_like(location[stagnant])*jitter_scale
        location[stagnant] = location[stagnant]+jitter
    return location
