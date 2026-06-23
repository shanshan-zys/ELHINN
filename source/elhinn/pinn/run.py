import os
import json
import time
import numpy
import torch
from pinn import PINN
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def collate_with_padding(batch):
    filenames,areas,velocities,locations = zip(*batch)
    lengths = [loc.shape[0] for loc in locations]
    max_length = max(lengths)
    padded_locations = pad_sequence(locations,batch_first=True)
    padded_lengths = torch.tensor(lengths,dtype=torch.long)
    mask = torch.arange(max_length).unsqueeze(0)<padded_lengths.unsqueeze(1)
    return filenames,torch.stack(areas),torch.stack(velocities),padded_locations,padded_lengths,mask

def collision_avoidance(destination,mask):
    threshold = 20
    B,N,_ = destination.shape
    device = destination.device
    diagonal = torch.eye(N,dtype=torch.bool,device=device).unsqueeze(0).expand(B,-1,-1)
    distance = destination.unsqueeze(2)-destination.unsqueeze(1)
    distance = torch.norm(distance,dim=-1)+1e-8
    collision = (distance<threshold)&(~diagonal)&mask.unsqueeze(1)&mask.unsqueeze(2)
    penalty = (threshold-distance).clamp(min=0.0)**2
    loss = (penalty*collision.float()).sum(dim=(1,2))/(collision.sum(dim=(1,2)).float()+1e-8)
    loss = torch.where(collision.sum(dim=(1,2))>0,loss,torch.zeros_like(loss))
    return loss.mean()

def velocity_difference(velocity,source,destination,mask):
    difference = destination-source
    loss = torch.nn.functional.smooth_l1_loss(difference[mask],velocity[mask])
    return loss

def loss_function(velocity,source,destination,mask):
    loss_col = 0.001*collision_avoidance(destination,mask)
    loss_vel = velocity_difference(velocity,source,destination,mask)
    loss = loss_col+loss_vel
    return loss

class DataList(torch.utils.data.Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.filelist = sorted([file for file in os.listdir(folder) if file.endswith('.pt')])

    def __len__(self):
        return len(self.filelist)

    def __erosion__(self,area,kernel_size=9):
        area = 1-torch.nn.functional.max_pool2d(1-area.unsqueeze(0),kernel_size,1,(kernel_size-1)//2)
        return area.squeeze(0)

    def __getitem__(self,idx):
        file = os.path.join(self.folder,self.filelist[idx])
        velocity = torch.load(file,weights_only=False)
        velocity = torch.tensor(velocity,dtype=torch.float)
        area = torch.load(f"{file.replace('heatmap','area')[:-3]}.pt",weights_only=False)
        area = torch.tensor(area,dtype=torch.float)
        location = torch.load(file.replace('heatmap','initial/position'),weights_only=False)
        location = torch.tensor(location,dtype=torch.float)
        initvelocity = torch.load(file.replace('heatmap','initial/velocity'),weights_only=False)
        initvelocity = torch.tensor(initvelocity,dtype=torch.float)
        x = location[:,0].round().long().clamp(0,width-1)
        y = location[:,1].round().long().clamp(0,height-1)
        valid = area[y,x]>0
        location = location[valid]
        location = location[:,[1,0]]
        initvelocity = initvelocity[valid]
        return self.filelist[idx],area,velocity,location

def save_trajectory(filename,trajectory,consistency):
    trajectory_pt = torch.zeros((frame,height,width),device=device)
    for t in range(frame):
        x = trajectory[t,:,1].round().long().clamp(0,width-1)
        y = trajectory[t,:,0].round().long().clamp(0,height-1)
        trajectory_pt[t,y,x] = 1
    torch.save(trajectory.detach().cpu().numpy(),os.path.join(ROOT,'output','points',f'{filename}.pt'))
    trajectory_json = {}
    id = 1
    for idx in range(trajectory.shape[1]):
        trajectory_single = []
        for t in range(frame):
            if consistency[t,idx]:
                x = trajectory[t,idx,1].round().long().clamp(0,width-1)
                y = trajectory[t,idx,0].round().long().clamp(0,height-1)
                trajectory_single.append({'frame':t+1,'x':float(x),'y':float(y)})
            else:
                if trajectory_single:
                    trajectory_json[f'id_{id}'] = trajectory_single
                    id += 1
                    trajectory_single = []
        if trajectory_single:
            trajectory_json[f'id_{id}'] = trajectory_single
            id += 1
    with open(os.path.join(ROOT,'output','points',f'{filename}.json'),'w') as file:
        json.dump(trajectory_json,file,indent=2)

def trajectory_visualization(filename,trajectory,velocity=None,frame=101,height=360,width=480,stride=20):
    fig,ax = plt.subplots(figsize=(8,6))
    grid_y,grid_x = numpy.meshgrid(numpy.arange(0,height,stride),numpy.arange(0,width,stride),indexing='ij')
    def update(timestep,velocity):
        ax.clear()
        x = trajectory[timestep,:,1].detach().cpu().numpy()
        y = trajectory[timestep,:,0].detach().cpu().numpy()
        ax.scatter(x,y,c='black',s=2,label='Trajectory')
        if velocity is not None:
            u = velocity[timestep,0,::stride,::stride].detach().cpu().numpy()
            v = velocity[timestep,1,::stride,::stride].detach().cpu().numpy()
            velocity_mag = numpy.hypot(u,v)
            ax.quiver(grid_x,grid_y,u,v,velocity_mag,scale=numpy.max(velocity_mag)/stride,scale_units='xy',
                      angles='xy',width=0.002,cmap='cool',clim=(0,numpy.max(velocity_mag)))
        ax.set_title(f'Timestep:{timestep}/{frame-1}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_xlim([0,width])
        ax.set_ylim([height,0])
        ax.legend(loc='upper right')
    gif = FuncAnimation(fig,update,frames=frame-1,fargs=(velocity,),interval=200)
    gif.save(os.path.join(ROOT,'output','points',f'{filename}.gif'),writer='pillow',fps=5)
    plt.close()
    with open(os.path.join(ROOT,'output','points',f'{filename}.json'),'r') as file:
        trajectory_json = json.load(file)
    plt.figure(figsize=(8,6))
    colors = plt.get_cmap('hsv',len(trajectory_json)+1)
    for idx,(_,points) in enumerate(trajectory_json.items()):
        xs = [point['x'] for point in points]
        ys = [point['y'] for point in points]
        plt.plot(xs,ys,marker='o',markersize=2,linewidth=1,color=colors(idx))
    plt.title('Trajectory')
    plt.xlim(0,width)
    plt.ylim(height,0)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig(os.path.join(ROOT,'output','points',f'{filename}.png'))
    plt.close()

def train(epochs,steps):
    data_list = DataList(folder=os.path.join(ROOT,'data','mot20','heatmap'))
    train_split,_ = train_test_split(data_list.filelist,train_size=0.8,test_size=0.2,random_state=42)
    train_split = sorted(train_split)
    train_list = DataList(folder=os.path.join(ROOT,'data','mot20','heatmap'))
    train_list.filelist = train_split
    train_loader = torch.utils.data.DataLoader(train_list,batch_size=16,shuffle=True,collate_fn=collate_with_padding)
    print('Train List:',train_split)
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    model.to(device)
    checkpoint = os.path.join(ROOT,'output','checkpoint',f'pinn_mot20_{epochs}.pth')
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(parameter['model_state_dict'],strict=False)
        optimizer.load_state_dict(parameter['optimizer_state_dict'])
        print('Checkpoint loaded!')
    start_time = time.time()
    for epoch in range(steps):
        model.train()
        print(f'Iter {epoch+epochs}:')
        epoch_loss = 0
        for batch_idx,(_,area,velocity,location,_,mask) in enumerate(train_loader):
            area,velocity,location,mask = area.to(device),velocity.to(device),location.to(device),mask.to(device)
            source = location
            batch_loss = 0
            optimizer.zero_grad()
            for i in range(frame-1):
                grid = torch.stack([source[:,:,1]/(width-1)*2-1,source[:,:,0]/(height-1)*2-1],dim=-1).unsqueeze(2)
                velocity_sam = torch.nn.functional.grid_sample(velocity[:,i],grid,mode='bilinear',padding_mode='border',align_corners=True)
                velocity_sam = velocity_sam.squeeze(3).permute(0,2,1)
                velocity_sam = velocity_sam[...,[1,0]]
                destination,consistency_single = model(area,velocity[:,i],velocity_sam,source,mask)
                if i>0:
                    loss = loss_function(velocity_sam,source,destination,consistency_single&mask)
                    batch_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()
                with torch.no_grad():
                    source = destination.detach()
            epoch_loss += batch_loss
            print(f'Batch {batch_idx},loss:{batch_loss}')
        avg_train_loss = epoch_loss/len(train_loader)
        print(f'Iter {epoch+epochs},training loss:{avg_train_loss}\n')
    elapsed = time.time()-start_time
    print(f'Training time:{elapsed}\n')
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},
               os.path.join(ROOT,'output','checkpoint',f'pinn_mot20_{epochs+steps}.pth'))

def test(epochs):
    data_list = DataList(folder=os.path.join(ROOT,'data','mot20','heatmap'))
    _,test_split = train_test_split(data_list.filelist,train_size=0.8,test_size=0.2,random_state=42)
    test_split = sorted(test_split)
    test_list = DataList(folder=os.path.join(ROOT,'data','mot20','heatmap'))
    test_list.filelist = test_split
    test_loader = torch.utils.data.DataLoader(test_list,batch_size=1,shuffle=False,collate_fn=collate_with_padding)
    print('Test List:',test_split)
    model = PINN()
    model.to(device)
    checkpoint = os.path.join(ROOT,'output','checkpoint',f'pinn_mot20_{epochs}.pth')
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(parameter['model_state_dict'],strict=False)
        print('Checkpoint loaded!')
    with torch.no_grad():
        model.eval()
        for _,(filename,area,velocity,location,_,mask) in enumerate(test_loader):
            filename = filename[0].replace('.pt','')
            area,velocity,location,mask = area.to(device),velocity.to(device),location.to(device),mask.to(device)
            source = location
            trajectory = [source.unsqueeze(1)]
            consistency = [torch.ones(area.shape[0],1,location.shape[1],dtype=torch.bool,device=device)&mask.unsqueeze(1)]
            for i in range(frame-1):
                grid = torch.stack([source[:,:,1]/(width-1)*2-1,source[:,:,0]/(height-1)*2-1],dim=-1).unsqueeze(2)
                velocity_sam = torch.nn.functional.grid_sample(velocity[:,i],grid,mode='bilinear',padding_mode='border',align_corners=True)
                velocity_sam = velocity_sam.squeeze(3).permute(0,2,1)
                velocity_sam = velocity_sam[...,[1,0]]
                destination,consistency_single = model(area,velocity[:,i],velocity_sam,source,mask)
                trajectory.append(destination.unsqueeze(1))
                consistency.append(consistency_single.unsqueeze(1))
                source = destination
            trajectory = torch.cat(trajectory,dim=1)
            consistency = torch.cat(consistency,dim=1)
            valid_idx = mask[0].nonzero(as_tuple=False).squeeze(1)
            save_trajectory(filename,trajectory.squeeze(0)[:,valid_idx],consistency.squeeze(0)[:,valid_idx])
            trajectory_visualization(filename,trajectory.squeeze(0)[:,valid_idx],velocity.squeeze(0))

if __name__== "__main__":
    global frame,height,width,device
    frame,height,width = 100,360,480
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(ROOT,'output','checkpoint'),exist_ok=True)
    os.makedirs(os.path.join(ROOT,'output','points'),exist_ok=True)
    epochs,steps_list = 0,[50]
    for step in steps_list:
        train(epochs,step)
        epochs += step
        test(epochs)
