import os
import sys
import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))
sys.path.insert(0,os.path.join(ROOT,'source','elhinn','hinnpp'))
from hinnpp import HINNPP

class DataList(torch.utils.data.Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.filelist = sorted([file for file in os.listdir(folder) if file.endswith('.pt')])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        file = os.path.join(self.folder,self.filelist[idx])
        velocity = torch.load(file,weights_only=False)
        velocity = torch.tensor(velocity,dtype=torch.float)
        border_max = torch.abs(velocity).max().item()
        border_quantile = numpy.quantile(torch.abs(velocity).numpy(),0.997).item()
        ratio = (border_max-border_quantile)/border_max
        border = border_quantile*1.1 if ratio>0.05 else border_max*0.95
        velocity = torch.clamp(velocity,-border,border)/border
        area = torch.load(f"{file.replace('velocity','area')[:-4]}.pt",weights_only=False)
        area = torch.tensor(area,dtype=torch.float).unsqueeze(0)
        return self.filelist[idx],area,velocity

def velocity_visualization(category,filename,groundtruth,prediction):
    gap = (frame//9)-1
    groundtruth = groundtruth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    _,axes = plt.subplots(nrows=4,ncols=10,figsize=(36,12))
    for channel in range(2):
        for timestep in range(10):
            axes[channel*2,timestep].imshow(
                groundtruth[gap*timestep,channel,:,:],vmin=-1,vmax=1,cmap='rainbow',origin='upper')
            axes[channel*2+1,timestep].imshow(
                prediction[gap*timestep,channel,:,:],vmin=-1,vmax=1,cmap='rainbow',origin='upper')
    for ax in axes.flatten():
        ax.axis('on')
        ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT,'data','dcfd','heatmap',category,f'{filename}.png'))
    plt.close()
    fig,((r0c0,r0c1),(r1c0,r1c1)) = plt.subplots(nrows=2,ncols=2)
    im00 = r0c0.imshow(groundtruth[0,0],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    im01 = r0c1.imshow(groundtruth[0,1],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    im10 = r1c0.imshow(prediction[0,0],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    im11 = r1c1.imshow(prediction[0,1],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    plt.tight_layout()
    def update(timestep,groundtruth,prediction):
        im00.set_data(groundtruth[timestep,0])
        im01.set_data(groundtruth[timestep,1])
        im10.set_data(prediction[timestep,0])
        im11.set_data(prediction[timestep,1])
        return [im00,im01,im10,im11]
    gif = FuncAnimation(fig,update,frames=frame-1,fargs=(groundtruth,prediction),interval=41)
    gif.save(os.path.join(ROOT,'data','dcfd','heatmap',category,f'{filename}.gif'),writer='pillow',fps=24)
    plt.close()

def generate_heatmap(category):
    data_list = DataList(folder=os.path.join(ROOT,'data','dcfd','velocity',category))
    data_loader = torch.utils.data.DataLoader(data_list,batch_size=1,shuffle=False)
    print('Data List:',data_list.filelist)
    model = HINNPP()
    model.to(device)
    checkpoint = os.path.join(ROOT,'output','checkpoint',f'hinnpp_{category}.pth')
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(parameter['model_state_dict'],strict=False)
        print('Checkpoint loaded!')
    with torch.no_grad():
        model.eval()
        for _,(filename,area,groundtruth) in enumerate(data_loader):
            filename = filename[0].replace('.pt','')
            print(f'Generating heatmap: {filename}')
            area = area.to(device)
            groundtruth = groundtruth.squeeze(0).to(device)
            prediction = torch.zeros_like(groundtruth)
            input = groundtruth[0,:,:,:].unsqueeze(0)
            prediction[0,:,:,:] = groundtruth[0,:,:,:]
            for i in range(frame-1):
                prediction[i+1,:,:,:] = model(area,input).squeeze(0)*area
                with torch.no_grad():
                    input = prediction[i+1,:,:,:].unsqueeze(0).detach()
            border = torch.abs(prediction[1:]).max().item()
            prediction[1:] = torch.clamp(prediction[1:],-border,border)/border
            torch.save(prediction.detach().cpu().numpy(),os.path.join(ROOT,'data','dcfd','heatmap',category,f'{filename}.pt'))
            velocity_visualization(category,filename,groundtruth,prediction)
        print('')

if __name__=="__main__":
    global frame,height,width,device
    frame,height,width = 25,360,480
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vel_dir = os.path.join(ROOT,'data','dcfd','velocity')
    folderlist = sorted(os.listdir(vel_dir))
    for category in folderlist:
        heatmap_dir = os.path.join(ROOT,'data','dcfd','heatmap',category)
        os.makedirs(heatmap_dir,exist_ok=True)
        generate_heatmap(category)
