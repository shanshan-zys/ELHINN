import os
import cv2
import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def velocity_visualization(filename,opticalflow):
    gap = (opticalflow.shape[0]//9)-1
    border_max = torch.abs(opticalflow).max().item()
    border_quantile = numpy.quantile(torch.abs(opticalflow).numpy(),0.997).item()
    ratio = (border_max-border_quantile)/border_max
    border = border_quantile*1.1 if ratio>0.05 else border_max*0.95
    opticalflow = torch.clamp(opticalflow,-border,border)/border
    fig,axes = plt.subplots(nrows=4,ncols=5,figsize=(18,12))
    for channel in range(2):
        for timestep in range(10):
            axes[channel*2+timestep//5,timestep%5].imshow(
                opticalflow[gap*timestep,channel,:,:],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    for ax in axes.flatten():
        ax.axis('on')
        ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT,'data','mot20','velocity',f'{filename}.png'))
    plt.close()
    fig,(r0c0,r0c1) = plt.subplots(nrows=1,ncols=2)
    im00 = r0c0.imshow(opticalflow[0,0],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    im01 = r0c1.imshow(opticalflow[0,1],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    plt.tight_layout()
    def update(timestep,opticalflow):
        im00.set_data(opticalflow[timestep,0])
        im01.set_data(opticalflow[timestep,1])
        return [im00,im01]
    gif = FuncAnimation(fig,update,frames=opticalflow.shape[0],fargs=(opticalflow,),interval=41)
    gif.save(os.path.join(ROOT,'data','mot20','velocity',f'{filename}.gif'),writer='pillow',fps=24)
    plt.close()

def video_to_velocity(filename):
    video_path = os.path.join(ROOT,'data','mot20','video',f'{filename}.mp4')
    video = cv2.VideoCapture(video_path)
    frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    info_path = os.path.join(ROOT,'data','mot20','video','info.txt')
    with open(info_path,'a') as file:
        file.write(f'video={filename}, frame={frame}, height={height}, width={width}\n')
    # calculate optical flow
    _,frame0 = video.read()
    frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    opticalflow = torch.zeros((frame,2,height,width),dtype=torch.float)
    for i in range(frame):
        _,frame0 = video.read()
        frame2 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame1,frame2,None,0.5,5,15,5,5,1.1,0)
        opticalflow[i,0,:,:] = torch.from_numpy(flow[...,0])
        opticalflow[i,1,:,:] = torch.from_numpy(flow[...,1])
        frame1 = frame2
    video.release()
    # save first 100 frames
    torch.save(opticalflow[:100,:,:,:].detach().cpu().numpy(),os.path.join(ROOT,'data','mot20','velocity',f'{filename}.pt'))
    velocity_visualization(filename,opticalflow[:100,:,:,:])

if __name__=="__main__":
    vel_dir = os.path.join(ROOT,'data','mot20','velocity')
    os.makedirs(vel_dir,exist_ok=True)
    video_dir = os.path.join(ROOT,'data','mot20','video')
    filelist = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    for idx in range(len(filelist)):
        filename = filelist[idx].replace('.mp4','')
        print(f'Processing: {filename}')
        if os.path.exists(os.path.join(vel_dir,f'{filename}.pt')):
            continue
        video_to_velocity(filename)
