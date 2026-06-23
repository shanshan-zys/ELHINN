import os
import re
import cv2
import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def read_pfm(category,filename):
    pfm_path = os.path.join(ROOT,'data','dcfd','depth',category,f'{filename}.pfm')
    with open(pfm_path,'rb') as file:
        header = file.readline().rstrip()
        color = True if header.decode('ascii')=='PF' else False
        dim_match = re.match(r'^(\d+)\s(\d+)\s$',file.readline().decode('ascii'))
        if dim_match:
            width,height = list(map(int,dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')
        endian = '<' if float(file.readline().decode('ascii').rstrip())<0 else '>'
        depth = numpy.fromfile(file,endian+'f')
        shape = (height,width,3) if color else (height,width)
        depth = numpy.reshape(depth,shape)
        depth = numpy.flipud(depth)
        return depth

def fitplane(points,iterations=1000,threshold=0.1):
    best_plane = None
    best_inliers = []
    sample_number = points.shape[0]
    for i in range(iterations):
        sample_indices = torch.randint(0,sample_number,(3,))
        sample = points[sample_indices]
        v1 = sample[1]-sample[0]
        v2 = sample[2]-sample[0]
        normal_vector = torch.cross(v1,v2,dim=0)
        norm = torch.norm(normal_vector)
        if norm<1e-6:
            continue
        normal_vector = normal_vector/norm
        distance = -torch.dot(normal_vector,sample[0])
        distances = torch.abs(torch.matmul(points,normal_vector)+distance)
        inliers = torch.nonzero(distances<threshold).squeeze()
        if len(inliers)>len(best_inliers):
            best_plane = (normal_vector,distance)
            best_inliers = inliers
    return best_plane,points[best_inliers]

def velocity_visualization(category,filename,opticalflow):
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
    plt.savefig(os.path.join(ROOT,'data','dcfd','velocity',category,f'{filename}.png'))
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
    gif.save(os.path.join(ROOT,'data','dcfd','velocity',category,f'{filename}.gif'),writer='pillow',fps=24)
    plt.close()

def video_to_velocity(category,filename):
    video_path = os.path.join(ROOT,'data','dcfd','video',category,f'{filename}.mp4')
    video = cv2.VideoCapture(video_path)
    frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    info_path = os.path.join(ROOT,'data','dcfd','video',category,'info.txt')
    with open(info_path,'a') as file:
        file.write(f'video={filename}, frame={frame}, height={height}, width={width}\n')
    # homography transform
    source = numpy.array([[0,height],[width,height],[width//2,height//2],[0,height//2]],dtype=numpy.float32)
    destination = numpy.array([[0,height*1.2],[width,height*1.2],[width//2,height*1.2//2],[0,height*1.2//2]],dtype=numpy.float32)
    homo,_ = cv2.findHomography(source,destination)
    # calculate optical flow
    _,frame0 = video.read()
    frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    opticalflow = torch.zeros((frame,2,height,width),dtype=torch.float)
    for i in range(frame):
        _,frame0 = video.read()
        frame2 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame1,frame2,None,0.5,5,15,5,5,1.1,0)
        flow_homo = cv2.perspectiveTransform(flow.reshape(-1,1,2).astype(numpy.float32),homo).reshape(flow.shape)
        flow_homo = torch.from_numpy(flow_homo).float()
        opticalflow[i,0,:,:] = flow_homo[...,0]
        opticalflow[i,1,:,:] = flow_homo[...,1]
        frame1 = frame2
    video.release()
    # camera offset removal using unwalkable area
    area_path = os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.pt')
    area = torch.load(area_path,weights_only=False)
    area = torch.tensor(area,dtype=torch.float).unsqueeze(0)
    unwalkable_area = (area==0).reshape(1,1,-1)
    offset = opticalflow.view(frame,2,-1)*unwalkable_area
    offset = offset.sum(dim=2,keepdim=True)/(unwalkable_area.sum(dim=2,keepdim=True)+1e-6)
    velocity = (opticalflow-offset.view(frame,2,1,1))*area
    # save velocity segments (4 segments of 25 frames each)
    vel_dir = os.path.join(ROOT,'data','dcfd','velocity',category)
    for i in range(4):
        torch.save(velocity[i*25:(i+1)*25,:,:,:].detach().cpu().numpy(),os.path.join(vel_dir,f'{filename}{i+1}.pt'))
        velocity_visualization(category,f'{filename}{i+1}',velocity[i*25:(i+1)*25,:,:,:])

if __name__=="__main__":
    video_dir = os.path.join(ROOT,'data','dcfd','video')
    folderlist = sorted(os.listdir(video_dir))
    for category in folderlist:
        vel_dir = os.path.join(ROOT,'data','dcfd','velocity',category)
        os.makedirs(vel_dir,exist_ok=True)
        filelist = sorted([f for f in os.listdir(os.path.join(video_dir,category)) if f.endswith('.mp4')])
        for idx in range(len(filelist)):
            filename = filelist[idx].replace('.mp4','')
            print(f'Processing: {filename}')
            if os.path.exists(os.path.join(vel_dir,f'{filename}1.pt')):
                continue
            video_to_velocity(category,filename)
