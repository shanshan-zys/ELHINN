import os
import cv2
import json
import numpy
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def initial_visualization(filename):
    frame = cv2.imread(os.path.join(ROOT,'data','mot20','frame',f'{filename}.png'))
    positions = torch.load(os.path.join(ROOT,'data','mot20','initial','position',f'{filename}.pt'),weights_only=False)
    velocities = torch.load(os.path.join(ROOT,'data','mot20','initial','velocity',f'{filename}.pt'),weights_only=False)
    for position,velocity in zip(positions,velocities):
        x,y,u,v = int(position[0]),int(position[1]),int(velocity[0]),int(velocity[1])
        cv2.circle(frame,(x,y),3,(0,0,255),-1)
        cv2.arrowedLine(frame,(x,y),(x+u,y+v),(255,0,0),2,tipLength=0.5)
    cv2.imwrite(os.path.join(ROOT,'data','mot20','initial','position',f'{filename}.png'),frame)

def trajectory_to_initial(filename):
    with open(os.path.join(ROOT,'data','mot20','trajectory',f'{filename}.json'),'r') as file:
        trajectories = json.load(file)
    positions,velocities = [],[]
    for _,trajectory in trajectories.items():
        trajectory = sorted(trajectory,key=lambda x:x['frame'])
        if len(trajectory)>=2 and trajectory[0]['frame']==1:
            position0 = numpy.array([trajectory[0]['x'],trajectory[0]['y']],dtype=numpy.float32)
            position1 = numpy.array([trajectory[1]['x'],trajectory[1]['y']],dtype=numpy.float32)
            velocity = position1-position0
            positions.append(position0.tolist())
            velocities.append(velocity.tolist())
    positions = torch.tensor(positions,dtype=torch.float)
    velocities = torch.tensor(velocities,dtype=torch.float)
    torch.save(positions.detach().cpu().numpy(),os.path.join(ROOT,'data','mot20','initial','position',f'{filename}.pt'))
    torch.save(velocities.detach().cpu().numpy(),os.path.join(ROOT,'data','mot20','initial','velocity',f'{filename}.pt'))
    initial_visualization(filename)

if __name__=='__main__':
    pos_dir = os.path.join(ROOT,'data','mot20','initial','position')
    vel_dir = os.path.join(ROOT,'data','mot20','initial','velocity')
    os.makedirs(pos_dir,exist_ok=True)
    os.makedirs(vel_dir,exist_ok=True)
    traj_dir = os.path.join(ROOT,'data','mot20','trajectory')
    filelist = sorted([f for f in os.listdir(traj_dir) if f.endswith('.json')])
    for idx in range(len(filelist)):
        filename = filelist[idx].replace('.json','')
        print(f'Processing: {filename}')
        trajectory_to_initial(filename)
