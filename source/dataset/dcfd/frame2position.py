# Requires P2PNet: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
# Clone the repository and download weights (SHTechA.pth) before running.

import os
import sys
import PIL
import numpy
import torch
import argparse
import torchvision
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))
P2PNET_DIR = os.path.join(ROOT,'third_party','p2pnet')
sys.path.insert(0,P2PNET_DIR)
from models import build_model

def position_visualization(category,filename):
    frame_path = os.path.join(ROOT,'data','dcfd','frame',category,f'{filename}.png')
    frame = PIL.Image.open(frame_path)
    frame = numpy.array(frame.convert('RGB'))
    position = torch.load(os.path.join(ROOT,'data','dcfd','position',category,f'{filename}.pt'),weights_only=False)
    plt.figure(figsize=(20,15))
    plt.imshow(frame)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    points = numpy.where(position)
    ax.scatter(points[1],points[0],c='red',s=5)
    plt.axis('off')
    plt.savefig(os.path.join(ROOT,'data','dcfd','position',category,f'{filename}.png'))
    plt.close()

def frame_to_position(category):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(argparse.Namespace(backbone='vgg16_bn',row=2,line=2))
    checkpoint = os.path.join(P2PNET_DIR,'weights','SHTechA.pth')
    parameter = torch.load(checkpoint,map_location=device)
    model.load_state_dict(parameter['model'])
    model.to(device)
    with torch.no_grad():
        model.eval()
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                                                                                     std=[0.229,0.224,0.225]),])
        frame_dir = os.path.join(ROOT,'data','dcfd','frame',category)
        filelist = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        for idx in range(len(filelist)):
            filename = filelist[idx].replace('.png','')
            print(f'Processing: {filename}')
            frame = PIL.Image.open(os.path.join(frame_dir,f'{filename}.png')).convert('RGB')
            width,height = frame.size
            new_width,new_height = width//128*128,height//128*128
            input = frame.resize((new_width,new_height),PIL.Image.Resampling.LANCZOS)
            input = transform(input).unsqueeze(0).to(device)
            output = model(input)
            score = torch.nn.functional.softmax(output['pred_logits'],-1)[:,:,1][0]
            points = output['pred_points'][0][score>0.5].detach().cpu().numpy().tolist()
            position = torch.zeros((height,width),dtype=torch.float)
            for point in points:
                x,y = int(point[0]*width/new_width),int(point[1]*height/new_height)
                if 0<=y<height and 0<=x<width:
                    position[y,x] = 1
            area = torch.load(os.path.join(ROOT,'data','dcfd','area',category,f'{filename[:-1]}.pt'),weights_only=False)
            area = torch.tensor(area,dtype=torch.float)
            position = position*area
            torch.save(position.detach().cpu().numpy(),os.path.join(ROOT,'data','dcfd','position',category,f'{filename}.pt'))
            position_visualization(category,filename)

if __name__=="__main__":
    frame_dir = os.path.join(ROOT,'data','dcfd','frame')
    folderlist = sorted(os.listdir(frame_dir))
    for category in folderlist:
        pos_dir = os.path.join(ROOT,'data','dcfd','position',category)
        os.makedirs(pos_dir,exist_ok=True)
        frame_to_position(category)
