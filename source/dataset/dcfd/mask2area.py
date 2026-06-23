import os
import cv2
import PIL
import json
import numpy
import torch
from functools import reduce
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def area_visualization(category,filename,frame,area):
    plt.figure(figsize=(20,15))
    plt.imshow(frame)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    fig = numpy.ones((area.shape[0],area.shape[1],4))
    fig[:,:,3] = 0
    color_mask = numpy.array([0,0,1,0.5])
    fig[area==1] = color_mask
    contours,_ = cv2.findContours(area.astype(numpy.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = [cv2.approxPolyDP(contour,epsilon=0.01,closed=True) for contour in contours]
    cv2.drawContours(fig,contours,-1,(0,0,1,0.4),thickness=1)
    ax.imshow(fig)
    plt.axis('off')
    plt.savefig(os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.png'))
    plt.close()

def mask_to_area(category,filename,mask_names):
    mask_path = os.path.join(ROOT,'data','dcfd','mask',category,f'{filename}.json')
    with open(mask_path,'r') as file:
        mask_info = json.load(file)
    segmentations = [torch.tensor(mask_info[mask]['segmentation']) for mask in mask_names]
    area = 1-reduce(torch.logical_or,segmentations).int()
    torch.save(area,os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.pt'))
    frame_path = os.path.join(ROOT,'data','dcfd','frame',category,f'{filename}.png')
    frame = PIL.Image.open(frame_path)
    frame = numpy.array(frame.convert('RGB'))
    area_visualization(category,filename,frame,area.detach().cpu().numpy())

if __name__=="__main__":
    # Manual configuration required per scene:
    # Set category, filename, and mask_names (which masks represent non-walkable regions)
    category = 'curve'
    filename = f'{category}01'
    mask_names = ['mask1']
    area_dir = os.path.join(ROOT,'data','dcfd','area',category)
    os.makedirs(area_dir,exist_ok=True)
    print(f'Processing: {filename}')
    mask_to_area(category,filename,mask_names)
