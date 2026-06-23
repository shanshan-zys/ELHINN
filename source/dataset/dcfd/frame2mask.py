# Requires SAM2: https://github.com/facebookresearch/sam2
# Clone the repository and download checkpoint (sam2.1_hiera_large.pt) before running.

import os
import cv2
import PIL
import json
import numpy
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))
SAM2_DIR = os.path.join(ROOT,'third_party','sam2')

import sys
sys.path.insert(0,SAM2_DIR)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def mask_visualization(category,filename,frame,masks):
    plt.figure(figsize=(20,15))
    plt.imshow(frame)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    fig = numpy.ones((masks[0]['segmentation'].shape[0],masks[0]['segmentation'].shape[1],4))
    fig[:,:,3] = 0
    for mask in masks:
        segmentation = mask['segmentation']
        color_mask = numpy.concatenate([numpy.random.random(3),[0.5]])
        fig[segmentation] = color_mask
        contours,_ = cv2.findContours(segmentation.astype(numpy.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour,epsilon=0.01,closed=True) for contour in contours]
        cv2.drawContours(fig,contours,-1,(0,0,1,0.4),thickness=1)
    ax.imshow(fig)
    plt.axis('off')
    plt.savefig(os.path.join(ROOT,'data','dcfd','mask',category,f'{filename}.png'))
    plt.close()

def frame_to_mask(category,filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numpy.random.seed(42)
    frame_path = os.path.join(ROOT,'data','dcfd','frame',category,f'{filename}.png')
    frame = PIL.Image.open(frame_path)
    frame = numpy.array(frame.convert('RGB'))
    config = os.path.join(SAM2_DIR,'configs','sam2.1','sam2.1_hiera_l.yaml')
    checkpoint = os.path.join(SAM2_DIR,'checkpoints','sam2.1_hiera_large.pt')
    sam = build_sam2(config,checkpoint,device,apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(sam)
    masks = generator.generate(frame)
    sorted_masks = sorted(masks,key=(lambda x: x['area']),reverse=True)
    mask_info = {}
    for idx,mask in enumerate(sorted_masks):
        mask_info[f'mask{idx+1}'] = {'area': mask['area'],
                                     'segmentation': mask['segmentation'].astype(numpy.uint8).tolist()}
    mask_path = os.path.join(ROOT,'data','dcfd','mask',category,f'{filename}.json')
    with open(mask_path,'w') as file:
        json.dump(mask_info,file)
    mask_visualization(category,filename,frame,sorted_masks)

if __name__=="__main__":
    frame_dir = os.path.join(ROOT,'data','dcfd','frame')
    folderlist = sorted(os.listdir(frame_dir))
    for category in folderlist:
        mask_dir = os.path.join(ROOT,'data','dcfd','mask',category)
        os.makedirs(mask_dir,exist_ok=True)
        filelist = sorted([f for f in os.listdir(os.path.join(frame_dir,category)) if f.endswith('.png')])
        for idx in range(len(filelist)):
            filename = filelist[idx].replace('.png','')
            print(f'Processing: {filename}')
            frame_to_mask(category,filename)
