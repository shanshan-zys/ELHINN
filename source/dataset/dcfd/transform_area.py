import os
import cv2
import PIL
import numpy
import torch
import scipy.io
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def area_visualization(category,filename):
    frame_path = os.path.join(ROOT,'data','dcfd','frame',category,f'{filename}.png')
    frame = PIL.Image.open(frame_path)
    frame = numpy.array(frame.convert('RGB'))
    area = torch.load(os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.pt'),weights_only=False)
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

def mat_to_pt(category,filename):
    mat_path = os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.mat')
    area = scipy.io.loadmat(mat_path)
    area_mat = area['area']
    area_pt = torch.tensor(area_mat)
    torch.save(area_pt.detach().cpu().numpy(),os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.pt'))
    area_visualization(category,filename)

def pt_to_mat(category,filename):
    pt_path = os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.pt')
    area = torch.load(pt_path,weights_only=False)
    mat_path = os.path.join(ROOT,'data','dcfd','area',category,f'{filename}.mat')
    scipy.io.savemat(mat_path,{'area':area})

if __name__=='__main__':
    area_dir = os.path.join(ROOT,'data','dcfd','area')
    folderlist = sorted(os.listdir(area_dir))
    for category in folderlist:
        cat_dir = os.path.join(area_dir,category)
        filelist = sorted([f for f in os.listdir(cat_dir) if f.endswith('.mat')])
        for idx in range(len(filelist)):
            filename = filelist[idx].replace('.mat','')
            print(f'Processing: {filename}')
            mat_to_pt(category,filename)
