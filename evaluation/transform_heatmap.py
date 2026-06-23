import os
import numpy
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

def transform_heatmap(baseline,category,filename):
    if baseline=='gt':
        velocity = torch.load(os.path.join(ROOT,'data','dcfd','velocity',category,f'{filename}.pt'),weights_only=False)
        velocity = torch.tensor(velocity,dtype=torch.float)
        border_max = torch.abs(velocity).max().item()
        border_quantile = numpy.quantile(torch.abs(velocity).numpy(),0.997).item()
        ratio = (border_max-border_quantile)/border_max
        border = border_quantile*1.1 if ratio>0.05 else border_max*0.95
        velocity = torch.clamp(velocity,-border,border)/border
    else:
        velocity = torch.load(os.path.join(ROOT,'output','evaluation','heatmap',baseline,category,f'{filename}.pt'),weights_only=False)
        velocity = torch.tensor(velocity,dtype=torch.float)
    frame,_,height,width = velocity.shape
    velocity = velocity.index_select(0,torch.arange(0,frame,(frame//4)-1))
    velocity = velocity.reshape(-1,height,width)
    ssim_dir = os.path.join(ROOT,'output','evaluation','ssim',baseline)
    inception_dir = os.path.join(ROOT,'output','evaluation','inception',baseline)
    for idx in range(velocity.shape[0]):
        plt.figure(figsize=(4,3))
        plt.imshow(velocity[idx,:,:],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(ssim_dir,f'{filename}_{idx}.png'))
        plt.close()
    torch.save(velocity.detach().cpu().numpy(),os.path.join(inception_dir,f'{filename}.pt'))

if __name__=="__main__":
    baselines = ['ours']
    for baseline in baselines:
        ssim_dir = os.path.join(ROOT,'output','evaluation','ssim',baseline)
        inception_dir = os.path.join(ROOT,'output','evaluation','inception',baseline)
        os.makedirs(ssim_dir,exist_ok=True)
        os.makedirs(inception_dir,exist_ok=True)
        vel_dir = os.path.join(ROOT,'data','dcfd','velocity')
        folderlist = sorted(os.listdir(vel_dir))
        for category in folderlist:
            filelist = sorted([f for f in os.listdir(os.path.join(vel_dir,category)) if f.endswith('.pt')])
            _,test_split = train_test_split(filelist,train_size=0.8,test_size=0.2,random_state=42)
            test_split = sorted(test_split)
            for idx in range(len(test_split)):
                filename = test_split[idx].replace('.pt','')
                print(f'Processing: {filename}')
                transform_heatmap(baseline,category,filename)
