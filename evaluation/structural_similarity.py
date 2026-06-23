import os
import cv2
import numpy
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

class DataList(torch.utils.data.Dataset):
    def __init__(self,folder):
        self.filelist = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.png')])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        image = cv2.imread(self.filelist[idx])
        image = torch.from_numpy(numpy.rollaxis(image,2)).float()
        return self.filelist[idx],image

def gaussian():
    gauss = torch.exp(-((torch.arange(11).float()-5)**2)/(2*1.5**2))
    return gauss/gauss.sum()

def create_window():
    _1D_window = gaussian().unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(3,1,11,11).contiguous())
    return window

def structural_similarity(baseline):
    gt_dir = os.path.join(ROOT,'output','evaluation','ssim','gt')
    pred_dir = os.path.join(ROOT,'output','evaluation','ssim',baseline)
    groundtruth_list = DataList(folder=gt_dir)
    groundtruth_loader = torch.utils.data.DataLoader(groundtruth_list,batch_size=32,shuffle=False,num_workers=4)
    generation_list = DataList(folder=pred_dir)
    generation_loader = torch.utils.data.DataLoader(generation_list,batch_size=32,shuffle=False,num_workers=4)
    window = create_window()
    ssim_values = []
    for _,((_,image1),(_,image2)) in enumerate(zip(groundtruth_loader,generation_loader)):
        penalty = torch.tensor(0.5) if torch.allclose(image2,255*torch.ones_like(image2)) else torch.tensor(0.0)
        mu1 = torch.nn.functional.conv2d(image1,window,padding=11//2,groups=3)
        mu2 = torch.nn.functional.conv2d(image2,window,padding=11//2,groups=3)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = torch.nn.functional.conv2d(image1*image1,window,padding=11//2,groups=3)-mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(image2*image2,window,padding=11//2,groups=3)-mu2_sq
        sigma12 = torch.nn.functional.conv2d(image1*image2,window,padding=11//2,groups=3)-mu1_mu2
        C1,C2 = 0.01**2,0.03**2
        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))+penalty
        ssim_values.append(ssim_map.mean().item())
    ssim_mean = sum(ssim_values)/len(ssim_values)
    print(f'{baseline} Structural Similarity: {ssim_mean}')

if __name__=="__main__":
    baselines = ['gt','ours']
    for baseline in baselines:
        structural_similarity(baseline)
