# Pretrained weights: https://drive.google.com/drive/folders/1ucgGPeGMmqzl__vkg9AyvHzP_6jIKXYl
import os
import sys
import numpy
import scipy
import torch

sys.path.insert(0,os.path.dirname(__file__))
from inception import FIDInception

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

class DataList(torch.utils.data.Dataset):
    def __init__(self,folder):
        self.filelist = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.pt')])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        velocity = torch.load(self.filelist[idx],weights_only=False)
        velocity = torch.tensor(velocity,dtype=torch.float)
        return self.filelist[idx],velocity

def frechet_inception_distance(baseline):
    gt_dir = os.path.join(ROOT,'output','evaluation','inception','gt')
    pred_dir = os.path.join(ROOT,'output','evaluation','inception',baseline)
    groundtruth_list = DataList(folder=gt_dir)
    groundtruth_loader = torch.utils.data.DataLoader(groundtruth_list,batch_size=32,shuffle=False,num_workers=4)
    generation_list = DataList(folder=pred_dir)
    generation_loader = torch.utils.data.DataLoader(generation_list,batch_size=32,shuffle=False,num_workers=4)
    model = FIDInception(10,6)
    checkpoint = os.path.join(ROOT,'evaluation','frechet_inception_distance.pth')
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(parameter,strict=False)
        print('Checkpoint loaded!')
    model.to(device)
    with torch.no_grad():
        model.eval()
        prediction1 = torch.tensor([],dtype=torch.float,device=device)
        prediction2 = torch.tensor([],dtype=torch.float,device=device)
        for _,(_,velocity) in enumerate(groundtruth_loader):
            velocity = velocity.to(device)
            output = model(velocity)
            classification = torch.nn.functional.softmax(output,dim=1)
            prediction1 = torch.cat([prediction1,classification],dim=0)
        for _,(_,velocity) in enumerate(generation_loader):
            velocity = velocity.to(device)
            output = model(velocity)
            classification = torch.nn.functional.softmax(output,dim=1)
            prediction2 = torch.cat([prediction2,classification],dim=0)
    prediction1 = prediction1.detach().cpu().numpy()
    prediction2 = prediction2.detach().cpu().numpy()
    mu1,sigma1 = numpy.mean(prediction1,axis=0),numpy.cov(prediction1,rowvar=False)
    mu2,sigma2 = numpy.mean(prediction2,axis=0),numpy.cov(prediction2,rowvar=False)
    difference = mu1-mu2
    covmean,_ = scipy.linalg.sqrtm(sigma1.dot(sigma2),disp=False)
    if not numpy.isfinite(covmean).all():
        offset = numpy.eye(sigma1.shape[0])*(1e-6)
        covmean = scipy.linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    if numpy.iscomplexobj(covmean):
        if not numpy.allclose(numpy.diagonal(covmean).imag,0,atol=1e-3):
            m = numpy.max(numpy.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = numpy.trace(covmean)
    fid = difference.dot(difference)+numpy.trace(sigma1)+numpy.trace(sigma2)-2*tr_covmean
    print(f'{baseline} Frechet Inception Distance: {max(fid,0.0)}')

if __name__=="__main__":
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baselines = ['gt','ours']
    for baseline in baselines:
        frechet_inception_distance(baseline)
