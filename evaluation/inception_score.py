# Pretrained weights: https://drive.google.com/drive/folders/1ucgGPeGMmqzl__vkg9AyvHzP_6jIKXYl
import os
import sys
import numpy
import scipy
import torch

sys.path.insert(0,os.path.dirname(__file__))
from inception import Inception

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

def inception_score(baseline):
    data_dir = os.path.join(ROOT,'output','evaluation','inception',baseline)
    generation_list = DataList(folder=data_dir)
    generation_loader = torch.utils.data.DataLoader(generation_list,batch_size=32,shuffle=False,num_workers=4)
    model = Inception(10,6)
    checkpoint = os.path.join(ROOT,'evaluation','inception_score.pth')
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(parameter,strict=False)
        print('Checkpoint loaded!')
    model.to(device)
    with torch.no_grad():
        model.eval()
        prediction = torch.tensor([],dtype=torch.float,device=device)
        for _,(_,velocity) in enumerate(generation_loader):
            velocity = velocity.to(device)
            output = model(velocity)
            classification = torch.nn.functional.softmax(output,dim=1)
            prediction = torch.cat([prediction,classification],dim=0)
    prediction = prediction.detach().cpu().numpy()
    scores = []
    for i in range(splits):
        part = prediction[i*(len(generation_list)//splits):(i+1)*(len(generation_list)//splits),:]
        py = numpy.mean(part,axis=0)
        score = []
        for j in range(part.shape[0]):
            pyx = part[j,:]
            score.append(scipy.stats.entropy(pyx,py))
        scores.append(numpy.exp(numpy.mean(score)))
    print(f'{baseline} Inception Score: {scores},\n mean: {numpy.mean(scores)}, std: {numpy.std(scores)}')

if __name__=="__main__":
    global splits,device
    splits = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baselines = ['gt','ours']
    for baseline in baselines:
        inception_score(baseline)
