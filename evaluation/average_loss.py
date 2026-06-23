import os
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

class DataList(torch.utils.data.Dataset):
    def __init__(self,folder):
        self.filelist = []
        folderlist = sorted(os.listdir(folder))
        for category in folderlist:
            cat_dir = os.path.join(folder,category)
            filelist = sorted([f for f in os.listdir(cat_dir) if f.endswith('.pt')])
            self.filelist.extend([os.path.join(cat_dir,f) for f in filelist])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        velocity = torch.load(self.filelist[idx],weights_only=False)
        velocity = torch.tensor(velocity,dtype=torch.float)
        return self.filelist[idx],velocity

def average_loss(baseline):
    gt_dir = os.path.join(ROOT,'output','evaluation','heatmap','gt')
    pred_dir = os.path.join(ROOT,'output','evaluation','heatmap',baseline)
    groundtruth_list = DataList(folder=gt_dir)
    groundtruth_loader = torch.utils.data.DataLoader(groundtruth_list,batch_size=32,shuffle=False,num_workers=4)
    generation_list = DataList(folder=pred_dir)
    generation_loader = torch.utils.data.DataLoader(generation_list,batch_size=32,shuffle=False,num_workers=4)
    al_values = []
    for _,((_,velocity1),(_,velocity2)) in enumerate(zip(groundtruth_loader,generation_loader)):
        al_loss = torch.nn.SmoothL1Loss()(velocity1,velocity2)*velocity1.shape[0]
        al_values.append(al_loss.item())
    al_mean = sum(al_values)/len(al_values)
    print(f'{baseline} Average Loss: {al_mean}')

if __name__=="__main__":
    baselines = ['gt','ours']
    for baseline in baselines:
        average_loss(baseline)
