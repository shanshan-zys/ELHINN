import os
import time
import numpy
import torch
import random
from hinnpp import HINNPP
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def get_teacher_forcing_ratio(epoch,max_epochs=200,initial_ratio=1.0,final_ratio=0.1):
    if epoch<max_epochs:
        step = torch.tensor((epoch-max_epochs//2)*0.1,dtype=torch.float)
        ratio = final_ratio+(initial_ratio-final_ratio)/(1+torch.exp(step))
    else:
        ratio = final_ratio
    return ratio

class DataList(torch.utils.data.Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.filelist = sorted([file for file in os.listdir(folder) if file.endswith('.pt')])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        file = os.path.join(self.folder,self.filelist[idx])
        velocity = torch.load(file,weights_only=False)
        velocity = torch.tensor(velocity,dtype=torch.float)
        border_max = torch.abs(velocity).max().item()
        border_quantile = numpy.quantile(torch.abs(velocity).numpy(),0.997).item()
        ratio = (border_max-border_quantile)/border_max
        border = border_quantile*1.1 if ratio>0.05 else border_max*0.95
        velocity = torch.clamp(velocity,-border,border)/border
        area = torch.load(f"{file.replace('velocity','area')[:-4]}.pt",weights_only=False)
        area = torch.tensor(area,dtype=torch.float).unsqueeze(0)
        return self.filelist[idx],area,velocity

def velocity_visualization(category,filename,epochs,groundtruth,prediction):
    gap = (frame//9)-1
    groundtruth = groundtruth.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    _,axes = plt.subplots(nrows=4,ncols=10,figsize=(36,12))
    for channel in range(2):
        for timestep in range(10):
            axes[channel*2,timestep].imshow(
                groundtruth[gap*timestep,channel,:,:],vmin=-1,vmax=1,cmap='rainbow',origin='upper')
            axes[channel*2+1,timestep].imshow(
                prediction[gap*timestep,channel,:,:],vmin=-1,vmax=1,cmap='rainbow',origin='upper')
    for ax in axes.flatten():
        ax.axis('on')
        ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT,'output','heatmap',category,f'{filename}_{epochs}.png'))
    plt.close()
    fig,((r0c0,r0c1),(r1c0,r1c1)) = plt.subplots(nrows=2,ncols=2)
    im00 = r0c0.imshow(groundtruth[0,0],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    im01 = r0c1.imshow(groundtruth[0,1],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    im10 = r1c0.imshow(prediction[0,0],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    im11 = r1c1.imshow(prediction[0,1],cmap='rainbow',vmin=-1,vmax=1,origin='upper')
    plt.tight_layout()
    def update(timestep,groundtruth,prediction):
        im00.set_data(groundtruth[timestep,0])
        im01.set_data(groundtruth[timestep,1])
        im10.set_data(prediction[timestep,0])
        im11.set_data(prediction[timestep,1])
        return [im00,im01,im10,im11]
    gif = FuncAnimation(fig,update,frames=frame-1,fargs=(groundtruth,prediction),interval=41)
    gif.save(os.path.join(ROOT,'output','heatmap',category,f'{filename}_{epochs}.gif'),writer='pillow',fps=24)
    plt.close()

def train(category,epochs,steps):
    data_list = DataList(folder=os.path.join(ROOT,'data','dcfd','velocity',category))
    train_split,_ = train_test_split(data_list.filelist,train_size=0.8,test_size=0.2,random_state=42)
    train_split = sorted(train_split)
    train_list = DataList(folder=os.path.join(ROOT,'data','dcfd','velocity',category))
    train_list.filelist = train_split
    train_loader = torch.utils.data.DataLoader(train_list,batch_size=3,shuffle=True)
    print('Train List:',train_split)
    model = HINNPP()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4,
                                 betas=(0.9,0.999),
                                 eps=1e-8,
                                 weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,200],gamma=0.1)
    model.to(device)
    checkpoint = os.path.join(ROOT,'output','checkpoint',f'hinnpp_{category}_{epochs}.pth')
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(parameter['model_state_dict'],strict=False)
        optimizer.load_state_dict(parameter['optimizer_state_dict'])
        scheduler.load_state_dict(parameter['scheduler_state_dict'])
        print('Checkpoint loaded!')
    start_time = time.time()
    for epoch in range(steps):
        model.train()
        print(f'Iter {epoch+epochs}:')
        with open(os.path.join(ROOT,'output','loss',f'hinnpp_{category}.txt'),'a') as file:
            file.write(f'Iter {epoch+epochs}:\n')
            epoch_loss = 0
            teacher_forcing = get_teacher_forcing_ratio(epoch+epochs)
            for batch,(_,area,groundtruth) in enumerate(train_loader):
                area = area.to(device)
                groundtruth = groundtruth.to(device)
                input = groundtruth[:,0,:,:,:]
                loss,batch_loss = None,0
                optimizer.zero_grad()
                for i in range(frame-1):
                    output = model(area,input)
                    target = groundtruth[:,i+1,:,:,:]
                    loss_i = ((i+1)/frame)*torch.nn.SmoothL1Loss()(target,output)
                    loss = loss_i if loss is None else (loss+loss_i)
                    batch_loss += loss_i.item()
                    with torch.no_grad():
                        input = target if random.random()<teacher_forcing else output.detach()
                    if (i+1)%5==0:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        loss = None
                epoch_loss += batch_loss
                print(f'Batch {batch}, loss: {batch_loss}')
                file.write(f'Batch {batch}, loss: {batch_loss}\n')
            avg_train_loss = epoch_loss/len(train_loader)
            print(f'Iter {epoch+epochs}, training loss: {avg_train_loss}\n')
            file.write(f'Iter {epoch+epochs}, training loss: {avg_train_loss}\n\n')
        scheduler.step()
    elapsed = time.time()-start_time
    print(f'Training time: {elapsed}\n')
    with open(os.path.join(ROOT,'output','loss',f'hinnpp_{category}.txt'),'a') as file:
        file.write(f'Training time: {elapsed}\n')
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()},os.path.join(ROOT,'output','checkpoint',f'hinnpp_{category}_{epochs+steps}.pth'))

def test(category,epochs):
    data_list = DataList(folder=os.path.join(ROOT,'data','dcfd','velocity',category))
    _,test_split = train_test_split(data_list.filelist,train_size=0.8,test_size=0.2,random_state=42)
    test_split = sorted(test_split)
    test_list = DataList(folder=os.path.join(ROOT,'data','dcfd','velocity',category))
    test_list.filelist = test_split
    test_loader = torch.utils.data.DataLoader(test_list,batch_size=1,shuffle=False)
    print('Test List:',test_split)
    model = HINNPP()
    model.to(device)
    checkpoint = os.path.join(ROOT,'output','checkpoint',f'hinnpp_{category}_{epochs}.pth')
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device,weights_only=True)
        model.load_state_dict(parameter['model_state_dict'],strict=False)
        print('Checkpoint loaded!')
    with torch.no_grad():
        model.eval()
        for _,(filename,area,groundtruth) in enumerate(test_loader):
            filename = filename[0].replace('.pt','')
            area = area.to(device)
            groundtruth = groundtruth.squeeze(0).to(device)
            prediction = torch.zeros_like(groundtruth)
            input = groundtruth[0,:,:,:].unsqueeze(0)
            prediction[0,:,:,:] = groundtruth[0,:,:,:]
            for i in range(frame-1):
                prediction[i+1,:,:,:] = model(area,input).squeeze(0)*area
                with torch.no_grad():
                    input = prediction[i+1,:,:,:].unsqueeze(0).detach()
            border = torch.abs(prediction[1:]).max().item()
            prediction[1:] = torch.clamp(prediction[1:],-border,border)/border
            loss = torch.nn.SmoothL1Loss()(groundtruth,prediction)
            print(f'Test {filename}, loss: {loss.item()}')
            with open(os.path.join(ROOT,'output','loss',f'hinnpp_{category}.txt'),'a') as file:
                file.write(f'Test {filename}, loss: {loss.item()}\n')
            torch.save(prediction.detach().cpu().numpy(),os.path.join(ROOT,'output','heatmap',category,f'{filename}_{epochs}.pt'))
            velocity_visualization(category,filename,epochs,groundtruth,prediction)
        print('')
        with open(os.path.join(ROOT,'output','loss',f'hinnpp_{category}.txt'),'a') as file:
            file.write(f'\n')

if __name__=="__main__":
    global frame,height,width,device
    frame,height,width = 25,360,480
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(ROOT,'output','checkpoint'),exist_ok=True)
    os.makedirs(os.path.join(ROOT,'output','loss'),exist_ok=True)
    folderlist = sorted(os.listdir(os.path.join(ROOT,'data','dcfd','velocity')))
    for category in folderlist:
        os.makedirs(os.path.join(ROOT,'output','heatmap',category),exist_ok=True)
        epochs,steps = 0,[20,30,50,100]
        for step in steps:
            train(category,epochs,step)
            epochs = epochs+step
            test(category,epochs)
