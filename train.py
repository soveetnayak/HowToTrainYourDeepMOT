from model import Deep_Hungarian_net 
import torch.optim as optim
from Data import Dataset
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import argparse
from os.path import realpath, dirname


def focal_loss(y_pred, y_target, weights=None, gamma=2):
    # y_pred = torch.sigmoid(y_pred)
    y_pred = torch.clamp(y_pred, 1e-8, 1 - 1e-8)
    y_target = y_target.float()
    
    if weights is not None:
        bin_foc_l = (torch.pow(1.0-y_pred, gamma)* torch.log(y_pred+1e-8))*torch.mul(weights[:, 1], y_target)
        bin_foc_l += (torch.pow(y_pred, gamma)* torch.log(1.0-y_pred+1e-8))*torch.mul(weights[:, 0], 1.0-y_target)
        ret_val = torch.mean(bin_foc_l)
        ret_val = torch.neg(ret_val)
    else:
        bin_foc_l = torch.log(y_pred+1e-8)*y_target 
        bin_foc_l += (1-y_target)*torch.log(1-y_pred+1e-8)
        ret_val = torch.mean(bin_foc_l)
        ret_val = torch.neg(ret_val)

    return ret_val


def save_checkpt(model, epoch, optimizer, best_acc):
    state = {
        'epoch' : epoch + 1,
        'model' : model.state_dict(),
        'best_accuracy' : best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, 'model_best_checkpoint_epoch' +str(epoch)+'.pth.tar')

def weighted_acc(tag_score, y_target, weight):
    accuracy = []
    y_pred = torch.zeros(tag_score.size(), layout=tag_score.layout).cuda().float()

    for batch in range(tag_score.size(0)):
        for row in range(tag_score.size(1)):
            val, ind = tag_score[batch, row].max(0)
            if float(val) > 0.5:
                y_pred[batch, row, int(ind)] = 1.0
        
        tot_p = float(y_target[batch, :, :].sum()) # adding ones i.e. total positives
        size = y_target.size(1)*y_target.size(2)
        tot_n = float(size-tot_p)
        
        tp = float((torch.mul(y_pred[batch, :, :],y_target[batch, :, :])).sum())
        tn = float((torch.mul(torch.sub(1,y_pred[batch, :, :]),torch.sub(1,y_target[batch, :, :]))).sum())

        acc = (1.0*(tp * float(weight[batch, 1, 0, 0]) + tn * float(weight[batch, 0, 0, 0]))/
                   (tot_p * float(weight[batch, 1, 0, 0]) + tot_n * float(weight[batch, 0, 0, 0])))
        accuracy.append(acc)
    
    return y_pred,torch.mean(torch.tensor(accuracy))
          
def get_weight(target,train=True):
    size_tar = target.size(1)*target.size(2)
    if train:    
        tot_p = target.detach().clone()
    else:
        tot_p = target.data

    tot_p = tot_p.view(target.size(0),-1).sum(dim = 1).unsqueeze(1)
    w2 = tot_p.float()/size_tar
    # case all zeros, then weight2negative = 1.0
    w2.masked_fill_((w2 == 0.0), 10)  # 10 is just a symbolic value representing 1.0
    # case all ones, then weight2negative = 0.0
    w2.masked_fill_((w2 == 1.0), 0.0)
    # change all fake values 10 to their desired value 1.0
    w2.masked_fill_((w2 == 10), 1.0)

    w1 = 1.0 - w2
    weight = torch.cat([w2,w1],dim=1)
    weight = weight.view(-1,2,1,1).contiguous()
    if is_cuda:
        weight = weight.cuda()
    return weight



learning_rate = 0.0003
last_acc = 0.0
curr_path = realpath(dirname(__file__))           
    


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True

# model #
model = Deep_Hungarian_net(element_size=1,hidden_size=256, targe_size=1)
model = model.train()
model = model.cuda()

# optimizer #
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)


# TensorboardX logs #

train_writer = SummaryWriter(os.path.join(curr_path, 'log/', 'train'))
val_writer = SummaryWriter(os.path.join(curr_path, 'log/', 'test'))


# data loaders #
train_dataset = Dataset("./DHN_data/", train=True)
val_dataset = Dataset("./DHN_data/", train=False)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_dl =torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

print(f"Train dataset length = {len(train_dl.dataset)} no. of batches")
print(f"Validation dataset length = {len(val_dl.dataset)} no. of batches")

# training #
is_cuda = True # set if your device supports cuda
is_best = False
val_loss = None
iterations =0
for epoch in range(3):
    for Dist,target in train_dl:

        model = model.train()
        Dist = Dist.squeeze(0)
        target = target.squeeze(0)
        if is_cuda:
            Dist = Dist.cuda()
            target = target.cuda()
        
        weight = get_weight(target)
        # after each sequence/matrix Dt, we init new hidden states
        model.hidden_row = model.init_hidden(Dist.size(0))
        model.hidden_col = model.init_hidden(Dist.size(0))

        # forward pass
        tag_score = model(Dist)
    
        #loss
        loss = focal_loss(tag_score, target.float(), weights=weight)
        train_writer.add_scalar('Loss/train', loss.item(), iterations)


        # backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy
        if iterations % 10 == 0:
            print('Epoch: [{}][{}/{}]\tLoss {:.4f}'.format(epoch, iterations % len(train_dl.dataset),len(train_dl.dataset), loss.item()))
        
        if iterations % 20 == 0:
            model = model.eval()
            val_acc = []
            val_loss = []
            val_p = []
            val_r = []

            for val_i,(data,target) in enumerate(val_dl):
                data = data.squeeze(0)
                target = target.squeeze(0)
                if is_cuda:
                    data = data.cuda()
                    target = target.cuda()
                if val_i == 50:
                    break
                weight = get_weight(target,train=False)
                model.hidden_row = model.init_hidden(data.size(0))
                model.hidden_col = model.init_hidden(data.size(0))
                tag_score = model(data)

                loss = 10.0 * focal_loss(tag_score, target.float(), weights=weight)
                val_loss.append(loss.item())
                # val_writer.add_scalar('Loss/val', loss.item(), iterations)
                y_pred,acc = weighted_acc(tag_score.float().detach(),target.float().detach(),weight.detach())
                val_acc.append(acc)
                # val_writer.add_scalar('Accuracy/val', acc, iterations)

                # calaculating jacard precesion
                tp = torch.sum((y_pred == target.float().detach())[target.data == 1.0]).double()
                fp = torch.sum((y_pred != target.float().detach())[y_pred.data == 1.0]).double()
                fn = torch.sum((y_pred != target.float().detach())[y_pred.data == 0.0]).double()


                p = tp/(tp+fp + 1e-9)
                r = tp/(tp+fn + 1e-9)
                val_p.append(p.item())
                val_r.append(r.item())

            print('Epoch: [{}][{}/{}]\tLoss {:.4f}\tWeighted_Acc {:.4f}\tVal P {:.4f}\tVal R {:.4f}'.format(epoch, iterations % len(train_dl.dataset),len(train_dl.dataset), np.mean(np.array(val_loss)),100.0*np.mean(np.array(val_acc)),np.mean(np.array(val_p)),np.mean(np.array(val_r))))

            val_writer.add_scalar('Loss', np.mean(np.array(val_loss)), iterations)
            val_writer.add_scalar('Weighted Accuracy', np.mean(np.array(val_acc)), iterations)
            val_writer.add_scalar('Precision', np.mean(np.array(val_p)), iterations)
            val_writer.add_scalar('Recall', np.mean(np.array(val_r)), iterations)

            if np.mean(np.array(val_acc)) > last_acc:
                is_best = True
                last_acc = np.mean(np.array(val_acc)) + 0.0
            
            if is_best:
                best_acc =np.mean(np.array(val_acc))
                save_checkpt(model=model,epoch=epoch,optimizer=optimizer,best_acc=best_acc)
        iterations += 1





