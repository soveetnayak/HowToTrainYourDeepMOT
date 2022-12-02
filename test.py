from model import Deep_Hungarian_net
from os.path import realpath, dirname
import torch
from torch.utils import data
import os
import numpy as np
torch.set_grad_enabled(False)
from Data import Dataset
import torch.nn.functional as F



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


def get_pred(tag_score):
    y_pred = torch.zeros(tag_score.size(), layout=tag_score.layout).cuda().float()

    for batch in range(tag_score.size(0)):
        for row in range(tag_score.size(1)):
            val, ind = tag_score[batch, row].max(0)
            if float(val) > 0.5:
                y_pred[batch, row, int(ind)] = 1.0
    
    return y_pred


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



def print_results(curr_corrects, elem_count, curr_tp, curr_fp, curr_fn):
    epoch_acc = curr_corrects.double() / elem_count
    tp = curr_tp.double()
    fp = curr_fp.double()
    fn = curr_fn.double()
    tn = 1.0*elem_count - tp - fp - fn
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)

    J = r + (tn/(tn+fp)) - 1

    epoch_f1 = 2 * p * r / (p + r + 1e-9)

    print('fn: ', fn.item())
    print('fp: ', fp.item())
    print('tp: ', tp.item())
    print('tn: ', tn.item())
    print('total elements: ', elem_count)
    print('precision: ', p.item())
    print('recall: ', r.item())
    print('Youden value:', J.item())
    print('f1_score: ', epoch_f1.item())
    print('weighted acc: ', np.mean(np.array(accuracy)) * 100)




model = Deep_Hungarian_net(1, 256, 1)
checkpoint = torch.load("model_best_checkpoint.pth.tar")
is_cuda = True
model.eval()
if is_cuda:
    model.cuda()

# load data #

test_ds = Dataset("./DHN_data/",train=False)
test_dl = data.DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)

model.load_state_dict(checkpoint['model'])


accuracy = []
curr_corrects = 0
curr_tp = 0
curr_fp = 0
curr_fn = 0
elem_count = 0.0
lines_count = 0.0
sum_many_ones = 0.0
sum_all_zeros = 0.0
tot_mat = 0.0
wr_mat1 = 0.0
wr_mat2 = 0.0


for data, target in test_dl:
    many_ones = 0
    not_all_zeros = 0
    curr_wr_mat1 = 0
    curr_wr_mat2 = 0

    if is_cuda:
        data = data.squeeze(0).cuda()
        target = target.squeeze(0).cuda()
    else:
        data = data.squeeze(0)
        target = target.squeeze(0)

    elem_count += data.shape[0]*data.shape[1]*data.shape[2]

    model.hidden_row = model.init_hidden(data.size(0))
    model.hidden_col = model.init_hidden(data.size(0))

    tag_scores = model(data).detach()
    y_pred = get_pred(tag_scores)

    # weighted accuracy
    weight = get_weight(target,train=False)
    y_pred,acc = weighted_acc(tag_scores, target, weight)
    accuracy.append(acc)

    # TP, TN, FP, FN, F1_score
    target = target.float()
    curr_corrects += torch.sum(y_pred == target.data).double()
    curr_tp += torch.sum((y_pred == target.data)[target.data == 1]).double()
    curr_fp += torch.sum((y_pred != target.data)[y_pred.data == 1]).double()
    curr_fn += torch.sum((y_pred != target.data)[y_pred.data == 0]).double()
    

    # prepare for fn, fp


    print_results(curr_corrects, elem_count, curr_tp, curr_fp, curr_fn)
    


    



