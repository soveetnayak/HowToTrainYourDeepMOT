from model import Deep_Hungarian_net as Munkrs
import torch.optim as optim
from Data import Dataset as RealData
import torch
import numpy as np
# try:
#     from tensorboardX import SummaryWriter
# except:
#     from torch.utils.tensorboard import SummaryWriter
from utils import adjust_learning_rate, eval_acc
# from loss.DICE import soft_dice_loss
import shutil
import os
import argparse
from os.path import realpath, dirname

def weighted_binary_focal_entropy(output, target, weights=None, gamma=2):
    # output = torch.clamp(output, min=1e-8, max=1 - 1e-8)
    if weights is not None:
        assert weights.size(1) == 2

        # weight is of shape [batch,2, 1, 1]
        # weight[:,1] is for positive class, label = 1
        # weight[:,0] is for negative class, label = 0

        loss = (torch.pow(1.0-output, gamma)*torch.mul(target, weights[:, 1]) * torch.log(output+1e-8)) + \
               (torch.mul((1.0 - target), weights[:, 0]) * torch.log(1.0 - output+1e-8)*torch.pow(output, gamma))
    else:
        loss = target * torch.log(output+1e-8) + (1 - target) * torch.log(1 - output+1e-8)

    return torch.neg(torch.mean(loss))







# model #
model = Munkrs(1,256, 1)
model = model.train()

# optimizer #
optimizer = optim.RMSprop(model.parameters(), lr=0.00003)

# load and finetune model #
starting_epoch = 0




# data loaders #
train_dataset = RealData("./DHN_data/", train=True)
val_dataset = RealData("./DHN_data/", train=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
val_dataloader =torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

print('val length in #batches: ', len(val_dataloader.dataset))
print('train length in #batches: ', len(train_dataloader.dataset))



is_best = False
val_loss = None

starting_iterations = 0
iteration = 0
iteration += starting_iterations
if iteration > 0 and iteration % len(train_dataloader.dataset) == 0:
    starting_epoch += 1
for epoch in range(max(0, starting_epoch), 1):

    for Dt, target in train_dataloader:

        model = model.train()
        Dt = Dt.squeeze(0)
        target = target.squeeze(0)
        # if args.is_cuda:
        #     Dt = Dt.cuda()
        #     target = target.cuda()

        # after each sequence/matrix Dt, we init new hidden states
        model.hidden_row = model.init_hidden(Dt.size(0))
        model.hidden_col = model.init_hidden(Dt.size(0))

        # input to model
        tag_scores = model(Dt)
        # num_positive = how many labels = 1
        num_positive = target.detach().clone().view(target.size(0), -1).sum(dim=1).unsqueeze(1)
        weight2negative = num_positive.float()/(target.size(1)*target.size(2))
        # case all zeros, then weight2negative = 1.0
        weight2negative.masked_fill_((weight2negative == 0.0), 10)  # 10 is just a symbolic value representing 1.0
        # case all ones, then weight2negative = 0.0
        weight2negative.masked_fill_((weight2negative == 1.0), 0.0)
        # change all fake values 10 to their desired value 1.0
        weight2negative.masked_fill_((weight2negative == 10), 1.0)
        weight = torch.cat([weight2negative, 1.0 - weight2negative], dim=1)
        weight = weight.view(-1, 2, 1, 1).contiguous()
        # if args.is_cuda:
        #     weight = weight.cuda()

        loss = 10.0 * weighted_binary_focal_entropy(tag_scores, target.float(), weights=weight)

        # clean gradients & back propagation
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust learning weight
        old_lr = adjust_learning_rate(optimizer, iteration, 0.00003)

        # # show loss
        # train_writer.add_scalar('Loss', loss.item(), iteration)
        # if val_loss is None:
        #     val_loss = loss.item()
        # val_writer.add_scalar('Loss', val_loss, iteration)

        if iteration % 1 == 0:
            print('Epoch: [{}][{}/{}]\tLoss {:.4f}'.format(epoch, iteration % len(train_dataloader.dataset),
                                                            len(train_dataloader.dataset), loss.item()))

        if iteration % 1 == 0:
            model = model.eval()
            test_loss = []
            acc = []
            test_j = []
            test_p = []
            test_r = []
            # val = random.sample(valset, 50)
            for test_num, (data, target) in enumerate(val_dataloader):
                data = data.squeeze(0)
                target = target.squeeze(0)
                if test_num == 50:
                    break
                # if args.is_cuda:
                #     data = data.cuda()
                #     target = target.cuda()
                # after each sequence/matrix Dt, we init new hidden states
                model.hidden_row = model.init_hidden(data.size(0))
                model.hidden_col = model.init_hidden(data.size(0))

                # input to model
                tag_scores = model(data)

                num_positive = target.data.view(target.size(0), -1).sum(dim=1).unsqueeze(1)
                # print num_positive
                weight2negative = num_positive.float() / (target.size(1) * target.size(2))
                # case all zeros
                weight2negative.masked_fill_((weight2negative == 0), 10)  # 10 is just a symbolic value representing 1.0
                # case all ones
                weight2negative.masked_fill_((weight2negative == 1), 0.0)
                weight2negative.masked_fill_((weight2negative == 10), 1.0)  # change all 100 to their true value 1.0
                weight = torch.cat([weight2negative, 1.0 - weight2negative], dim=1)
                # print weight
                weight = weight.view(-1, 2, 1, 1).contiguous()
                # print weight
                # if args.is_cuda:
                #     weight = weight.cuda()

                loss = 10.0 * weighted_binary_focal_entropy(tag_scores, target.float(), weights=weight)

                test_loss.append(float(loss.item()))
                # scores = F.sigmoid(tag_scores)
                predicted, curr_acc = eval_acc(tag_scores.float().detach(), target.float().detach(), weight.detach())
                acc.append(curr_acc)

                # calculate J value
                tp = torch.sum((predicted == target.float().detach())[target.data == 1.0]).double()
                fp = torch.sum((predicted != target.float().detach())[predicted.data == 1.0]).double()
                fn = torch.sum((predicted != target.float().detach())[predicted.data == 0.0]).double()

                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                test_p.append(p.item())
                test_r.append(r.item())

            print('Epoch: [{}][{}/{}]\tLoss {:.4f}\tweighted Accuracy {:.2f} %'.format(epoch, iteration % len(train_dataloader.dataset),
                                                                                        len(train_dataloader.dataset),
                                                                                        np.mean(np.array(test_loss)),
                                                                                        100.0*np.mean(np.array(acc))))

            print('P {:.2f}% \t R {:.2f}%'.format(100.0*np.mean(np.array(test_p)), 100.0*np.mean(np.array(test_r))))

            # show loss and accuracy
            val_loss = np.mean(np.array(test_loss))
            # val_writer.add_scalar('Weighted Accuracy', np.mean(np.array(acc)), iteration)

            # val_writer.add_scalar('recall', np.mean(np.array(test_r)), iteration)

            # val_writer.add_scalar('precision', np.mean(np.array(test_p)), iteration)
            # val_writer.add_scalar('Loss', val_loss, iteration)

            if old_acc < np.mean(np.array(acc)):
                old_acc = np.mean(np.array(acc)) + 0.0
                is_best = True

            # save checkpoints
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'iters': iteration,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': np.mean(np.array(acc)),
            #     'optimizer': optimizer.state_dict(),
            # }, is_best, os.path.join(args.save_path, args.save_name) + "/DHN_" + str(epoch+1) + "_.pth.tar",
            # best_model_name=os.path.join(args.save_path, args.save_name) + "/DHN_" + str(epoch+1) + "_best.pth.tar")

            # is_best = False
        iteration += 1