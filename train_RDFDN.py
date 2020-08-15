from RDFDN import R_DFDN, Loss
from argparse import Namespace
import sys
import argparse
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import pickle as pkl
import torch.nn.functional as F
from utils import correlation_reg
import glob
import tqdm
import torch.utils.data as utils
import json
from data import get_dataset
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Predicting with adgm')

# Directories
parser.add_argument('--data', type=str, default='datasets/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='default/',
                    help='root dir path to save the log and the final model')
parser.add_argument('--save_dir', type=str, default='0/',
                    help='dir path (inside root_dir) to save the log and the final model')
parser.add_argument('--load_dir', type=str, default='',
                    help='dir path (inside root_dir) to load model from')
parser.add_argument('--use_tfboard', type=bool, default=True,
                    help='use tensorboard')

# Baseline (correlation based) method
parser.add_argument('--beta', type=float, default=0.1,
                    help='coefficient for correlation based penalty')

# adaptive batch norm
parser.add_argument('--bn_eval', action='store_true',
                    help='adapt BN stats during eval')

# dataset and architecture
parser.add_argument('--dataset', type=str, default='fgbg_cmnist_cpr0.5-0.5',
                    help='dataset name')
parser.add_argument('--arch', type=str, default='resnet',
                    help='arch name (resnet,cnn)')

# Optimization hyper-parameters
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--bs', type=int, default=128, metavar='N',
                    help='batch size')

parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate ')

parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limit')

parser.add_argument('--init', type=str, default="he")

parser.add_argument('--wdecay', type=float, default=1e-4,
                    help='weight decay applied to all weights')

# meta specifications
parser.add_argument('--validation', action='store_true',
                    help='Compute accuracy on validation set at each epoch')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])

args = parser.parse_args()
args.root_dir = os.path.join('runs/', args.root_dir)
args.save_dir = os.path.join(args.root_dir, args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
log_dir = args.save_dir + '/'

with open(args.save_dir + '/config.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

with open(args.save_dir + '/log.txt', 'w') as f:
    f.write('python ' + ' '.join(s for s in sys.argv) + '\n')

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu)

if args.use_tfboard:
    writer = SummaryWriter("final_result")

# Set the random seed manually for reproducibility.
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
print('==> Preparing data..')
trainloader_s, validloader_s, testloader_s, nb_classes_s, dim_inp_s = get_dataset(args)
args.dataset = 'svhn'
trainloader_svhn, validloader_svhn, testloader_svhn, nb_classes_svhn, dim_inp_svhn = get_dataset(args)
args.dataset = 'mnist'
trainloader_mnist, validloader_mnist, testloader_mnist, nb_classes_mnist, dim_inp_mnist = get_dataset(args)
args.dataset = 'mnistm'
trainloader_mnistm, validloader_mnistm, testloader_mnistm, nb_classes_mnistm, dim_inp_mnistm = get_dataset(args)



###############################################################################
# Build the model
###############################################################################
model = R_DFDN()

params = list(model.parameters())
model = torch.nn.DataParallel(model, device_ids=range(len(args.gpu)))
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)


nb = 0
if args.init == 'he':
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nb += 1
            # print ('Update init of ', m)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            # print ('Update init of ', m)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
print('Number of Conv layers: ', (nb))

if use_cuda:
    model.cuda()

total_params = sum(np.prod(x.size()) if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)
with open(args.save_dir + '/log_adgm.txt', 'a') as f:
    f.write(str(args) + ',total_params=' + str(total_params) + '\n')

loss = Loss()

###############################################################################
# Training/Testing code
###############################################################################

tot_iters = len(trainloader_s)


def test(loader, model, save=False, is_generation=True):
    global best_acc, args
    model.eval()
    correct, total = 0, 0
    tot_iters = len(loader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs, targets = next(iter(loader))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            class_pred = model(inputs, inputs, is_generation=is_generation, is_train=False)

            _, predicted = torch.max(nn.Softmax(dim=1)(class_pred).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    acc = 100. * float(correct) / float(total)
    return acc

best_acc = [0, 0, 0, 0,0]
for epoch in range(args.epochs):
    # adjust_lr(init_lr=1e-3, optimizer=optimizer, epoch=epoch, total_epo=args.epochs)
    model.train()
    correct_gen, correct_adp = 0, 0
    total = 0
    # totol_class_loss,totol_class_loss2,totol_regulazation_loss,totol_l=0,0,0,0
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs_s, targets_s = next(iter(trainloader_s))
        inputs_t, targets_t = next(iter(trainloader_mnistm))
        if use_cuda:
            inputs_s, targets_s = inputs_s.cuda(), targets_s.cuda()
            inputs_t, targets_t = inputs_t.cuda(), targets_t.cuda()
        inputs_s = Variable(inputs_s)
        inputs_t = Variable(inputs_t)
        hid, source_pred,tar_pred,source_inv_diff,tar_inv_diff,tar_cha_diff,tar_rec,source_inv,tar_inv= model(
            sourse_input=inputs_s,tar_input=inputs_t, is_generation=False, is_train=True)
        hid_1_loss, s_class_loss, t_class_loss, t_recognition_loss,fa_loss,t_diff_loss,totol_loss = loss(hid = hid,
                targets = targets_s,source_pred = source_pred,tar_pred= tar_pred,source_inv_diff = source_inv_diff,tar_inv_diff = tar_inv_diff,
                tar_cha_diff = tar_cha_diff,tar_rec = tar_rec,tar_img = inputs_t,source_inv = source_inv,tar_inv = tar_inv,
                weight_list=[0.1,1,1,0.1,0.1,0.1])
        totol_loss.backward()
        _, predicted_gen = torch.max(nn.Softmax(dim=1)(source_pred).data, 1)
        _, predicted_adp = torch.max(nn.Softmax(dim=1)(tar_pred).data, 1)
        total += targets_s.size(0)
        correct_gen += predicted_gen.eq(targets_s.data).cpu().sum()
        correct_adp += predicted_adp.eq(targets_s.data).cpu().sum()
        optimizer.step()
        optimizer.zero_grad()
        if args.use_tfboard:
            writer.add_scalar("Loss/hid_1_loss", hid_1_loss.data.cpu(), global_step=tot_iters * epoch + batch_idx)
            writer.add_scalar("Loss/s_class_loss", s_class_loss.data.cpu(),
                              global_step=tot_iters * epoch + batch_idx)
            writer.add_scalar("Loss/t_class_loss", t_class_loss.data.cpu(), global_step=tot_iters * epoch + batch_idx)
            writer.add_scalar("Loss/t_recognition_loss", t_recognition_loss.data.cpu(), global_step=tot_iters * epoch + batch_idx)
            writer.add_scalar("Loss/fa_loss", fa_loss.data.cpu(), global_step=tot_iters * epoch + batch_idx)
            writer.add_scalar("Loss/t_diff_loss", t_diff_loss.data.cpu(), global_step=tot_iters * epoch + batch_idx)
            writer.add_scalar("Loss/totol_loss", totol_loss.data.cpu(), global_step=tot_iters * epoch + batch_idx)

    acc_gen = 100. * correct_gen / total
    acc_adp = 100. * correct_adp / total
    print(f"|| Epoch: {epoch} || train_gen_acc: {acc_gen} || train_adp_acc: {acc_adp} || lr:{optimizer.state_dict()['param_groups'][0]['lr']}")
    # sche.step()
    if epoch % 1 == 0:
        all_acc = []
        svhn_test_acc = test(testloader_svhn, model, is_generation=True)
        all_acc.append(svhn_test_acc)
        minists_test_acc = test(testloader_s, model, is_generation=True)
        all_acc.append(minists_test_acc)
        mnist_test_acc = test(testloader_mnist, model, is_generation=True)
        all_acc.append(mnist_test_acc)
        mnistm_test_acc = test(testloader_mnistm, model, is_generation=False)
        all_acc.append(mnistm_test_acc)
        usps_test_acc = test(testloader_usps, model, is_generation=True)
        all_acc.append(usps_test_acc)

        for i in range(len(all_acc)):
            if all_acc[i] > best_acc[i]:
                best_acc[i] = all_acc[i]
        print(
            f"Epoch: {epoch} ######## best_sv{best_acc[0]}######## best_s{best_acc[1]}######## best_minist{best_acc[2]}"
            f"######## best_ministm{best_acc[3]} ######## best_usps{best_acc[4]}")
        writer.add_scalar("Eval/minists_test_acc", minists_test_acc, global_step=epoch)
        writer.add_scalar("Eval/svhn_test_acc", svhn_test_acc, global_step=epoch)
        writer.add_scalar("Eval/mnist_test_acc", mnist_test_acc, global_step=epoch)
        writer.add_scalar("Eval/mnistm_test_acc", mnistm_test_acc, global_step=epoch)
        writer.add_scalar("Eval/usps_test_acc", usps_test_acc, global_step=epoch)