
import logging
import os
import sys
from transform3D import *
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval_dsnt import eval_net
from unet import UNet_dsnt
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_dnst import BasicDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
from scipy.io import loadmat, savemat
from torch.nn import init
import dsntnn
import visdom
import json
from INFO import *
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)



def train_net(net,
              device,
              epochs=100,
              batch_size=6,
              lr=0.01,
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              sigma_t = 1.0):
    train_datasetlist = []
    for dir_train in dir_trainlist:
        train_datasetlist.insert(-1,BasicDataset(dir_train, dir_train, img_scale))
    train_dataset = ConcatDataset([train_datasetlist[0],train_datasetlist[1],train_datasetlist[2]])
    val_dataset = BasicDataset(dir_val, dir_val, img_scale)
    n_train = int(len(train_dataset))
    n_val = int(len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    viz = visdom.Visdom(env='main')
    optimizer = optim.Adam(net.parameters(), lr=lr,)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_descent_stepsize, gamma=lr_descent_rate)
    trainloss = np.array([])
    valloss = np.array([])
    iter = 0

    for epoch in range(epochs):
        net.train()
        epoch_validation = 0
        scheduler.step()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                iter = iter+1
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)
                coords, heatmap = net(imgs)
                loss1 = dsntnn.euclidean_losses(coords,true_masks)
                loss2 = dsntnn.js_reg_losses(heatmap,true_masks,sigma_t=sigma_t)
                loss = dsntnn.average_loss(loss1 + loss2)


                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                trainloss = np.append(trainloss,loss.item())
                viz.line(X=np.arange(0,iter),Y=trainloss,win='train_loss',opts={'title':'train_loss'})

                if (epoch + 1) % val_epoch == 0 and epoch_validation == 0 and epoch !=0 :
                    val_score = eval_net(net, val_loader, device)
                    if epoch == val_epoch-1:
                        best_val = val_score
                        torch.save(net.state_dict(),
                                   dir_checkpoint + f'/best_model.pth')
                    if (val_score) < best_val:
                        best_val = val_score
                        torch.save(net.state_dict(),
                                   dir_checkpoint + f'/best_model.pth')
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation loss1: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                    epoch_validation += 1
                    torch.save(net.state_dict(),
                               dir_checkpoint + f'/CP_epoch{epoch + 1}.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                    valloss = np.append(valloss,val_score)
                    viz.line(X=np.arange(0, (epoch + 1)//val_epoch), Y = valloss, win='val_loss', opts={'title': 'val_loss'})
                    if (epoch+1) == epochs:
                        win_data_train = viz.get_window_data('train_loss', 'main')
                        pre_data_train = json.loads(win_data_train)
                        y_train = pre_data_train['content']['data'][0]['y']
                        win_data_val = viz.get_window_data('val_loss', 'main')
                        pre_data_val = json.loads(win_data_val)
                        y_val = pre_data_val['content']['data'][0]['y']
                        savemat(dir_checkpoint + '/' +  'lossline.mat', {'trainloss': y_train, 'valloss': y_val})

        writer.add_scalar('Loss/train_epoch', epoch_loss/11, epoch)
    writer.close()





if __name__ == '__main__':
    for i in range(len(foldlist)):
        foldlist_train = foldlist.copy()
        foldlist_train.remove(foldlist[i])
        dir_trainlist = []
        for j in range(len(foldlist_train)):
            dir_trainlist.insert(-1,dir_file + '/' + foldlist_train[j])
        dir_val = dir_file + '/' + foldlist[i]
        task = tasklist[i]
        dir_savepath = f'./checkpoints'
        dir_checkpoint = dir_savepath + '/' + taskname + '/' + task
        if not os.path.isdir(dir_checkpoint):
            os.makedirs(dir_checkpoint)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f'Test fold: \t{foldlist[i]}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net = UNet_dsnt(n_channels=n_channels, n_classes=n_classes)
        logging.info(f'Network:\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)\n'
                     f'load parameters from INFO.py\n'
                    )
        f = open(os.path.join(dir_checkpoint,'training_parameters.txt'), mode = 'w')
        f.write(f'Crossvalidation: \t{crossvalidation}\n'
                f'foldname:\t{foldlist[i]}\n'
                f'random seed: \t{seed}\n'
                f'input channels \t{net.n_channels}\n'
                f'output channels \t{net.n_classes} (classes)\n'
                f'task: \t{task}\n'
                f'dir_checkpoint: \t{dir_checkpoint}\n'
                f'epochs: \t{epochs}\n'
                f'batch_size: \t{batch_size}\n'
                f'lr: \t{lr}\n'
                f'device: \t{device}\n'
                f'img_scale: \t{img_scale}\n'
                f'lr_descent_stepsize: \t{lr_descent_stepsize}\n'
                f'lr_descent_rate: \t{lr_descent_rate}\n'
                f'val_epoch: \t{val_epoch}\n'
                f'sigma_t: \t{sigma_t}\n'
                )
        f.close()

        init_weights(net, 'kaiming')
        net.to(device=device)
        try:
            train_net(net=net,
                      epochs= epochs,  # args.epochs,
                      batch_size= batch_size,
                      lr= lr,  # args.lr,
                      device=device,
                      img_scale=img_scale,  # args.scale,
                      val_percent=0.1,
                      sigma_t = sigma_t)  # args.val / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
