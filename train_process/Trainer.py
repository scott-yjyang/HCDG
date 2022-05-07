from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytz
from tensorboardX import SummaryWriter
import math
import tqdm
import socket
from utils.metrics import *
from utils.Utils import *
from .fourier import fourier_amplitude_mix

import copy
from torch.distributions.uniform import Uniform
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
softmax = torch.nn.Softmax(-1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(object):

    def __init__(self, cuda, s_model, t_model, lr, val_loader, train_loader, target_loader, out, max_epoch, optim, stop_epoch=None,
                 lr_decrease_rate=0.1, interval_validate=None, batch_size=8, gam=200):
        self.cuda = cuda
        self.S_model = s_model
        self.T_model = t_model

        self.optim = optim
        self.lr = lr
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.target_loader = target_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(5)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'valid/cup_dice',
            'valid/disc_dice',
            'valid/loss_CE',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.total_iters = self.stop_epoch * len(self.train_loader)
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1


        self.temperature = 10.0
        self.gamma = gam
        self.m = 0.9995
        self.w = 1


    def validate(self):
        training = self.S_model.training
        self.S_model.eval()

        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        metrics = []
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):

                image = sample['image']
                label = sample['label']

                data = image.cuda()
                target_map = label.cuda()

                with torch.no_grad():
                    predictions = self.S_model(data, classmates=False)

                loss_seg = bceloss(torch.sigmoid(predictions), target_map)
                #loss_cls = mseloss(softmax(domain_predict), domain_code)
                loss_data = loss_seg.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                dice_cup, dice_disc = dice_coeff_2label(np.asarray(torch.sigmoid(predictions.data.cpu())) > 0.75, target_map)
                val_cup_dice += dice_cup
                val_disc_dice += dice_disc
            val_loss /= len(self.val_loader)
            val_cup_dice /= len(self.val_loader)
            val_disc_dice /= len(self.val_loader)
            metrics.append((val_loss, val_cup_dice, val_disc_dice))
            self.writer.add_scalar('val_data/loss', val_loss, self.epoch * (len(self.train_loader)))
            #self.writer.add_scalar('val_data/loss_cls', loss_cls.data.item(), self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.train_loader)))

            mean_dice = val_cup_dice + val_disc_dice
            record_str = "\n[Epoch: {:d}] val CUP dice: {:f}, val DISC dice: {:f}, val Loss: {:.5f}".format(self.epoch + 1, val_cup_dice, val_disc_dice, val_loss)

            print(record_str)
            is_best = mean_dice > self.best_mean_dice
            '''
            log_headers = [
                self.epoch+1,
                val_cup_dice,
                val_disc_dice,
                val_loss,
            ]
            '''

            with open(osp.join(self.out, 'log.txt'), 'a') as f:
                f.write(record_str)

            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.S_model.__class__.__name__,
                    'optim_state_dict': self.optim.state_dict(),
                    'model_state_dict': self.S_model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d_best.pth.tar' % self.best_epoch))
            else:
                if (self.epoch + 1) % 20 == 0 or (self.epoch + 1) % 50 == 0:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.S_model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.S_model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim),
                        'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))

            if training:
                self.S_model.train()

    def train_epoch(self):
        self.S_model.train()
        #self.T_model.train()
        self.running_seg_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        self.running_cls_loss = 0
        targetloader_iter = iter(self.target_loader)
        start_time = timeit.default_timer()

        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % (self.epoch+1), ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            assert self.S_model.training
            self.optim.zero_grad()


            image = None
            label = None
            target_boundary = None
            for domain in sample:
                if image is None:
                    image = domain['image']
                    label = domain['label']
                    target_boundary = domain['boundary']

                else:
                    image = torch.cat([image, domain['image']], 0)
                    label = torch.cat([label, domain['label']], 0)
                    target_boundary = torch.cat([target_boundary, domain['boundary']], 0)


            # domainup
            image_t = None
            label_t = None
            boundary_t = None
            image_fs = None
            image_ft = None
            with torch.no_grad():
                for idx in range(image.shape[0]):
                    ori_img, lab = image[idx], label[idx]

                    ori_img = torch.unsqueeze(ori_img, dim=0)
                    lab = torch.unsqueeze(lab, dim=0)

                    max_loss = 0
                    max_trg = None
                    max_label = None
                    max_boundary = None
                    max_aug_src = None
                    max_aug_trg = None
                    for i in range(self.w):
                        try:
                            trg_sample = next(targetloader_iter)
                        except:
                            del targetloader_iter
                            targetloader_iter = iter(self.target_loader)
                            trg_sample = next(targetloader_iter)
                        for domain in trg_sample:

                            src_in_trg, trg_in_src = fourier_amplitude_mix(ori_img, domain['image'], L=1.0)
                            aug_src = torch.cat([src_in_trg, trg_in_src], 0).cuda()
                            aug_lab = torch.cat([lab, domain['label']], 0).cuda()
                            sa = self.S_model(aug_src, classmates=False)
                            #ta = self.S_model(trg_in_src)
                            loss_sa = bceloss(torch.sigmoid(sa), aug_lab)
                            #loss_ta = bceloss(torch.sigmoid(ta), domain['label'])
                            loss = loss_sa
                            if loss > max_loss:
                                max_loss = loss
                                max_trg = domain['image']
                                max_label = domain['label']
                                max_boundary = domain['boundary']
                                max_aug_src = src_in_trg
                                max_aug_trg = trg_in_src

                    if image_t is None:
                        image_t = max_trg
                        label_t = max_label
                        boundary_t = max_boundary
                    else:
                        image_t = torch.cat([image_t, max_trg], 0)
                        label_t = torch.cat([label_t, max_label], 0)
                        boundary_t = torch.cat([boundary_t, max_boundary], 0)

                    if image_fs is None:
                        image_fs = max_aug_src
                    else:
                        image_fs = torch.cat([image_fs, max_aug_src], 0)
                    if image_ft is None:
                        image_ft = max_aug_trg
                    else:
                        image_ft = torch.cat([image_ft, max_aug_trg], 0)

            image = torch.cat([image, image_t], 0)
            image_fs = torch.cat([image_fs, image_ft], 0)
            label = torch.cat([label, label_t], 0)
            target_boundary = torch.cat([target_boundary, boundary_t], 0)


            gamma = self.get_current_consistency_weight(self.epoch, self.gamma, 5)
            image = image.cuda()
            image_fs = image_fs.cuda()
            target_map = label.cuda()
            target_boundary = target_boundary.cuda()

            so, bd, bn = self.S_model(image, classmates=True)
            sa = self.S_model(image_fs, classmates=False)


            loss_so = bceloss(torch.sigmoid(so), target_map)
            loss_sa = bceloss(torch.sigmoid(sa), target_map)
            loss_bd = mseloss(torch.sigmoid(bd), target_boundary)
            loss_bn = mseloss(torch.sigmoid(bn), target_boundary)

            #loss_cls = loss_seg
            loss_seg = loss_so + loss_sa + loss_bd + loss_bn

            to = self.T_model(image)
            ta = self.T_model(image_fs)
            loss_o2a = F.kl_div(F.log_softmax(so/self.temperature, dim=-1), F.softmax(ta/self.temperature, dim=-1), reduction='mean')
            loss_a2o = F.kl_div(F.log_softmax(sa/self.temperature, dim=-1), F.softmax(to/self.temperature, dim=-1), reduction='mean')

            # convert boundary to mask
            bd_to_mask = torch.sigmoid(20 * bd)
            bn_to_mask = torch.sigmoid(20 * bn)
            so_soft = torch.sigmoid(so)

            loss_dropout = F.kl_div(F.log_softmax(so_soft, dim=-1), F.softmax(bd_to_mask, dim=-1), reduction='mean')
            loss_noise = F.kl_div(F.log_softmax(so_soft, dim=-1), F.softmax(bn_to_mask, dim=-1), reduction='mean')

            loss_cls = gamma * (loss_o2a + loss_a2o + loss_dropout + loss_noise)



            self.running_seg_loss += loss_seg.item()
            self.running_cls_loss += loss_cls.item()
            loss_data = (loss_seg + loss_cls).data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss = loss_seg + loss_cls
            loss.backward()
            self.optim.step()

            # Momentum update T
            self.update_ema_variables(self.S_model, self.T_model)


            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(
                    image[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/image', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_cup', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_disc', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(so)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/prediction_cup', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(so)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/prediction_disc', grid_image, iteration)

            # write loss log
            self.writer.add_scalar('train_gen/loss', loss_data, iteration)
            self.writer.add_scalar('train_gen/loss_seg', loss_seg.data.item(), iteration)
            self.writer.add_scalar('train_gen/loss_cls', loss_cls.data.item(), iteration)

        self.running_seg_loss /= len(self.train_loader)
        self.running_cls_loss /= len(self.train_loader)
        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Average clsLoss: %f, Execution time: %.5f' %
              (self.epoch+1, get_lr(self.optim), self.running_seg_loss, self.running_cls_loss, stop_time - start_time))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch + 1) % (self.max_epoch//2) == 0:
                _lr_gen = self.lr * self.lr_decrease_rate
                for param_group in self.optim.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.train_loader)))
            if (self.epoch + 1) % self.interval_validate == 0 or self.epoch == 0:
                self.validate()
        self.writer.close()

    def get_current_consistency_weight(self, epoch, consistency, consistency_rampup):
        return consistency * self.sigmoid_rampup(epoch, consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return (np.exp(-0.5 * phase * phase))


    def update_ema_variables(self, model1, model2):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.iteration + 1), self.m)
        #print(alpha)
        for t_param, s_param in zip(model2.decoder.parameters(), model1.decoder.parameters()):
            t_param.data.mul_(alpha).add_(1 - alpha, s_param.data)
        for t_param, s_param in zip(model2.aspp.parameters(), model1.aspp.parameters()):
            t_param.data.mul_(alpha).add_(1 - alpha, s_param.data)
        for t_param, s_param in zip(model2.backbone.parameters(), model1.backbone.parameters()):
            t_param.data.mul_(alpha).add_(1 - alpha, s_param.data)



