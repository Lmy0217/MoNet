import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils


class Generator(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.block1 = self._block(input_size, 128, normalize=False)
        self.block2 = self._block(128, 256)
        self.block3 = self._block(256, 512)
        self.fc = nn.Linear(512, output_size)
        self.tanh = nn.Tanh()

    def _block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x


class Backbone(nn.Module):

    def __init__(self, in_planes, num_classes, cfg):
        super().__init__()
        self.cfg = cfg
        lstm_planes = 512
        self.resnet = timm.create_model('resnet18', pretrained=True, in_chans=in_planes, num_classes=0, global_pool='')
        if cfg.accele:
            self.gen = Generator(3, 90)
            self.lstm_a = models.layers.convolutional_rnn.Conv2dLSTM(lstm_planes, lstm_planes, kernel_size=1, batch_first=True)
        if cfg.angle:
            self.angle_planes = 1
            self.gen_angle = Generator(num_classes - 3, 90 * self.angle_planes)
            lstm_planes = lstm_planes + self.angle_planes
        self.lstm = models.layers.convolutional_rnn.Conv2dLSTM(lstm_planes, lstm_planes, kernel_size=3, batch_first=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(lstm_planes, num_classes)

    def forward(self, x, accele=None, angle=None):
        x = 2 * (x / 255.0) - 1.0
        b, t, c = x.shape[:3]
        x = x.view(b * t, c, *x.shape[3:])
        x = self.resnet(x)
        x = x.view(b, t, *x.shape[1:])

        if self.cfg.accele:
            accele = accele.view(b * (t - 1), -1)
            a = self.gen(accele)
            a = a.view(b, t - 1, 1, *x.shape[3:])
            a = torch.cat([x[:, 0:1], a + x[:, :-1]], dim=1)
            a = self.lstm_a(a)[0]
            x = x + a

        if self.cfg.angle:
            angle = angle.view(b * t, *angle.shape[2:])
            g = self.gen_angle(angle)
            g = g.view(b, t, self.angle_planes, *x.shape[3:])
            x = torch.cat([x, g], dim=2)

        x = self.lstm(x)[0]
        x = self.avg(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)

        return x


class Online_Backbone(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.backbone = Backbone(self.data_cfg.source.channel, self.data_cfg.target.elements - 9, cfg=self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.run.lr, betas=self.run.betas, weight_decay=self.run.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.run.step_size, gamma=self.run.gamma)

    def criterion(self, real_target, fake_target):
        real_dist, real_angle = real_target.split([3, self.data_cfg.target.elements - 12], dim=-1)
        fake_dist, fake_angle = fake_target.split([3, self.data_cfg.target.elements - 12], dim=-1)

        loss_dist = F.l1_loss(real_dist, fake_dist) * 3
        loss_angle = F.l1_loss(real_angle, fake_angle) * 3
        loss_corr = utils.metric.correlation_loss(real_target, fake_target)

        loss_dict = {'loss_dist': loss_dist, 'loss_angle': loss_angle, 'loss_corr': loss_corr}

        return loss_dict

    def train(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device)
        angle = sample_dict['imu_angle'].to(self.device) if self.cfg.angle else None
        accele = sample_dict['imu_accele'][:, 1:-1].to(self.device) if self.cfg.accele else None

        real_target = real_target[:, :-1, :-9]
        real_target[:, :, 3:] = real_target[:, :, 3:] * 100
        if self.cfg.angle:
            angle = angle * 100

        self.backbone.train()
        self.optimizer.zero_grad()
        input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)

        fake_target = self.backbone(input, accele=accele, angle=angle)

        losses = self.criterion(real_target, fake_target)
        loss = sum(losses.values())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        return {'loss': loss, **losses}

    def test(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device).squeeze(0)
        angle = sample_dict['imu_angle'].to(self.device) if self.cfg.angle else None
        accele = sample_dict['imu_accele'][:, 1:-1].to(self.device) if self.cfg.accele else None

        real_series = real_target[:, -9:].view(-1, 3, 3)
        if self.cfg.angle:
            angle = angle * 100

        self.backbone.eval()
        input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)

        fake_gaps = self.backbone(input, accele=accele, angle=angle)

        fake_gaps = fake_gaps[0, :, :]
        fake_gaps[:, 3:] /= 100

        fake_series = utils.functional.dof_to_series(real_series[0:1, :, :], fake_gaps.unsqueeze(0)).squeeze(0)
        losses = utils.metric.get_metric(real_series, fake_series)

        return losses

    def test_return_hook(self, epoch_info, return_all):
        return_info = {}
        for key, value in return_all.items():
            return_info[key] = np.sum(value) / epoch_info['batch_per_epoch']
        if return_info:
            self.logger.info_scalars('{} Epoch: {}\t', (epoch_info['log_text'], epoch_info['epoch']), return_info)
        return return_all
