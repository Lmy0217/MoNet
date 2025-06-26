import os

import torch
import torch.nn.functional as F

import configs
import models
import utils


class Online_Framework(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.backbone = models.online_backbone.Backbone(self.data_cfg.source.channel, self.data_cfg.target.elements - 9, cfg=self.cfg).to(self.device)
        self.backbone_start_weight = torch.load(configs.env.getdir(self.cfg.backbone_weight))
        self.backbone.load_state_dict(self.backbone_start_weight)

    def train(self, epoch_info, sample_dict):
        return {}

    def criterion(self, fake_target, accele=None, angle=None):
        fake_dist, fake_angle = fake_target.split([3, self.data_cfg.target.elements - 12], dim=-1)

        accele = accele / torch.mean(torch.abs(accele), dim=1, keepdim=True)
        fake_target = torch.cat([fake_target[..., :3], fake_target[..., 3:] / 100], dim=-1)
        fake_accele = utils.functional.accele_from_dof(fake_target.squeeze(0)).unsqueeze(0)
        fake_accele = fake_accele / torch.mean(torch.abs(fake_accele), dim=1, keepdim=True)

        loss_corr_accele = utils.metric.correlation_loss(accele, fake_accele)
        loss_angle = F.l1_loss(angle, fake_angle)

        loss_dict = {'loss_corr_accele': loss_corr_accele, 'loss_angle': loss_angle}
        return loss_dict

    def test_optimize(self, epoch_info, real_source, real_target, angle, accele, epoch):
        self.backbone.load_state_dict(self.backbone_start_weight)

        real_gaps = real_target[0, :-1, :-9]
        real_series = real_target[0, :, -9:].view(-1, 3, 3)
        if self.cfg.angle:
            angle = angle * 100

        real_input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)

        value = {'real_source': real_source, 'real_gaps': real_gaps.clone(), 'real_series': real_series.clone(), 'angle': angle.clone(), 'accele': accele.clone()}
        self.backbone.eval()
        fake2_gaps = self.backbone(real_input, accele=accele[:, 1:-1], angle=angle)
        fake2_gaps = fake2_gaps[0, :, :]
        fake2_gaps[:, 3:] /= 100
        fake2_series = utils.functional.dof_to_series(real_series[0:1, :, :], fake2_gaps.unsqueeze(0)).squeeze(0)
        losses2 = utils.metric.get_metric(real_series, fake2_series)
        value['fake_gaps'] = [fake2_gaps]
        value['fake_series'] = [fake2_series]
        value['loss'] = [losses2]

        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.run.lr, betas=self.run.betas, weight_decay=self.run.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.run.step_size, gamma=self.run.gamma)

        with torch.enable_grad():
            for idx in range(1, epoch + 1):
                self.logger.info(f"MoNet: Data {epoch_info['index'].item() + 1}/{epoch_info['count_data']} Epoch {idx}/{epoch}")
                self.backbone.train()

                self.optimizer.zero_grad()
                fgaps = self.backbone(real_input, accele=accele[:, 1:-1], angle=angle)
                losses = self.criterion(fgaps, accele=accele[:, 1:-1], angle=angle)
                loss = sum(losses.values())

                self.logger.info_scalars('Online\t', (), {'loss': loss.item(), 'loss_accele': losses['loss_corr_accele'].item(), 'loss_angle': losses['loss_angle'].item()})
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(idx)
                torch.cuda.empty_cache()

                with torch.no_grad():
                    self.backbone.eval()
                    fake2_gaps = self.backbone(real_input, accele=accele[:, 1:-1], angle=angle)
                    fake2_gaps = fake2_gaps[0, :, :]
                    fake2_gaps[:, 3:] /= 100
                    fake2_series = utils.functional.dof_to_series(real_series[0:1, :, :], fake2_gaps.unsqueeze(0)).squeeze(0)
                    losses2 = utils.metric.get_metric(real_series, fake2_series)
                    value['fake_gaps'].append(fake2_gaps)
                    value['fake_series'].append(fake2_series)
                    value['loss'].append(losses2)

        self.backbone.eval()
        return value

    def test(self, epoch_info, sample_dict):
        if epoch_info['index'].item() < 0:
           return {}
        utils.common.set_seed(epoch_info['index'].item() * 42)
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device)
        angle = sample_dict['imu_angle'].to(self.device) if self.cfg.angle else None
        accele = sample_dict['imu_accele'].to(self.device) if self.cfg.accele else None
        frame_rate = sample_dict['frame_rate']
        length = min(sample_dict['info'], real_source.shape[1])

        value = self.test_optimize(epoch_info, real_source, real_target, angle, accele, epoch=self.run.ol_epochs)
        value['frame_rate'] = [frame_rate]
        value['length'] = [length]

        path = os.path.join(self.path, 'MoNet')
        if not os.path.exists(path):
            os.makedirs(path)
        source = value.pop('real_source')
        torch.save(source, os.path.join(path, 'source_' + str(epoch_info['index'].item()) + '.pth'))
        torch.save(value, os.path.join(path, 'value_' + str(epoch_info['index'].item()) + '.pth'))

        return {}
