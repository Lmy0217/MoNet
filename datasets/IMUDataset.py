import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import configs
import datasets
import utils


class IMUDataset(datasets.BaseDataset):

    @staticmethod
    def more(cfg):
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.channel
        cfg.paths.source = configs.env.getdir(cfg.paths.source)
        cfg.device = torch.device('cpu')
        return cfg

    def load(self):
        source_data_path = os.path.join(self.cfg.paths.source, self.__class__.__name__ + self.cfg.data_mode + '.npy')
        if not os.path.exists(source_data_path):
            source_data = []
            self.source_idx = []
            obj_list = [obj for obj in sorted(os.listdir(self.cfg.paths.source)) if os.path.isdir(os.path.join(self.cfg.paths.source, obj))]
            print(obj_list)
            for idx_obj, obj_dir in enumerate(obj_list):
                obj_path = os.path.join(self.cfg.paths.source, obj_dir)
                print(idx_obj, obj_path)
                data = []
                for idx_scan, file in enumerate(sorted(os.listdir(obj_path))):
                    data_path = os.path.join(obj_path, file)
                    print('Data:', data_path)
                    source = h5py.File(data_path, 'r')
                    source = {
                        'frame': source['frames'][()],
                        'series': source['series'][()],
                        'imu_series': source['imu_series'][()],
                        'imu_accele': source['imu_accele'][()],
                    }
                    data.append(source)
                    self.source_idx.append((idx_obj, idx_scan))
                source_data.append(data)
            np.save(source_data_path, {'data': source_data, 'idx': self.source_idx})
        else:
            npy = np.load(source_data_path, allow_pickle=True)[()]
            source_data, self.source_idx = npy['data'], npy['idx']
        return {'source': source_data}, len(self.source_idx)

    def split(self):
        train_obj_size, test_obj_size = self.cfg.train_test_range
        self.trainset_length = np.sum([len(obj) for obj in self.data['source'][:train_obj_size]], dtype=np.int64)
        self.testset_length = np.sum([len(obj) for obj in self.data['source'][train_obj_size:train_obj_size + test_obj_size]], dtype=np.int64)
        self.trainset_length = self.trainset_length * self.cfg.series_train_count
        self.testset_length = self.testset_length * self.cfg.series_test_count
        self.cfg.data_count = self.trainset_length + self.testset_length
        index_range_trainset = [[0, self.trainset_length]]
        index_range_testset = [[self.trainset_length, self.cfg.data_count]]
        return datasets.BaseSplit(self, index_range_trainset), datasets.BaseSplit(self, index_range_testset)

    def get_idx(self, index):
        if index < self.trainset_length:
            idx = index // self.cfg.series_train_count
            utils.common.set_seed(int(time.time() * 1000) % (1 << 32) + index)
        else:
            idx = (index - self.trainset_length) // self.cfg.series_test_count + self.trainset_length // self.cfg.series_train_count
            utils.common.set_seed((index - self.trainset_length) * 3)
        return idx

    def preprocessing(self, tp):
        tp = tp.view(-1, 3, 3)
        pall = torch.cat([tp, 2 * tp[:, 0:1, :] - tp[:, 1:2, :], 2 * tp[:, 0:1, :] - tp[:, 2:3, :]], dim=1)
        min_loca = torch.min(pall.reshape(-1, 3), dim=0)[0]
        tp = tp - min_loca.unsqueeze(0).unsqueeze(0)
        td = utils.functional.series_to_dof(tp)
        tp = tp.view(-1, 9)
        return tp, td

    def __getitem__(self, index):
        idx_obj, idx_scan = self.source_idx[self.get_idx(index)]
        data = self.data['source'][idx_obj][idx_scan]

        source = data['frame']
        series = data['series']
        imu_series = data['imu_series']
        imu_accele = data['imu_accele']
        info = torch.tensor([len(source)])

        flip = torch.rand(1)[0] > 0.5
        frame_rate = torch.randint(self.cfg.frame_rate[0], self.cfg.frame_rate[1] + 1, (1,))[0]
        if not flip:
            index_select = torch.arange(0, len(source), frame_rate, dtype=torch.long, device=source.device)
        else:
            index_select = torch.arange(len(source) - 1, -1, -frame_rate, dtype=torch.long, device=source.device)

        if len(index_select) >= self.cfg.series_length[0]:
            series_length = torch.randint(self.cfg.series_length[0], min(len(index_select), self.cfg.series_length[1]) + 1, (1,))[0]
            series_start = torch.randint(0, len(index_select) - series_length + 1, (1,))[0]
        else:
            series_length, series_start = len(index_select), 0
        index_select = index_select[series_start:series_start + series_length]

        source = torch.index_select(source, 0, index_select)
        series = torch.index_select(series, 0, index_select)
        imu_series = torch.index_select(imu_series, 0, index_select)
        imu_accele = torch.index_select(imu_accele, 0, index_select)
        if flip:
            imu_accele = -imu_accele

        series, dof = self.preprocessing(series)
        imu_angle = self.preprocessing(imu_series)[1][:, 3:]

        source = source.unsqueeze(1)
        target = torch.cat([F.pad(dof, (0, 0, 0, 1)), series], dim=-1)

        source = source.type(torch.float32).to(self.cfg.device)
        target = target.type(torch.float32).to(self.cfg.device)
        imu_angle = imu_angle.type(torch.float32).to(self.cfg.device)
        imu_accele = imu_accele.type(torch.float32).to(self.cfg.device)

        sample_dict = {
            'source': source, 'target': target,
            'imu_angle': imu_angle,
            'imu_accele': imu_accele,
            'flip': flip,
            'frame_rate': frame_rate,
            'series_start': series_start,
            'series_length': series_length,
            'info': info
        }
        return sample_dict, index
