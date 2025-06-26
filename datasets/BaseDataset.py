import abc
import os

from torch.utils.data import Dataset

import configs
from utils import Logger

__all__ = ['BaseDataset', 'BaseSplit']


class BaseDataset(Dataset, metaclass=abc.ABCMeta):

    logger: Logger

    def __init__(self, cfg, **kwargs):
        self.name = os.path.splitext(os.path.split(cfg._path)[1])[0]
        self.cfg = self.more(self._more(cfg))
        self.data, self.cfg.data_count = self.load()

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _more(cfg):
        for name, value in configs.env.dataset.dict().items():
            setattr(cfg, name, getattr(cfg, name, value))
        return cfg

    @staticmethod
    def more(cfg):
        return cfg

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.cfg.data_count


class BaseSplit(Dataset):

    def __init__(self, dataset, index_range_set):
        self.dataset = dataset
        self.indexset = self._index(index_range_set)
        self.count = len(self.indexset)

        if hasattr(self.dataset, 'logger'):
            self.logger = self.dataset.logger

    def _index(self, index_range_set):
        indexset = []
        for index_range in index_range_set:
            indexset.extend(range(index_range[0], index_range[1]))
        return indexset

    def __getitem__(self, index):
        return self.dataset[self.indexset[index]][0], index

    def __len__(self):
        return self.count
