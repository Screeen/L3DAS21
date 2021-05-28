import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import cfg

import logging
logger = logging.getLogger(__name__)

def load_dir_names():
    ov = 1
    split = 1
    db = 30
    nfft = 512
    feat_formatted_name = 'spec_ov{}_split{}_{}db_nfft{}_norm'.format(ov, split, db, nfft)
    label_formatted_name = 'label_ov{}_split{}_nfft{}_regr0'.format(ov, split, nfft)
    dataset_root = os.path.join(cfg.conf.datasets_root_dir, cfg.conf.dataset_format, cfg.conf.dataset_name)
    features_dir = os.path.join(dataset_root, feat_formatted_name)
    labels_dir = os.path.join(dataset_root, label_formatted_name)
    return features_dir, labels_dir

class DcaseDataset(Dataset):
    def __init__(self, features_dir, labels_dir, feature_transform=None, label_transform=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        logger.info(f"features_dir {features_dir}")
        logger.info(f"labels_dir {labels_dir}")
        
        self.feature_transform = feature_transform
        self.label_transform = label_transform

        self.labels_paths = sorted(os.listdir(self.labels_dir))
        self.features_paths = sorted(os.listdir(self.features_dir))
        if len(self.labels_paths) == 0 or len(self.features_paths) == 0:
            raise FileNotFoundError
        logger.info(f"Total number of samples in dataset {self.__len__()}")

    def __len__(self):
        return len(self.labels_paths)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.features_dir, self.features_paths[idx])
        label_path = os.path.join(self.labels_dir, self.labels_paths[idx])
        feature = np.load(feature_path)
        label = np.load(label_path)

        if self.feature_transform:
            feature = self.feature_transform(feature)
        if self.label_transform:
            label = self.label_transform(label)
            
        return feature, label