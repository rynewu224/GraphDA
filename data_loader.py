import os

import numpy as np
import sklearn
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


class ChunkSampler(Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InfluenceDataSet(Dataset):
    def __init__(self, file_dir, shuffle=False, seed=27):
        self.original_adjs = np.load(os.path.join(file_dir, "adjacency_matrix_int8.npy")).astype(np.float32)
        print("Original graphs loaded!")

        self.influence_features = np.load(os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)
        print("Influence features loaded!")

        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        print("Labels loaded!")

        self.vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        print("Vertex IDs loaded!")

        if shuffle:
            self.original_adjs, self.influence_features, self.labels, self.vertices, = \
                sklearn.utils.shuffle(self.original_adjs,
                                      self.influence_features,
                                      self.labels,
                                      self.vertices,
                                      random_state=seed)

        vertex_features = np.load(os.path.join(file_dir, "vertex_feature.npy"))
        vertex_features = preprocessing.scale(vertex_features)
        self.vertex_features = torch.FloatTensor(vertex_features)
        print("Global vertex features loaded!")

        embedding = np.load(os.path.join(file_dir, "deepwalk.npy"))
        self.embedding = torch.FloatTensor(embedding)
        print("64-dim Deepwalk embedding loaded!")

        self.N = self.original_adjs.shape[0]
        print("%d ego networks loaded, each with size %d" % (self.N, self.original_adjs.shape[1]))

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)

    def get_embedding(self):
        return self.embedding

    def get_vertex_features(self):
        return self.vertex_features

    def get_feature_dimension(self):
        return self.influence_features.shape[-1]

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.original_adjs[idx], self.influence_features[idx], self.labels[idx], self.vertices[idx]


def load_influence_dataset(path, train_ratio, val_ratio, batch_size, shuffle, seed, num_workers=10):
    inf_ds = InfluenceDataSet(path, shuffle, seed)
    class_weight = inf_ds.get_class_weight()
    n_feat = inf_ds.get_feature_dimension()

    N = len(inf_ds)
    train_start = 0
    valid_start = int(N * train_ratio)
    test_start = int(N * (train_ratio + val_ratio))

    train_loader = DataLoader(dataset=inf_ds,
                              batch_size=batch_size,
                              drop_last=False,
                              sampler=ChunkSampler(valid_start - train_start, 0),
                              shuffle=False,
                              pin_memory=False,
                              num_workers=num_workers)

    valid_loader = DataLoader(dataset=inf_ds,
                              batch_size=batch_size,
                              drop_last=False,
                              sampler=ChunkSampler(test_start - valid_start, valid_start),
                              shuffle=False,
                              num_workers=num_workers)

    test_loader = DataLoader(dataset=inf_ds,
                             batch_size=batch_size,
                             drop_last=False,
                             sampler=ChunkSampler(N - test_start, test_start),
                             shuffle=False,
                              num_workers=num_workers)

    return inf_ds, n_feat, class_weight, train_loader, valid_loader, test_loader
