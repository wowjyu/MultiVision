from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
import glob, os
import numpy as np

def collate_fn_padd_batched(batch):
    """A batch contains n records"""
    merged = np.concatenate(batch, axis=0)
    
    pos = torch.Tensor([t[0] for t in merged])
    neg = torch.Tensor([t[1] for t in merged])
    seq_lengths = [t[2] for t in merged]
        
    return pos, neg, seq_lengths

class RankPairLoader:
    """
        Load Rank_pair dataset
    """
    def __init__(self, folder_name, batch_size = 2, validation_split = 0.2, num_workers = 8):
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers

        self.dataset = RankPairDataSetBatched(self.folder_name)

    def load_dataset(self):
        """
            Load dataset.
            Return train, val
        """
        ## Read dataset
        dataset = self.dataset

        ## Train/val split
        shuffle_dataset = True

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        if shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler, collate_fn= collate_fn_padd_batched, num_workers=self.num_workers)
        validation_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler, collate_fn= collate_fn_padd_batched, num_workers=self.num_workers)

        return train_loader, validation_loader

class RankPairDataSetBatched(Dataset):
    def __init__(self, folder_name: str):
        """
            fname is [[fname, feature]...]
            read raw file and concat features
        """
        self.folder_name = folder_name
        self.file_paths = glob.glob(os.path.join(folder_name, '*.npy'))
        
                       
    def __getitem__(self, index):
        """
            each is currently a [#column_indices, #feature]= 2*96 tensor
        """
        file_path = self.file_paths[index]
        data = np.load(file_path,allow_pickle=True)
        unpacked = [[n[0][0], n[0][1], n[1]] for n in data]  # (pos, neg, len)
        return unpacked
    
    def __len__(self):
        return len(self.file_paths)


class ChartTypeLoader:
    """
        Load chart type dataset
    """
    def __init__(self, file_name):
        self.file_name = file_name

        self.dataset = ChartTypeDataset(self.file_name)

    def load_dataset(self, batch_size = 2, validation_split = 0.2):
        """
            Load dataset.
            Return train, val
        """
        ## Read dataset
        dataset = self.dataset

        ## Train/val split
        shuffle_dataset = True

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, validation_loader

class ChartTypeDataset(Dataset):
    def __init__(self, file_name: str):
        """
            fname is [[fname, feature]...]
            read raw file and concat features
        """
        self.classes = ['area', 'bar', 'bubble', 'line', 'pie', 'radar', 'scatter', 'stock', 'surface']

        self.file_name = file_name
        self.data = [[x[0], self.chartTypeMap(x[1])] for x in np.load(file_name, allow_pickle=True)]
        
    def chartTypeMap(self, raw_name : str):
        if raw_name in ['area3DChart', 'areaChart']:
            return 0
        elif raw_name in ['bar3DChart', 'barChart']:
            return 1
        elif raw_name in ['bubbleChart']:
            return 2
        elif raw_name in ['line3DChart', 'lineChart']:
            return 3
        elif raw_name in ['doughnutChart', 'ofPieChart', 'pie3DChart', 'pieChart']:
            return 4
        elif raw_name in ['radarChart']:
            return 5
        elif raw_name in ['scatterChart']:
            return 6
        elif raw_name in ['stockChart']:
            return 7
        elif raw_name in ['surface3DChart', 'surfaceChart']:
            return 8
        else:
            return raw_name
      
    def __getitem__(self, index):
        """
            each is currently a [#column_indices, #feature]= 2*96 tensor
        """
        feature, label = self.data[index]
        return  torch.Tensor(feature), label
    
    def __len__(self):
        return len(self.data)


class ProvenanceDataset(Dataset):
    def __init__(self, file_name: str):
        """
            fname is [[fname, feature]...]
            read raw file and concat features
        """

        self.file_name = file_name
        pos, neg = np.load(file_name, allow_pickle=True)
        self.data = list(map(list, zip(pos, neg)))
        
    def __getitem__(self, index):
        return  self.data[index]
    
    def __len__(self):
        return len(self.data)

class ProvenanceLoader:
    """
        Load chart type dataset
    """
    def __init__(self, file_name):
        self.file_name = file_name

        self.dataset = ProvenanceDataset(self.file_name)

    def load_dataset(self, batch_size = 1024, validation_split = 0.2):
        """
            Load dataset.
            Return train, val
        """
        ## Read dataset
        dataset = self.dataset

        ## Train/val split
        shuffle_dataset = True

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, validation_loader

# def collate_fn_padd(batch):
#     '''
#     Padds batch of variable length [depracated]
#     '''
#     ## get sequence lengths
#     seq_lengths = torch.tensor([ len(t[0]) for t in batch ]).cuda()
#     ## padd
#     pos = [ torch.Tensor(t[0]).cuda() for t in batch ]
#     pos = torch.nn.utils.rnn.pad_sequence(pos, batch_first = True)
    
#     neg = [ torch.Tensor(t[1]).cuda() for t in batch ]
#     neg = torch.nn.utils.rnn.pad_sequence(neg, batch_first = True)
# #     ## compute mask
# #     mask = (batch != 0).to(device)
#     return pos, neg, seq_lengths

# class RankPairDataSet(Dataset):
#     def __init__(self, fname: str):
#         """
#             fname is [[fname, feature]...]
#             read raw file and concat features
#         """
#         self.data = np.load(fname, allow_pickle=True)
                         
#     def __getitem__(self, index):
#         """
#             each is currently a [#column_indices, #feature]= 2*96 tensor
#         """
#         (pos, neg), seq_len = self.data[index]   
#         return pos, neg, seq_len
    
#     def __len__(self):
#         return len(self.data)