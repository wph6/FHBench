import os
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datautil.uea import *
from datautil.datasplit import getdataloader
from natsort import natsorted

def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'PTBXL': 'PTBXL', 'ADFTD': 'ADFTD'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]

def getlabeldataloader(args, data):
    trl, val, tel = getdataloader(args, data)
    trd, vad, ted = [], [], []
    for i in range(len(trl)):
        trd.append(torch.utils.data.DataLoader(
            trl[i], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))
    return trd, vad, ted

class TimeSeriesDataset(Dataset):
    def __init__(self,args,root_path):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        self.args = args
        self.pth_path = os.path.join(self.root_path,str(self.args.datapercent)+".pth")
        
        if os.path.exists(self.pth_path):
            print(f"Loading dataset...")
            pth_dataset = torch.load(self.pth_path)
            self.data, self.targets= pth_dataset['data'], pth_dataset['targets']
            print(f"Dataset loaded !")
        else:
            print(f"File {self.pth_path} does not exist. Creating dataset...")         
            self.data, self.targets = self.load_data(self.data_path, self.label_path)
            data_dict = {"data": self.data, "targets": self.targets}
            torch.save(data_dict, self.pth_path)
            print(f"File {self.pth_path} Saved!")
        
        # preprocess
        self.data = normalize_batch_ts(self.data)
        self.seq_len = 512
        
    def load_data(self,data_path, label_path):
        feature_list = []
        label_list = []
        labels = np.load(label_path)
        filenames = os.listdir(data_path)
        filenames = natsorted(filenames)
        for i,filename in enumerate(filenames):
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(labels[i][0])

        X = np.array(feature_list)
        y = np.array(label_list)

        if self.args.datapercent is not None:
            subset_size = int(len(X) * self.args.datapercent)
            indices = self.args.random_state.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = y[indices]
            
        return X, y

    def __getitem__(self, index):
        timeseries = self.data[index]
        label = self.targets[index]
        padding = self.seq_len - len(timeseries)
        timeseries = np.pad(timeseries,((padding, 0), (0, 0))).T
        return torch.from_numpy(timeseries), torch.tensor(label)
    
    def __len__(self):
        return len(self.targets)

def PTBXL(args):
    root_path = '/home/cxy/wph/timeseries_Lora/PTBXL'
    dataset = TimeSeriesDataset(args,root_path)
    trd, vad, ted = getlabeldataloader(args,dataset)
    num_classes = 5
    num_channels = 12
    return trd, vad, ted, num_classes, num_channels
def ADFTD(args):
    root_path = '/home/cxy/wph/timeseries_Lora/ADFTD'
    dataset = TimeSeriesDataset(args,root_path)
    trd, vad, ted = getlabeldataloader(args,dataset)
    num_classes = 3
    num_channels = 19
    return trd, vad, ted,num_classes, num_channels
