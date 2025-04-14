import torch
import torchaudio
import librosa
from torchaudio import transforms as T
from torch.utils.data import Dataset
from datasets import Dataset as D
import pandas as pd
import math
import os
from datautil.datasplit import *

def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'ICBHI': 'ICBHI', 'SPRS': 'SPRS'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]

def convert_all_clients(partitions):
    datasets = [partition_to_dataset(part) for part in partitions]
    return datasets
def partition_to_dataset(partition):
    data_list = []
    label_list = []
    
    for i in range(len(partition)):
        data, label = partition[i]
        data_list.append(data)
        label_list.append(label)
    
    return D.from_dict({
        'data': data_list,
        'labels': label_list
    })

def getdatasets(args, data):
    trl, val, tel = getdataloader(args, data)
    train_datasets = convert_all_clients(trl)
    val_datasets = convert_all_clients(val)
    test_datasets = convert_all_clients(tel)
    return train_datasets,val_datasets,test_datasets

def ICBHI(args):
    dataset = get_ICBHI(args)
    trd, vad, ted = getdatasets(args,dataset)
    num_classes = 4
    return trd, vad, ted, num_classes
def SPRS(args):
    dataset = get_SPRS(args)
    trd, vad, ted = getdatasets(args,dataset)
    num_classes = 7
    return trd, vad, ted,num_classes

DESIRED_DURATION = 8 # only 15 respiratory cycles have a length >= 8 secs, and the 5 cycles that have a length >= 9 secs contain artefacts towards the end
DESIRED_SR = 16000 # sampling rate
SPRS_CLASS_DICT = {'Normal' : 0, 'Fine Crackle' : 1, 'Wheeze' : 2, 'Coarse Crackle' : 3,'Wheeze+Crackle' : 4, 'Rhonchi' : 5, 'Stridor' : 6}

# ICBHI label mapping
"""
LABEL_N, LABEL_C, LABEL_W, LABEL_B = 0, 1, 2, 3
label 0 for normal respiration
label 1 for crackles
label 2 for wheezes
label 3 for both
"""
class get_ICBHI(Dataset):
    def __init__(self,args,duration=DESIRED_DURATION, samplerate=DESIRED_SR, fade_samples_ratio=16, pad_type="circular"):

        self.data_path = '/mnt/dataset/ICBHI_final_database/'
        self.csv_path = 'datautil/ICBHI.csv'
        self.df = pd.read_csv(self.csv_path)
        self.duration = duration
        self.samplerate = samplerate
        self.targetsample = self.duration * self.samplerate
        self.pad_type = pad_type
        self.fade_samples_ratio = fade_samples_ratio
        self.fade_samples = int(self.samplerate/self.fade_samples_ratio)
        self.fade = T.Fade(fade_in_len=self.fade_samples, fade_out_len=self.fade_samples, fade_shape='linear')
        self.fade_out = T.Fade(fade_in_len=0, fade_out_len=self.fade_samples, fade_shape='linear')
        self.pth_path = os.path.join(self.data_path, "icbhi"+'_duration'+str(self.duration)+".pth")
        self.args = args
        if os.path.exists(self.pth_path):
            print(f"Loading dataset...")
            pth_dataset = torch.load(self.pth_path)
            self.data, self.targets= pth_dataset['data'], pth_dataset['targets']
            print(f"Dataset loaded !")
        else:
            print(f"File {self.pth_path} does not exist. Creating dataset...")         
            self.data, self.targets = self.get_dataset()
            data_dict = {"data": self.data, "targets": self.targets}
            torch.save(data_dict, self.pth_path)
            print(f"File {self.pth_path} Saved!")
            

    def get_sample(self, i):

        ith_row = self.df.iloc[i]
        filepath = ith_row['filepath']
        filepath = os.path.join(self.data_path, filepath)
        onset = ith_row['onset']
        offset = ith_row['offset']
        bool_wheezes = ith_row['wheezes']
        bool_crackles = ith_row['crackles']

        if not bool_wheezes:
            if not bool_crackles:
                label = 0
            else:
                label = 1
        else:
            if not bool_crackles:
                label = 2
            else:
                label = 3

        sr = librosa.get_samplerate(filepath)
        audio, _ = torchaudio.load(filepath, int(onset*sr), (int(offset*sr)-int(onset*sr)))
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.samplerate:
            resample = T.Resample(sr, self.samplerate)
            audio = resample(audio)
        
        return self.fade(audio), label

    def get_dataset(self):

        dataset = []
        targets = []

        for i in range(len(self.df)):
            audio, label = self.get_sample(i)   
            if audio.shape[-1] > self.targetsample:     
                audio = audio[...,:self.targetsample]
            else:
                if self.pad_type == 'circular':
                    ratio = math.ceil(self.targetsample / audio.shape[-1])
                    audio = audio.repeat(1, ratio)
                    audio = audio[...,:self.targetsample]
                    audio = self.fade_out(audio)
                elif self.pad_type == 'zero':
                    tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                    diff = self.targetsample - audio.shape[-1]
                    tmp[...,diff//2:audio.shape[-1]+diff//2] = audio
                    audio = tmp
            dataset.append(audio)
            targets.append(label)
        data =  torch.unsqueeze(torch.vstack(dataset), 1)
        label = torch.tensor(targets)
        if self.args.datapercent is not None:
            subset_size = int(len(data) * self.args.datapercent)
            indices = self.args.random_state.choice(len(data), subset_size, replace=False)
            data = data[indices]
            label = label[indices]
            
        return data,label
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class get_SPRS(Dataset):
    def __init__(self,args,duration=DESIRED_DURATION, samplerate=DESIRED_SR, fade_samples_ratio=16, pad_type="circular"):
        self.csv_path = 'datautil/SPRS.csv'
        self.data_path = '/mnt/dataset/SPRSound/'
        self.df = pd.read_csv(self.csv_path)
        self.duration = duration
        self.samplerate = samplerate
        self.targetsample = self.duration * self.samplerate
        self.pad_type = pad_type
        self.fade_samples_ratio = fade_samples_ratio
        self.fade_samples = int(self.samplerate/self.fade_samples_ratio)
        self.fade = T.Fade(fade_in_len=self.fade_samples, fade_out_len=self.fade_samples, fade_shape='linear')
        self.fade_out = T.Fade(fade_in_len=0, fade_out_len=self.fade_samples, fade_shape='linear')
        self.pth_path = os.path.join(self.data_path, "sprs"+'_duration'+str(self.duration)+".pth")
        self.args = args
        if os.path.exists(self.pth_path):
            print(f"Loading dataset...")
            pth_dataset = torch.load(self.pth_path)
            self.data, self.targets = pth_dataset['data'], pth_dataset['targets']
            print(f"Dataset loaded !")
        else:
            print(f"File {self.pth_path} does not exist. Creating dataset...")         
            self.data, self.targets = self.get_dataset()
            data_dict = {"data": self.data, "targets": self.targets}
            torch.save(data_dict, self.pth_path)
            print(f"File {self.pth_path} Saved!")
        
    def get_sample(self, i):

        ith_row = self.df.iloc[i]
        filepath = ith_row['wav_path']
        filepath = os.path.join(self.data_path, filepath)
        onset = ith_row['onset']
        offset = ith_row['offset']
        class_label = ith_row['event_label']

        label = SPRS_CLASS_DICT[class_label]

        _, sr = torchaudio.load(filepath, 0, 1)
        audio, _ = torchaudio.load(filepath, onset*int(sr/1000), offset*int(sr/1000) - onset*int(sr/1000))
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.samplerate:
            resample = T.Resample(sr, self.samplerate)
            audio = resample(audio)
        
        return self.fade(audio), label

    def get_dataset(self):

        dataset = []
        targets = []
        for i in range(len(self.df)):
            audio, label = self.get_sample(i)   
            if audio.shape[-1] > self.targetsample:     
                audio = audio[...,:self.targetsample]
            else:
                if self.pad_type == 'circular':
                    ratio = math.ceil(self.targetsample / audio.shape[-1])
                    audio = audio.repeat(1, ratio)
                    audio = audio[...,:self.targetsample]
                    audio = self.fade_out(audio)
                elif self.pad_type == 'zero':
                    tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                    diff = self.targetsample - audio.shape[-1]
                    tmp[...,diff//2:audio.shape[-1]+diff//2] = audio
                    audio = tmp
            dataset.append(audio)
            targets.append(label)
        data =  torch.unsqueeze(torch.vstack(dataset), 1)
        label = torch.tensor(targets)
        if self.args.datapercent is not None:
            subset_size = int(len(data) * self.args.datapercent)
            indices = self.args.random_state.choice(len(data), subset_size, replace=False)
            data = data[indices]
            label = label[indices]

        return data,label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    