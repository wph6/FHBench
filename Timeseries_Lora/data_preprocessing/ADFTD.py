import os
import mne
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
# root dir
root = '/home/cxy/wph/data/ADFTD/'
# participants file path
participants_path = os.path.join(root, 'participants.tsv')
participants = pd.read_csv(participants_path, sep='\t')
labels = np.empty(shape=(participants.shape[0],2), dtype='int32')
label_map = {'A':2, 'F':1, 'C':0}
for i, participant in enumerate(participants.values):
    # print(participant)
    pid = int(participant[0][-3:])
    label = label_map[participant[3]]
    # print(pid)
    # print(label)
    labels[i,0] = label
    labels[i,1] = pid
label_path = '/home/cxy/wph/timeseries_Lora/ADFTD/Label'
if not os.path.exists(label_path):
    os.makedirs(label_path)
np.save(label_path + '/label.npy', labels)
derivatives_root = os.path.join(root, 'derivatives/')
# Test for bad channels, sampling freq and shape
bad_channel_list, sampling_freq_list, data_shape_list = [], [], []
for sub in os.listdir(derivatives_root):
    if 'sub-' in sub:
        sub_path = os.path.join(derivatives_root, sub, 'eeg/')
        # print(sub_path)
        for file in os.listdir(sub_path):
            if '.set' in file:
                file_path = os.path.join(sub_path, file)
                raw = mne.io.read_raw_eeglab(file_path, preload=False)
                # get bad channels
                bad_channel = raw.info['bads']
                bad_channel_list.append(bad_channel)
                # get sampling frequency
                sampling_freq = raw.info['sfreq']
                sampling_freq_list.append(sampling_freq)
                # get eeg data
                data = raw.get_data()
                data_shape = data.shape
                data_shape_list.append(data_shape)
def resampling(array, freq=500, kind='linear'):
    t = np.linspace(1, len(array), len(array))
    f = interpolate.interp1d(t, array, kind=kind)
    t_new = np.linspace(1, len(array), int(len(array)/freq * 256))
    new_array = f(t_new)
    return new_array

# segmentation with no overlapping (2560 timestamps)
# start from the middle position
def segment(df, window_size=2560):
    res = []
    start = int(df.shape[0]/2)
    left_index = start - int(start/window_size) * window_size
    right_index = start + int((df.shape[0]-start)/window_size) * window_size
    for i in range(left_index, right_index, window_size):
        res.append(df.iloc[i: i+window_size, :])
    return res


def eeg_data(eeg_path):
    # read .set file
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False)
    # raw = raw.pick(picks=li_common_channels)
    signals = raw.get_data()
    trial = []
    for i in range(signals.shape[0]):
        data = resampling(signals[i], freq=500, kind='linear')
        trial.append(data)
    #print(data.shape)
    df = pd.DataFrame(trial)
    df = np.transpose(df)
    # segmentation
    # res_df = segment(df, window_size=2560)
    res_df = segment(df, window_size=256)
    return res_df
feature_path = '/home/cxy/wph/timeseries_Lora/ADFTD/Feature'
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

sub_id = 1
for sub in tqdm(os.listdir(derivatives_root)):
    if 'sub-' in sub:
        li_sub = []
        sub_path = os.path.join(derivatives_root, sub, 'eeg/')
        # print(sub_path)
        for file in os.listdir(sub_path):
            if '.set' in file:
                file_path = os.path.join(sub_path, file)
                res_df = eeg_data(file_path)
                for df_std in res_df:
                    # print(df_std)
                    print('--------------------------------------------------------------------------')
                    li_sub.append(df_std.values)
        array_sub = np.array(li_sub)
        # print(array_sub.shape)
        np.save(feature_path + '/feature_{:02d}.npy'.format(sub_id), array_sub)
        sub_id += 1

path = '/home/cxy/wph/timeseries_Lora/ADFTD/Feature'

for file in os.listdir(path):
    sub_path = os.path.join(path, file)
    print(np.load(sub_path).shape)