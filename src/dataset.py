import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################

class Multimodal_Datasets(Dataset):
    def __init__(self, data_path, bed_path, device):
        super(Multimodal_Datasets, self).__init__()

        bed_df = pd.read_csv(bed_path, delimiter = '\t')
        
        vision = list()
        text = list()
        audio = list()
        filtered = list()
        for idx, sv in bed_df.iterrows():
            str_chr = str(sv.CHROM)
            str_pos = str(sv.POS)
            str_end = str(sv.END)
            ## Image
            file_path = data_path+'image/DEL_chr'+str_chr+"_"+str_pos+"_"+str_end+".npy"
            npz = np.load(file_path)
            if int(np.max(npz)) == 0: 
                filtered.append(idx)
                continue
            npz[np.where(npz>= 255)] = 255
            vision.append(npz)

            ## text
            file_path = data_path+'text/DEL_chr'+str_chr+"_"+str_pos+"_"+str_end+".npy"
            npz = np.load(file_path)
            text.append(npz)

            ## Signal
            file_path = data_path+'signal/DEL_chr'+str_chr+"_"+str_pos+"_"+str_end+".npy"
            npz = np.load(file_path)
            npz = (npz - np.mean(npz)) / np.std(npz)
            audio.append(npz)

#         labels = torch.tensor(bed_df['Y']).type(torch.float32)
        labels = np.array(bed_df['Y'], dtype=np.float32)
        vision = np.array(vision)
        audio = np.array(audio)
        text = np.array(text)
        if device:
            self.vision = torch.tensor(vision.astype(np.float32), dtype=torch.float32).cuda()
            self.text = torch.tensor(text.astype(np.float32), dtype=torch.float32).cuda()
            self.audio = torch.tensor(audio.astype(np.float32), dtype=torch.float32).cuda()
            self.labels = labels.cuda()
        else:
            self.vision = vision
            self.text = text
            self.audio = audio
            self.labels = labels

        self.n_modalities = 3 # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]   
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        return X, Y
