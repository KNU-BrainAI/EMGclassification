from torch.utils.data import Dataset
import numpy as np
import torch

class Nina1Dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        target_row = self.dataframe.iloc[idx]
        data = target_row['emg'][0:500]
        "Zero-Padding"
        if len(data)<500:
            data = np.concatenate((data,np.zeros(((500-len(data)),10))),axis=0)
        #mean = np.mean(data, axis =0)
        #std = np.mean(data, axis=0)
        #data = (data-mean)/std
        "Division data by time-segment"
        input_data = torch.tensor(np.transpose(data.reshape((25,20,10)),(0,2,1)),dtype=torch.float)
        
        label = torch.tensor(target_row['stimulus'],dtype=torch.long)
        
        return {'input_data': input_data,
                'label': label}