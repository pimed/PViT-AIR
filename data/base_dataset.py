from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import numpy as np
from PIL import Image
import copy
import random
import pandas as pd

####################################################################################################################################
class baseDataset(Dataset):      
    def __init__(self, catalog_path, device='cpu', filter=True):
        self.catalog = pd.read_csv(catalog_path)
        self.labels = self.catalog['label'].unique()

        self.device = device
        self.filter = filter
    #  ---------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)                                 
    # ----------------------------------------------------------------------------------------------------------------    
    def load_data(self, image_path, mask_path):        
        # image
        image_PIL = Image.open(image_path).convert('RGB')
        image_npy = np.array(image_PIL).astype(np.uint8)    

        # dpi
        dpi = np.array(image_PIL.info['dpi']).astype('float')       

        # mask
        mask_PIL = Image.open(mask_path).convert('1').convert('L')
        mask_npy = np.repeat( np.expand_dims(np.array(mask_PIL).astype(np.uint8), axis=2), 3, axis=2)

        if self.filter: 
            image_npy = (image_npy * np.divide(mask_npy, 255.).astype('int')).astype('uint8')

        image = Variable(torch.Tensor(image_npy.transpose((2,0,1))), requires_grad=False).to(self.device)      
        mask = Variable(torch.Tensor(mask_npy.transpose((2,0,1))), requires_grad=False).to(self.device) 
    
        return image, mask, dpi                               
    # ----------------------------------------------------------------------------------------------------
    def split(self, tr_ratio=.3):  
        N = len(self.labels)
        indexes = list(range(N))
        random.shuffle(indexes) 
        N_tr = int(N*float(tr_ratio))
               
        train_dataset = copy.deepcopy(self)
        train_dataset.labels = [self.labels[idx] for idx in indexes[:N_tr]]

        val_dataset = copy.deepcopy(self)
        val_dataset.labels = [self.labels[idx] for idx in indexes[N_tr:]]        

        return train_dataset, val_dataset 
####################################################################################################################################
