from torch.autograd import Variable
import torch
import pandas as pd

from .base_dataset import baseDataset


####################################################################################################################################
class realDataset(baseDataset):  
    def __getitem__(self, idx):         
        label = self.labels[idx] 
        sample = dict(label=label) 

        rows = self.catalog[self.catalog['label']==label].reset_index(drop=True)
        row = rows.iloc[0]
        sample['target_image'], sample['target_mask_GT'], sample['target_dpi'] = self.load_data(row['target_image'], row['target_mask'])
        lmks = self.load_landmarks(landmarks_path=row['target_landmarks'])
        if len(lmks): sample['target_landmarks'] = lmks 

        sample['num_sources'] = len(rows)
        for n, row in rows.iterrows():
            sample[f'source_image_{n}'], sample[f'source_mask_GT_{n}'], sample['source_dpi'] = self.load_data(row['source_image'], row['source_mask'])
            lmks = self.load_landmarks(landmarks_path=row['source_landmarks'])
            if len(lmks): sample[f'source_landmarks_{n}'] = lmks 

        return sample
    # ----------------------------------------------------------------------------------------------------------------    
    def load_landmarks(self, landmarks_path):        
        try: 
            landmarks_npy = pd.read_csv(landmarks_path)[['x','y']].values.astype('int')
        except: 
            landmarks_npy = [] 

        return Variable(torch.Tensor(landmarks_npy), requires_grad=False)           
####################################################################################################################################

