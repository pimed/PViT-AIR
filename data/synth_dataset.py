from math import cos, sin, pi
import torch
import random
import numpy as np
import random

from geotnf.transformation import SynthPairTnf
from data.base_dataset import baseDataset


####################################################################################################################################
class synthDataset(baseDataset):
    def __init__( self, catalog_path, device='cpu', filter=True,
                  geometric_model='affine', scale_range=0, rot_range=0, move_range=0, shear_range=0, 
                  image_h=224, image_w=224):
        super().__init__(catalog_path, device, filter)
        
        self.geometric_model = geometric_model
        self.rot_range = rot_range
        self.move_range = move_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.synthPairTnf = SynthPairTnf(geometric_model=self.geometric_model, device=self.device, output_size=(image_h, image_w))          
    #  ---------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):   
        label = self.labels[idx] 
        sample = dict(label=label)

        rows = self.catalog[self.catalog['label']==label].reset_index(drop=True)

        row = rows.iloc[0]
        image, mask, _  = self.load_data(row['target_image'], row['target_mask'])

        for n, row in rows.iterrows():
            image_segment, mask_segment, _ = self.load_data(row['source_image'], row['source_mask'])
            theta = self.get_theta()   
            sample[ f'theta_GT_{n}'] = self.reverse_theta(theta)

            # image
            complete_sample_n = self.synthPairTnf({'imageA':image.unsqueeze(0), 'imageB':image.unsqueeze(0), 'theta':theta.unsqueeze(0)} )        
            segment_sample_n = self.synthPairTnf({'imageA':image_segment.unsqueeze(0), 'imageB':image_segment.unsqueeze(0), 'theta':theta.unsqueeze(0)} )
            sample['target_image'] = complete_sample_n['source_image'].squeeze(0)   
            sample[f'target_image_{n}'] = segment_sample_n['source_image'].squeeze(0)
            sample[f'source_image_{n}'] = segment_sample_n['target_image'].squeeze(0)

            # mask
            complete_sample_n = self.synthPairTnf({'imageA':mask.unsqueeze(0), 'imageB':mask.unsqueeze(0), 'theta':theta.unsqueeze(0)}, asBinary=True)           
            segment_sample_n = self.synthPairTnf({'imageA':mask_segment.unsqueeze(0), 'imageB':mask_segment.unsqueeze(0), 'theta':theta.unsqueeze(0)}, asBinary=True)
            sample['target_mask_GT'] = complete_sample_n['source_image'].squeeze(0)   
            sample[f'target_mask_GT_{n}'] = segment_sample_n['source_image'].squeeze(0)
            sample[f'source_mask_GT_{n}'] = segment_sample_n['target_image'].squeeze(0)
        
        return sample    
    # ----------------------------------------------------------------------------------------------------------------
    def get_theta(self):
        if self.geometric_model == 'affine':

            if self.scale_range == 0: 
                Sx, Sy = 1., 1.
            else: 
                Sx = random.randint( (1-self.scale_range)*1000, (1+self.scale_range)*1000) * .001       # scaling coefficients within [0.9 : 1.1]
                Sy = random.randint( (1-self.scale_range)*1000, (1+self.scale_range)*1000) * .001       # scaling coefficients within [0.9 : 1.1]
            rot = random.randint(-self.rot_range, self.rot_range) * pi/180                              # rotation angle within [-20 : 20] degrees
            Tx =  random.randint(-self.move_range*1000, self.move_range*1000) * .001                    # shifting coefficients within 20% of image size
            Ty =  random.randint(-self.move_range*1000, self.move_range*1000) * .001                    # shifting coefficients within 20% of image size
            Shx =  random.randint(-self.shear_range*1000, self.shear_range*1000) * .001                 # shearing coefficients within 5% of image size
            Shy =  random.randint(-self.shear_range*1000, self.shear_range*1000) * .001                 # shearing coefficients within 5% of image size

            theta = [   Sx*(cos(rot)-Shy*sin(rot)),   Sy*(Shx*cos(rot)-sin(rot)),    Tx, 
                        Sx*(sin(rot)+Shy*cos(rot)),   Sy*(Shx*sin(rot)+cos(rot)),    Ty   ]                        
        
        
        else:
            x = np.array([2*i/(self.grid_size-1)-1 for i in range(self.grid_size)]).reshape(-1,1)
            grd = np.array([*np.repeat(x,self.grid_size), *np.reshape([x]*self.grid_size, (1,-1))[0]]).reshape(-1,)
            theta = grd + (2*np.random.rand(2*self.grid_size*self.grid_size)-1)*self.random_t_tps 

        return torch.Tensor(theta)                  
    #  ---------------------------------------------------------------------------------------------------------------
    def reverse_theta(self, theta):
        if self.geometric_model == 'affine':
            return torch.cat( [theta.view(6), torch.tensor([0, 0, 1])]).view(3,3).inverse().reshape(9)[:6]
        
        else: 
            return torch.Tensor([])
####################################################################################################################################
