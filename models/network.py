  

import numpy as np 
import torch
import torch.nn as nn


from .utils.vit_blocks import *
from geotnf.transformation import GeometricTnf
from geotnf.landmark_tnf import LandmarkTnf


####################################################################################################################################
class regViTNet(nn.Module):
    def __init__(self, image_h=224, image_w=224, in_chans=3, 
                 patch_size=16, stride=16, num_layers=3, num_heads=2, embed_dim=256, dropout=0., theta_lr=.1, 
                 bias=False, pos_embedding=False, device='cpu', crop_margin=.1):
        super().__init__()
        assert image_h == image_w, 'Image dimensions must be the same.'

        image_size = image_h
        self.device = device
        self.crop_margin = crop_margin

        self.ViT = Transformer(
            image_size=image_size, in_chans=in_chans, patch_size=patch_size, embed_dim=embed_dim, 
            stride=stride, num_layers=num_layers, num_heads=num_heads, dropout=dropout, bias=bias, 
            pos_embedding=pos_embedding, device=device)
        
        self.crossViT = CrossTransformer(
            image_size=image_size, in_chans=in_chans, patch_size=patch_size, embed_dim=embed_dim, 
            stride=stride, num_layers=num_layers, num_heads=num_heads, dropout=dropout, bias=bias, 
            pos_embedding=pos_embedding, device=device)        

        self.head = nn.Sequential(  nn.Linear(2*embed_dim, embed_dim, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(embed_dim, 6, bias=False),
                                    nn.Tanh()  ).to(device)

        self.resampler = GeometricTnf(out_h=image_h, out_w=image_w, device=device)
        self.geoTransformer = GeometricTnf(out_h=image_h, out_w=image_w, device=device)
        self.lmkTransformer = LandmarkTnf(out_h=image_h, out_w=image_w, device=device)
        self.TnfStabilizer = TnfStabilizer(device=device, theta_lr=theta_lr)     
    #-----------------------------------------------------------------------------------------------------------------------------
    def forward(self, target_image, source_images, source_masks, source_dpi=None, target_dpi=None): 
        src_cropped_list, trg_cropped_list = self.crop_batch(target_image, source_images, source_masks, source_dpi=source_dpi, target_dpi=target_dpi)        
        complete_out = self.ViT( *[self.resampler(s) for s in source_images] )

        theta_list = list()
        for (src_cropped, trg_cropped) in zip(src_cropped_list, trg_cropped_list):
            cropped_out = self.crossViT(src_cropped, trg_cropped)
            theta_n = self.head( torch.cat([complete_out, cropped_out], dim=-1) ).mean(dim=1)
            theta_n = self.TnfStabilizer(theta_n)
            theta_list.append(theta_n)
        
        return theta_list            
    #-----------------------------------------------------------------------------------------------------------------------------
    def crop_batch(self, target_image, source_images, source_masks, source_dpi=None, target_dpi=None):
        num_batches = target_image.size(0)
        source_image_cropped, target_image_cropped = list(), list()

        for (source_image_n, source_mask_n) in zip(source_images, source_masks):       
            source_image_cropped_n, target_image_cropped_n = list(), list()
            for b in range(num_batches):
                src_img_crp_n, trg_img_crp_n = self.crop_image(target_image[b], source_image_n[b], source_mask_n[b], source_dpi=source_dpi, target_dpi=target_dpi)                
                source_image_cropped_n.append( self.resampler(src_img_crp_n.unsqueeze(0)).squeeze(0) )
                target_image_cropped_n.append( self.resampler(trg_img_crp_n.unsqueeze(0)).squeeze(0) )
                
            source_image_cropped.append( torch.stack(source_image_cropped_n) )
            target_image_cropped.append( torch.stack(target_image_cropped_n) )        
        
        return source_image_cropped, target_image_cropped 
    #-----------------------------------------------------------------------------------------------------------------------------
    def crop_image(self, target_image, source_image, source_mask, source_dpi=None, target_dpi=None, **kwargs):   
        x1, y1, x2, y2 = self.get_bbox( source_mask, **kwargs )
        source_image_cropped = source_image[:, y1:y2, x1:x2] 
        if source_dpi is not None and target_dpi is not None:
                x1 = int( x1 / source_dpi[0] * target_dpi[0] )
                x2 = int( x2 / source_dpi[0] * target_dpi[0] )
                y1 = int( y1 / source_dpi[1] * target_dpi[1] )
                y2 = int( y2 / source_dpi[1] * target_dpi[1] )  
        target_image_cropped = target_image[:, y1:y2, x1:x2] 
        
        return source_image_cropped, target_image_cropped
    #-----------------------------------------------------------------------------------------------------------------------------
    def get_bbox(self, mask_tensor, size_th=10):
        H, W = mask_tensor.shape[1:]
        try:
            col = mask_tensor.any(dim=1).all(dim=0).int()
            row = mask_tensor.any(dim=2).all(dim=0).int()

            x1, x2 = torch.where(col)[0][[0, -1]]
            y1, y2 = torch.where(row)[0][[0, -1]]
            cx, cy = (x1+x2).div(2, rounding_mode='trunc'), (y1+y2).div(2, rounding_mode='trunc')
            w, h = int((1+self.crop_margin)*(x2-x1)), int((1+self.crop_margin)*(y2-y1))
            w, h = max(w, h), max(h,w)
            x1, y1 = max(0, cx-w//2), max(0, cy-h//2)
            x2, y2 = min(W, cx+w//2), min(H, cy+h//2)

            if x2-x1 > size_th and y2-y1 > size_th: 
                return x1, y1, x2, y2
        except: pass 
        return 0, 0, H, W
####################################################################################################################################          
class TnfStabilizer(nn.Module):
    def __init__(self, geometric_model='affine', device='cpu', theta_lr=.1):
        super().__init__()
        self.theta_lr = theta_lr
        self.device = device
        if geometric_model=='affine':
            self.identity = torch.tensor([1.0,0,0,0,1.0,0])           
        else:
            grid_size = int(geometric_model.split('_')[1])
            x = np.array([2*i/(grid_size-1)-1 for i in range(grid_size)]).reshape(-1,1)
            grd = np.array([*np.repeat(x,grid_size), *np.reshape([x]*grid_size, (1,-1))[0]]).reshape(-1,)
            self.identity = torch.tensor(grd, dtype=torch.float32)
    # ----------------------------------------------------------------------------------------------------
    def forward(self, theta):
        if self.theta_lr == 1: return theta
        theta = theta.view(-1, self.identity.shape[0])
        adjust = self.identity.repeat(theta.shape[0],1).to(self.device)
        theta = self.theta_lr*theta + adjust
        theta = theta.to(self.device)

        return theta                 
####################################################################################################################################

