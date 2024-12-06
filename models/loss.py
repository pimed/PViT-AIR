
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils.loss_functions import hd, hd95, get_contour, get_matching_lines
from geotnf.point_tnf import PointTnf

InchMM_ratio = 25.4



####################################################################################################################################
class Metric(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
    # ----------------------------------------------------------------------------------------------------    
    def forward(self, *args, make_imageBatch=False):
        args_new = list()
        for a in args:
            a_new = a if torch.is_tensor(a) else torch.Tensor(a) 
            if make_imageBatch and a.ndim==3: a_new = a_new.unsqueeze(0)
            a_new = a_new.to(self.device)                                   # Device (cuda/cpu)
            args_new.append(a_new)
        return args_new
####################################################################################################################################
class WeightedLoss(Metric):
    def __init__(self, weightedMetrics_str):
        super().__init__()

        self.metrics = list()
        self.weights = list()

        for wm in weightedMetrics_str.split('+'):
            if '*' in wm: w, m = wm.split('*')
            else: w, m = 1, wm

            self.metrics.append(m)
            self.weights.append(float(w))
    # ----------------------------------------------------------------------------------------------------
    def forward(self, evals_dict):    
        return torch.sum(torch.stack( [w*evals_dict[m] for (w, m) in zip(self.weights, self.metrics)] ))
####################################################################################################################################
class TransformedGridLoss(Metric):
    def __init__(self, geometric_model='affine', grid_size=120, device='cpu'):
        super().__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False).to(device)
        self.pointTnf = PointTnf(device=device)
    # ----------------------------------------------------------------------------------------------------
    def forward(self, theta, theta_GT):
        # theta, theta_GT are batches

        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model=='affine':
            P_prime = self.pointTnf.affPointTnf(theta,P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT,P)        
        elif self.geometric_model=='tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT,P)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss = torch.mean(loss)
        
        return loss                
####################################################################################################################################
class MSE(Metric):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.mseLoss = nn.MSELoss()
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B):
        A_, B_ =  super().forward(A, B)
        A_, B_ = A_.div(255.0), B_.div(255.0)
        return  self.mseLoss(A_, B_)
####################################################################################################################################
class MLE(Metric):
    # Landmark Distance Error (mm)
    def forward(self, target_landmarks, warped_landmarks, target_dpi, warped_dpi):
        if warped_dpi is None: warped_dpi = target_dpi
        # landmarks are expected in [B,N,2] (torch)

        trg_lmk, wrp_lmk, trg_dpi, wrp_dpi = super().forward(target_landmarks, warped_landmarks, target_dpi, warped_dpi)
        if wrp_dpi is None: wrp_dpi = trg_dpi
        
        pxl2mm_target = torch.Tensor([ [float(InchMM_ratio/trg_dpi[0]), 0], [0, float(InchMM_ratio/trg_dpi[1])] ]).to(self.device)
        pxl2mm_warped = torch.Tensor([ [float(InchMM_ratio/wrp_dpi[0]), 0], [0, float(InchMM_ratio/wrp_dpi[1])] ]).to(self.device)
        pointsA = torch.matmul(trg_lmk, pxl2mm_target)
        pointsB = torch.matmul(wrp_lmk, pxl2mm_warped)

        # lmkDist = (pointsA - pointsB).pow(2).sum(dim=2).sqrt().sum(dim=0)
        lmkDist = (pointsA - pointsB).pow(2).sum(dim=-1).sqrt()
        if len(np.shape(lmkDist)) > 1: lmkDist = lmkDist.sum(dim=0)
        
        return lmkDist                  
####################################################################################################################################
class Dice(Metric):
    def forward(self, A, B, eps=1e-8):    
        A_, B_ = super().forward(A, B)
        A_, B_ = A_.div(255.0), B_.div(255.0)

        A_ = A_.contiguous().view(-1)
        B_ = B_.contiguous().view(-1)        
        intersection = (A_ * B_).sum()                            
        dice = (2.*intersection)/(A_.sum() + B_.sum() + eps)      

        return dice  
####################################################################################################################################
class HD(Metric):
    # Reference: https://github.com/mavillan/py-hausdorff
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B, DPI): 
        # A, B are masks in shape of (B,C,H,W)
        A_, B_ = super().forward(A, B, make_imageBatch=True)

        v = 0
        for (m1, m2, dpi) in zip(A_, B_, DPI):
            m1 = m1.data.cpu().numpy().sum(axis=0)
            m2 = m2.data.cpu().numpy().sum(axis=0)
            spacing = InchMM_ratio/np.array(dpi)             
            v += hd(m1, m2, spacing)        
            
        return Variable(torch.Tensor( [v/A_.size()[0]]) , requires_grad=False).sum()
####################################################################################################################################
class HD95(Metric):
    # Reference: https://github.com/mavillan/py-hausdorff
    # ----------------------------------------------------------------------------------------------------
    def forward(self, A, B, DPI): 
        # A, B are masks in shape of (B,C,H,W)
        A_, B_ = super().forward(A, B, make_imageBatch=True)

        v = 0
        for (m1, m2, dpi) in zip(A_, B_, DPI):
            m1 = m1.data.cpu().numpy().sum(axis=0)
            m2 = m2.data.cpu().numpy().sum(axis=0)
            spacing = InchMM_ratio/np.array(dpi)             
            v += hd95(m1, m2, spacing)        
            
        return Variable(torch.Tensor( [v/A_.size()[0]]) , requires_grad=False).sum()
####################################################################################################################################
class SDM(Metric):
    # Stitch Distance Metric
    # ---------------------------------------------------------------------------------------------------- 
    def forward_old(self, A, B, dpi, num_samples=10):    
        A_, B_ = A.detach().cpu(), B.detach().cpu()
        for (M1, M2) in zip(A_, B_):
            contour1 = get_contour(M1)
            contour2 = get_contour(M2)
            line1, line2 = get_matching_lines(contour1, contour2, num_samples)

            # dist_mm_list = np.linalg.norm((line1-line2)/dpi, axis=1) * InchMM_ratio
            dist_mm_list = np.array( [np.min( np.linalg.norm((p1-line2)/dpi, axis=1)) for p1 in line1] ) * InchMM_ratio
            dist_mm = np.mean(np.sort(dist_mm_list)[:int(.95*num_samples)-1])

        return torch.Tensor([dist_mm])[0]
    # ---------------------------------------------------------------------------------------------------- 
    def forward(self, dpi, *args, num_samples=10):   
        try: 
            masks = [arg.detach().cpu().squeeze(0) for arg in args]
            num_masks = len(masks)
            if num_masks < 2: return
            dists = np.ones((num_masks,num_masks)) * np.inf

            for i in range(num_masks-1):
                for j in range(i+1, num_masks):
                    contour1 = get_contour(masks[i])
                    contour2 = get_contour(masks[j])
                    line1, line2 = get_matching_lines(contour1, contour2, num_samples)

                    # dist_mm_list = np.linalg.norm((line1-line2)/dpi, axis=1) * InchMM_ratio
                    dist_mm_list = np.array( [np.min( np.linalg.norm((p1-line2)/dpi, axis=1)) for p1 in line1] ) * InchMM_ratio
                    dist_mm = np.mean(np.sort(dist_mm_list)[:int(.95*num_samples)-1])

                    dists[i,j] = dist_mm
                    dists[j,i] = dist_mm

            dist_mm = np.mean(np.min(dists, axis=0))

            return torch.Tensor([dist_mm])[0]    

        except: torch.nan
####################################################################################################################################    




 