
import os
import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torchvision.transforms as T


from .network import*
from .loss import*


####################################################################################################################################
class regModel(nn.Module):
    def __init__(self, checkpoint, loss='mse', device='cpu', lr=0.0001, step_size=1, gamma=0.95):
        super().__init__()

        # Checkpoints directory
        self.save_dir = checkpoint
        if not os.path.exists(self.save_dir): 
            os.makedirs(self.save_dir) 
       
        # Initialization
        self.set_network(device=device)
        self.set_optimizer(lr=lr, step_size=step_size, gamma=gamma)
        self.set_criterions(loss=loss, device=device)
        self.epoch = 0
    # ----------------------------------------------------------------------------------------------------
    def __call__(self, test_data, verbose=True, criterions=['mle'], **kwargs):
        start_time = time.time()
        self.set_inputs(test_data)                                              # set inputs
        self.forward(isTraining=False, **kwargs)                            # forward pass
        et = time.time() - start_time                                           # execution time

        evalRes = self.eval(criterions=criterions)                              # calculate eval results    
        evalRes['et'] = float(et)

        # Display Results     
        if verbose: 
            label = test_data['label'] if 'label' in test_data else ''
            self.print_results(str(label), evalRes)

        # Output data from batch
        output_data = dict()
        for key, val in self.batch.items():
            try: output_data[key] = val.squeeze(0)
            except: output_data[key] = val

        # torch.cuda.empty_cache()
        return output_data, evalRes 
    # ----------------------------------------------------------------------------------------------------
    def set_network(self, device='cpu'):
        self.device = device
        self.net = regViTNet(   image_h=224, image_w=224, in_chans=3, 
                                patch_size=16, stride=8, num_layers=5, num_heads=2, embed_dim=256, dropout=0., theta_lr=.1, 
                                bias=False, pos_embedding='sine2D', device=self.device, crop_margin=.1)        
    # ----------------------------------------------------------------------------------------------------
    def set_optimizer(self, lr=0.0001, step_size=1, gamma=0.95):
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma) 
    # ---------------------------------------------------------------------------------------------------------
    def set_criterions(self, loss='mse', grid_size=120, device='cpu'):
        self.tnfGridLoss = TransformedGridLoss(grid_size=grid_size, device=device) 
        self.MSE = MSE(device=device)
        self.DICE = Dice(device=device) 
        self.MLE = MLE(device=device)
        self.HD = HD(device=device)
        self.HD95 = HD95(device=device)
        self.SDM = SDM(device=device)
        self.LOSS = WeightedLoss(loss)    
    # ----------------------------------------------------------------------------------------------------
    def set_inputs(self, input_data):
        '''
            INPUT PARAMS (NOTE [n=0,1,...])
                [REQUIRED] target_image, target_mask, source_image_n, source_mask_GT_n            
                [OPTIONAL] theta_GT_n, target_landmarks, source_landmarks_n, source_dpi, target_dpi 
        '''
        # Get number of source images
        if 'num_sources' in input_data:
            self.Nsource = input_data['num_sources']
        else:
            self.Nsource = 0
            while True:
                self.Nsource += 1
                if f'source_image_{self.Nsource}' not in input_data: break

        # data to batch (input data can be batch or single data)  
        self.batch = input_data.copy() 
        if input_data['target_image'].dim() == 3:
            for key, val in self.batch.items():
                try: self.batch[key] = val.unsqueeze(0)
                except: pass      
    # ----------------------------------------------------------------------------------------------------
    def forward(self, isTraining=False, **kwargs):
        '''
            ADDED PARAMS (NOTE [n=0,1,...])
                [REQUIRED] theta_n, warped_image_n, warped_mask_GT_n 
                [OPTIONAL] warped_dpi, warped_landmarks_n
        '''        
        if isTraining: self.net.train()
        else: self.net.eval()
        theta_list = self.net(  self.batch["target_image"], 
                                [self.batch[f"source_image_{n}"] for n in range(self.Nsource)],
                                source_masks = [self.batch[f"source_mask_GT_{n}"] for n in range(self.Nsource)],
                                source_dpi = self.batch[f"source_dpi"] if 'source_dpi' in self.batch else None,
                                target_dpi = self.batch[f"target_dpi"] if 'target_dpi' in self.batch else None,
                                **kwargs)
        for n, theta_n in enumerate(theta_list): self.batch[f'theta_{n}'] = theta_n  

        # Warp data -----
        source_size = self.batch['source_image_0'].size()[2:]            
        # Delete previous warped data  
        for key in self.batch.keys():
            if 'warped' in key: self.batch.pop(key)  
        # dpi
        if 'source_dpi' in self.batch:
            self.batch['warped_dpi'] = np.array(self.batch['source_dpi']) 
        # images
        self.batch[f'warped_image'] = 0 
        for n in range(self.Nsource):
            self.batch[f'warped_image_{n}'] = self.net.geoTransformer(self.batch[f'source_image_{n}'], self.batch[ f'theta_{n}'], output_size=source_size)  
            self.batch[f'warped_image'] += self.batch[f'warped_image_{n}']
        # GT masks
        self.batch[f'warped_mask_GT'] = 0
        for n in range(self.Nsource):
            if f'source_mask_GT_{n}' in self.batch:
                self.batch[f'warped_mask_GT_{n}'] = self.net.geoTransformer( self.batch[f'source_mask_GT_{n}'], self.batch[ f'theta_{n}'], output_size=source_size, asBinary=True)  
                self.batch[f'warped_mask_GT'] += self.batch[f'warped_mask_GT_{n}']
        self.batch[f'warped_mask_GT'][self.batch[f'warped_mask_GT'] > 255.0] = 255.0
        # landmarks (NOTE: only for test data with self.batch=1)
        warped_landmarks_list = list()
        for n in range(self.Nsource):
            if f'source_landmarks_{n}' in self.batch:
                if self.batch[ f'theta_{n}'] is None: continue
                self.batch[f'warped_landmarks_{n}'] = self.net.lmkTransformer(self.batch[f'source_landmarks_{n}'].squeeze(0), self.batch[ f'theta_{n}'].squeeze(0), source_size)
                warped_landmarks_list.append( self.batch[f'warped_landmarks_{n}'])
        if len(warped_landmarks_list)>0: self.batch['warped_landmarks'] = torch.cat(warped_landmarks_list)       
    # ----------------------------------------------------------------------------------------------------
    def backward(self):
        self.optimizer.zero_grad()                                          # set gradients to zero     
        self.evalRes_tensor['loss'].backward(retain_graph=True)             # calculate gradients
        self.optimizer.step()          
    # ----------------------------------------------------------------------------------------------------
    def eval(self, criterions):
        self.evalRes_tensor = dict()
        batch = dict()

        for key, val in self.batch.items():
            if any([k in key for k in ['warped_image', 'warped_mask_GT']]): val = T.Resize(size = self.batch['target_image'].size()[-2:])(val)
            batch[key] = val
        if 'tnfGridLoss' in criterions: self.evalRes_tensor['tnfGridLoss'] = torch.sum(torch.stack( [self.tnfGridLoss(batch[f'theta_{n}'], batch[f'theta_GT_{n}']) for n in range(self.Nsource)]  ))
        if 'mse' in criterions: self.evalRes_tensor['mse'] = self.MSE( batch['target_image'], batch['warped_image'] )            
        if 'hd' in criterions: self.evalRes_tensor['hd'] = self.HD( batch['target_image'], batch['warped_image'], batch['target_dpi'])                                       
        if 'hd95' in criterions: self.evalRes_tensor['hd95'] = self.HD95( batch['target_image'], batch['warped_image'], batch['target_dpi'])          
        if 'dice' in criterions: self.evalRes_tensor['dice'] = self.DICE( batch['target_mask_GT'], batch['warped_mask_GT'])    
        if 'diceLoss' in criterions: self.evalRes_tensor['diceLoss'] = 1-self.DICE( batch['target_mask_GT'], batch['warped_mask_GT'])   
        if 'diceSrc' in criterions: self.evalRes_tensor['diceSrc'] = self.DICE( batch['warped_mask_GT_0'], batch['warped_mask_GT_1'])   
        if 'sdm' in criterions: 
            warped_mask_GT_list = [batch[f'warped_mask_GT_{n}'] for n in range(self.batch['num_sources'])]            
            self.evalRes_tensor['sdm'] = self.SDM(batch['warped_dpi'], *warped_mask_GT_list)
        if 'mle' in criterions: 
            if all([k in batch for k in ['target_landmarks','warped_landmarks','target_dpi','warped_dpi']]):
                mle = self.MLE(batch['target_landmarks'].unsqueeze(0), batch['warped_landmarks'].unsqueeze(0), batch['target_dpi'], batch['warped_dpi'])                
                self.evalRes_tensor['mle'] = mle.nanmean()
                self.evalRes_tensor['mle_list'] = mle             

        try: self.evalRes_tensor['loss'] = self.LOSS(self.evalRes_tensor)  
        except: self.evalRes_tensor['loss'] = None

        # evalRes (numpy)
        evalRes_npy = dict()
        for key, val in  self.evalRes_tensor.items(): 
            try:  val = val.data.cpu().numpy()
            except: pass            
            if val is not None: evalRes_npy[key] = val

        return evalRes_npy                                                                  
    # ----------------------------------------------------------------------------------------------------
    def run_singleEpoch(self, dataset, isTraining=False, num_workers=4, batch_size=1, shuffle=False, criterions=[]):   
        criterions_ = [*criterions, *self.LOSS.metrics]                             # make sure all criterions for loss are considered to be calculated
        evalRes_epoch = list()                                              # epoch eval
        
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers):
            self.set_inputs(batch)
            self.forward(isTraining=isTraining)                                     # forward pass
            evalRes_batch = self.eval(criterions=criterions_)                       # calculate eval results 
            if isTraining: self.backward()                                          # get gradients and update network weights 
            evalRes_epoch.append(evalRes_batch) # Batch evaluation

        evalRes_epoch = pd.DataFrame(evalRes_epoch)
        evalRes_epoch = evalRes_epoch.apply(lambda col: col.map(lambda x: x.item() if isinstance(x, np.ndarray) else x))        
        evalRes_epoch = evalRes_epoch.mean().to_dict()                              # Results (epoch evaluation)

        return evalRes_epoch         
    # ----------------------------------------------------------------------------------------------------
    def train(self, train_dataset, verbose=True, **kwargs):
        evalRes_epoch = self.run_singleEpoch(train_dataset, isTraining=True, **kwargs)
        self.scheduler.step() # Update scheduler and epoch
        self.epoch += 1  
        
        # Display Results 
        if verbose: 
            print( f'Epoch {self.epoch}' )     
            self.print_results('Train', evalRes_epoch)

        return evalRes_epoch
    # ----------------------------------------------------------------------------------------------------
    def validate(self, test_dataset, verbose=True, **kwargs):
        evalRes_epoch = self.run_singleEpoch( test_dataset, isTraining=False, **kwargs)  
        if verbose: self.print_results('Test', evalRes_epoch)   # Display Results     

        return evalRes_epoch
    # ----------------------------------------------------------------------------------------------------
    def save_states(self, save_dir=None, filename='net'):    
        if save_dir is None: save_dir = self.save_dir   
        save_filename = f"{filename}.pth"
        save_path = os.path.join(save_dir, save_filename)
        state = {
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),         
            'epoch': self.epoch
            }        

        torch.save(state, save_path)       
    # ----------------------------------------------------------------------------------------------------
    def load_states(self, load_dir=None, filename='net'):
        if load_dir is None: load_dir = self.save_dir
        load_path = os.path.join(load_dir, f"{filename}.pth")

        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
            self.net.load_state_dict( checkpoint['net_state_dict'] )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
    # ----------------------------------------------------------------------------------------------------
    def print_results(self, header, evalRes, N=8):
        txt = "\t" + header + ' '*(N-len(header)) + ':'
        txt_last = ''

        for key, val in evalRes.items(): 
            if key == 'mle_list': 
                # txt_last = f"{key}={val}" 
                continue
            else: txt += f"\t {key}={val:.6f}"
        
        print( f"{txt} \t {txt_last}")
####################################################################################################################################

