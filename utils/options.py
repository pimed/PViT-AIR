
import torch 
import os 

####################################################################################################################################
def set_options(opt, save_dir=None, disp=True):
    if opt.device >= 0 and torch.cuda.is_available():
        if opt.device > torch.cuda.device_count():
            opt.device = "cuda:0"
        torch.cuda.set_device(opt.device)
        torch.cuda.manual_seed(opt.seed)
    else: opt.device = "cpu"

    # Display
    if disp:
        opt_str = ''
        opt_str += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            opt_str += '{:>25} : {:<30}\n'.format(str(k), str(v))
        opt_str += '----------------- End -------------------'        
        print(opt_str)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)            

        with open( os.path.join(save_dir, 'params.txt' ), 'wt') as file: 
            file.write(opt_str)    
    
    return opt
####################################################################################################################################
