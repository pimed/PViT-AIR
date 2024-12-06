import os
from .registration import regModel


####################################################################################################################################
def load_model(checkpoint_dir, device='cpu'):
    with open(os.path.join(checkpoint_dir, 'params.txt') , 'r') as f: 
        content = f.readlines()
    
    params = dict()
    for line in content[1:-1]:  
        line = line.strip().split('\t[default:')[0]
        idx =  line.find(':')
        key, val = line[:idx], line[idx+1:]
        key, val = key.strip(), val.strip()
        
        if val == 'True': val_ = True
        elif val == 'False': val_ = False
        elif val.isdigit(): val_ = int(val)
        else:
            try: val_ = float(val)
            except: val_ = val
        params[key] = val_

    model = regModel(params['save_dir'], loss=params['loss'], lr=params['lr'], step_size=params['step_size'], gamma=params['gamma'], device=device)
    model.load_states()  
    
    return model
####################################################################################################################################
