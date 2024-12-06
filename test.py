import argparse 

from utils.options import set_options
from data.real_dataset import realDataset
from models import load_model
from utils.result_logger import Logger



#------------------------------------------------------------------------------------------
if __name__ == "__main__":
#------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # NOTE: Update directories
    parser.add_argument('--dataset', type=str, default='path/to/test/dataset/csv', help='Path to catalog of dataset')     
    parser.add_argument('--checkpoint', type=str, default='path/to/checkpoint/directory', help='Directory to save results')   
    parser.add_argument('--save_dir', type=str, default='path/to/save/directory', help='Directory to save results')     
    #
    parser.add_argument('--silent', action='store_true', help='if specified, stops displaying results and debugging information')                
    parser.add_argument('--device', type=int, default=-1, help='gpu ids: e.g. 0,1,2. use -1 for CPU')
    parser.add_argument('--seed', type=int, default=42, help='Pseudo-RNG seed')     
    parser.add_argument('--criterions', nargs='+', default=['mle', 'dice', 'hd', 'hd95', 'ssim', 'mse', 'sdm'], help=f"List of criterions to use.")
    opt = parser.parse_args()                  
    opt = set_options(opt, save_dir=opt.save_dir)


    print('Creating Dataset ...')  
    test_dataset = realDataset(opt.dataset, device=opt.device, filter=True)

    print('Loading Model ...')
    model = load_model(opt.checkpoint, device=opt.device)

    print('Start testing ...')
    result_logger = Logger(opt.checkpoint, 'test_evalRes')
    for data in test_dataset:
        warped_data, evalRes = model(data, verbose=not opt.silent,  criterions=opt.criterions)    # keys in warped data : 'warped_image', 'warped_mask_GT', 'theta_0', 'theta_1', etc  
        result_logger.log(evalRes, index=data['label'])      
