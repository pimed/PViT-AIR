import argparse 
from utils.options import set_options
from data.synth_dataset import synthDataset
from models import regModel
from utils.result_logger import Logger

  
#------------------------------------------------------------------------------------------
if __name__ == "__main__":
#------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # NOTE: Update directories
    parser.add_argument('--dataset', type=str, default='path/to/train/dataset/csv', help='Path to catalog of dataset')     
    parser.add_argument('--save_dir', type=str, default='path/to/checkpoint/directory', help='Directory to save results')     
    #
    parser.add_argument('--silent', action='store_true', help='if specified, stops displaying results and debugging information')                
    parser.add_argument('--device', type=int, default=-1, help='gpu ids: e.g. 0,1,2. use -1 for CPU')
    parser.add_argument('--seed', type=int, default=42, help='Pseudo-RNG seed')     
    parser.add_argument('--image_w', type=int, default=224, help='image width')
    parser.add_argument('--image_h', type=int, default=224, help='image height')
    parser.add_argument('--rot_range', type=int, default=20, help='range of rotation angle for synthetic dataset (-/+ degree)')
    parser.add_argument('--move_range', type=float, default=.1, help='range of image shifting coefficients for synthetic dataset (ratio of movement to image size)')
    parser.add_argument('--scale_range', type=float, default=.1, help='range of image scaling coefficients for synthetic dataset') # NOTE: 1-scale_range:1+scale_range
    parser.add_argument('--shear_range', type=float, default=.05, help='range of image shearing coefficients for synthetic dataset (ratio of movement to image size)')
    parser.add_argument('--loss', type=str, default='mse', help='type of loss [tnfGrid|tnf|mse|dice|lde]')       # NOTE: for more than one loss function use "+". e.g.: 2*tnfGrid+1*ssd or alternate_tnfGrid+ssd
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma for scheduler')
    parser.add_argument('--step_size', type=int, default=1, help='step size for scheduler')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker of multi-process data loader')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs') 
    parser.add_argument('--split', type=float, default=.7, help='split ratio of training-test datasets') 
    opt = parser.parse_args()                  
    opt = set_options(opt, save_dir=opt.save_dir)


    print('Creating Dataset ...')
    train_dataset = synthDataset( opt.dataset, device=opt.device, filter=True,
                                  scale_range=opt.scale_range, rot_range=opt.rot_range, move_range=opt.move_range, 
                                  shear_range=opt.shear_range, image_h=opt.image_h, image_w=opt.image_w)
    train_dataset, val_dataset = train_dataset.split(opt.split)    
    

    print('Creating Model ...')
    model = regModel(opt.save_dir, loss=opt.loss, device=opt.device, lr=opt.lr, step_size=opt.step_size, gamma=opt.gamma)

    print('Start training ...')
    trainRes_logger = Logger(opt.save_dir, 'training_evalRes')
    valRes_logger = Logger(opt.save_dir, 'validation_evalRes')


    while model.epoch < opt.num_epochs:
        # Train
        trainRes = model.train(train_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, verbose=not opt.silent)   
        trainRes_logger.log(trainRes, index=model.epoch)                                        
        model.save_states()                                                 

        # Validation
        valRes = model.validate(val_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, verbose=not opt.silent)  
        valRes_logger.log(valRes, index=model.epoch)                          
        
    print('Done!')