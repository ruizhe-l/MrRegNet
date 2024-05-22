import glob
import scipy.io as sio
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

from snmi.core import Trainer
from snmi.utils import augmentation_methods as A, loss_fuctions as LF, process_methods as P, utils as U, LogWriterTensorboard

from mrregnet.reg_dataset import RegDataset
from mrregnet.net_reg_3D import MrReg
from mrregnet.regmodel import RegModel


class cfg:
    data_path = './_testdata/'
    output_path = './results/test_3D'

    # training parameters
    epochs = 200
    learning_rate = 1e-4
    train_batch_size = 1
    eval_batch_size = 1
    eval_frequency = 5

    # set loss functions
    loss_functions = {LF.GNCC(): 1} # loss_functions = {LF.NCC(win=3): 1}  
    disp_loss_functions = {LF.Grad('l2'): [16,8,4,2]} # disp_loss_functions = {LF.Grad('l2'): [1,1,1,1]} (if uinsg NCC loss)
    lab_loss_functions = None # lab_loss_functions = {LF.RegDice(): 1} (if use label as guidance)
    n_layers = 4
    int_steps = 0
    
    # training setttings
    test_only = False
    log_train_image = False
    log_validation_image = False
    save_frequency=None
    val_loss_key='loss' # key of evaluation result to control early stop and best ckpt save
    early_stop_patience=None
    learning_rate_schedule=None

    # set suffix for image and label (the difference between image path and label path) for data loading
    img_suffix = 'norm.nii.gz'
    lab_suffix = 'seg4.nii.gz'
    source_key = 'source'
    target_key = 'target'

    # set pre-process functions for image and label, #TODO: more method see utils/process_methods
    # functions in torchvision.transforms also works here, see https://pytorch.org/vision/stable/transforms.html#functional-transforms
    pre = {img_suffix: [P.min_max, P.ExpandDim(0)],
            lab_suffix: [P.OneHot([0,1,2,3,4])]}


if not cfg.test_only:
    U.config_to_txt(vars(cfg), f'{cfg.output_path}/config.txt') # save config to file [config.txt] 

# list all data and split to train (5) / valid (2) / test (3)
data = glob.glob(f'{cfg.data_path}/**/*{cfg.img_suffix}')
train = data[:5]
valid = data[5:7]
test = data[7:]

source_test = test[-1:] # using {1} image as source image for test
test = sorted(test[:-1]*5) # repeat and sort the target images, make sure every source image register to every target image

target_train = train.copy()
target_valid = valid.copy()
tmp_random = np.random.RandomState(123)
tmp_random.shuffle(target_train)
tmp_random.shuffle(target_valid)

# build pytorch dataset, see core/basic_dataset
train_set = RegDataset(train, target_train, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix], cfg.pre)
valid_set = RegDataset(valid, target_valid, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix], cfg.pre)

# build pytorch data loader, shuffle train set while training
trainloader = DataLoader(train_set, batch_size=cfg.train_batch_size, shuffle=True)
validloader = DataLoader(valid_set, batch_size=cfg.eval_batch_size)


# set device to gpu if gpu is available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# build 2D unet for 2 classes segmentation, the input channel is 3 since the data is rgb image 
net = MrReg(n_layers=cfg.n_layers, int_steps=cfg.int_steps)
# init optimizer, adam is used here
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
# init the model, #TODO: more details see modelnet/model
model = RegModel(net, optimizer, cfg.img_suffix, cfg.source_key, cfg.target_key, lab_suffix=cfg.lab_suffix, device=device,
                    loss_functions=cfg.loss_functions, disp_loss_functions=cfg.disp_loss_functions, lab_loss_functions=cfg.lab_loss_functions)

# get a random image for graph draw in tensorboard (no need to do if don't need it)
# random_img = torch.tensor([train_set[0][img_suffix]]).to(device)
random_img = None

# init train and start train
trainer = Trainer(model)
if not cfg.test_only:
    trainer.train(trainloader, validloader, 
        epochs=cfg.epochs, 
        output_path=cfg.output_path, 
        log_train_image=cfg.log_train_image, 
        log_validation_image=cfg.log_validation_image,
        log_graph_input=random_img,
        eval_frequency=cfg.eval_frequency,
        log_writer=LogWriterTensorboard)

# test on test set
trainer.restore(cfg.output_path+'/ckpt/model_final.pt')
test_set = RegDataset(source_test, test, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix], cfg.pre)
testloader = DataLoader(test_set, batch_size=5)
test_results = trainer.test(testloader, cfg.output_path, log_image=cfg.log_validation_image)
sio.savemat(f'{cfg.output_path}/test_results.mat', test_results)
with open(f'{cfg.output_path}/test_results.txt', 'a+') as f:
    f.write(U.dict_to_str(test_results) + '\n')

