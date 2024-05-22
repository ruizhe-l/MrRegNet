import glob
import scipy.io as sio

import torch
from torch.utils.data.dataloader import DataLoader

import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/../../')
from snmi.core import Trainer, BaseDataset
from snmi.nets import SegModel
from snmi.nets.unet2D import Unet3D
from snmi.utils import augmentation_methods as A, loss_fuctions as LF, process_methods as P, utils as U


class cfg:
    # training parameters
    epochs = 5
    learning_rate = 1e-4
    train_batch_size = 5
    eval_batch_size = 10
    dropout_rate = 0.2

    # set loss functions
    loss_function = {LF.CrossEntropy(): 0.5, LF.SoftDice(): 0.5} # loss functions {method: weight}
    
    # training setttings
    test_only = False
    log_train_image = False
    log_validation_image = True
    save_frequency=None
    val_loss_key='loss'
    early_stop_patience=None
    learning_rate_schedule=None

    # path settings TODO: chagne to your path
    data_path = 'OASIS'
    output_path = 'results/seg3D_example'

    # set suffix for image and label (the difference between image path and label path) for data loading
    img_suffix = 'norm.nii.gz'
    lab_suffix = 'seg4.nii.gz'

    # set pre-process functions for image and label, #TODO: more method see utils/process_methods
    # functions in torchvision.transforms also works here, see https://pytorch.org/vision/stable/transforms.html#functional-transforms
    pre = {img_suffix: [P.Resize([128,128]), 
                P.min_max,
                P.Transpose([2,0,1]),
                ],
            lab_suffix: [P.Resize([128,128], nearest=True),
                        P.OneHot([0,255])
            ]}
    # set data augmentation method
    aug = A.RandomAugmentation(img_suffix, lab_suffix)


U.config_to_txt(vars(cfg), f'{cfg.output_path}/config.txt') # save config to file [config.txt] 

# list all data and split to train (60) / valid (10) / test (30)
data_list = glob.glob(data_path + '/**/norm.nii.gz')
train = data_list[:6]
valid = data_list[6:8]
test = data_list[8:]


# set device to gpu if gpu is available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build pytorch dataset, see core/basic_dataset
train_set = BaseDataset(train, [cfg.img_suffix, cfg.lab_suffix], cfg.pre, cfg.aug)
valid_set = BaseDataset(valid, [cfg.img_suffix, cfg.lab_suffix], cfg.pre, cfg.aug)

# build pytorch data loader, shuffle train set while training
trainloader = DataLoader(train_set, batch_size=cfg.train_batch_size, shuffle=True)
validloader = DataLoader(valid_set, batch_size=cfg.eval_batch_size)



# build 2D unet for 2 classes segmentation, the input channel is 3 since the data is rgb image 
net = Unet3D(n_classes=5, in_channels=1)
# init optimizer, adam is used here
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
# init the model, #TODO: more details see modelnet/model
model = SegModel(net, optimizer, device, cfg.img_suffix, cfg.lab_suffix, dropout_rate=cfg.dropout_rate, loss_functions=cfg.loss_function)

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
        log_graph_input=random_img)

# test on test set
trainer.restore(cfg.output_path+'/ckpt/model_final.pt')
test_set = BaseDataset(test, [cfg.img_suffix, cfg.lab_suffix], cfg.pre)
testloader = DataLoader(test_set, batch_size=cfg.eval_batch_size)
test_results = trainer.test(testloader, cfg.output_path, log_image=True)
sio.savemat(f'{cfg.output_path}/test_results.mat', test_results)
with open(f'{cfg.output_path}/test_results.txt', 'a+') as f:
    f.write(U.dict_to_str(test_results) + '\n')


