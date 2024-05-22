import glob
import scipy.io as sio

import torch
from torch.utils.data.dataloader import DataLoader

import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/../../')
from snmi.core import Trainer, BaseDataset
from snmi.models import SegModel
from snmi.nets.unet2D import Unet2D
from snmi.utils import augmentation_methods as A, loss_fuctions as LF, process_methods as P, utils as U


class cfg:
    # training parameters
    epochs = 100
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
    val_loss_key='loss' # key of evaluation result to control early stop and best ckpt save
    early_stop_patience=None
    learning_rate_schedule=None

    # path settings TODO: chagne to your path
    data_path = 'data/skin'
    output_path = 'results/seg2D_example'

    # set suffix for image and label (the difference between image path and label path) for data loading
    img_suffix = '.jpg'
    lab_suffix = '_segmentation.png'

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
data_list = glob.glob(cfg.data_path + '/*.jpg')
train = data_list[:60]
valid = data_list[60:70]
test = data_list[70:]

# build pytorch dataset, see core/basic_dataset
train_set = BaseDataset(train, [cfg.img_suffix, cfg.lab_suffix], cfg.pre, cfg.aug)
valid_set = BaseDataset(valid, [cfg.img_suffix, cfg.lab_suffix], cfg.pre)

# build pytorch data loader, shuffle train set while training
trainloader = DataLoader(train_set, batch_size=cfg.train_batch_size, shuffle=True)
validloader = DataLoader(valid_set, batch_size=cfg.eval_batch_size)


# set device to gpu if gpu is available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build 2D unet for 2 classes segmentation, the input channel is 3 since the data is rgb image 
net = Unet2D(n_classes=2, in_channels=3)
# init optimizer, adam is used here
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
# init the model, #TODO: more details see modelnet/model
model = SegModel(net, optimizer, cfg.img_suffix, cfg.lab_suffix, dropout_rate=cfg.dropout_rate, loss_functions=cfg.loss_function, device=device)


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


