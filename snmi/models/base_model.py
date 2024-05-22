from abc import ABCMeta, abstractmethod
import numpy as np

import torch
import torch.nn.functional as F

from ..utils import eval_functions as EF, loss_fuctions as LF, utils as U



""" Abstract class
    Custom model should inherit this class
    function train_step and eval_step must be defined

"""
class BaseModel(metaclass=ABCMeta):
    def __init__(self, net, optimizer, device=None):

        self.net = net
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # move to device
        nets = net if type(net) in [list, tuple] else [net]
        [n.to(self.device) for n in nets]
        

    @abstractmethod
    def train_step(self, batch, epoch):
        """
        """

    @abstractmethod
    def eval_step(self, batch, **kwargs):
        """
        """

    def get_imgs_eval(self, imgs, tag='imgs'):
        assert type(imgs) is list, 'Error: input should be a list of numpy array'
        # check the images are 2d or 3d
        if len(np.shape(imgs[0])) == 4:
            img = U.combine_2d_imgs_from_tensor(imgs, channel_first=True)
        elif len(np.shape(imgs[0])) == 5:
            img = U.combine_3d_imgs_from_tensor(imgs, vis_slices=6, channel_first=True)
        else:
            raise('Error: image dimension should be 4 (2d) or 5 (3d)!')
        return {tag: img}


class SegModel(BaseModel):

    # init function, custom according to requirements
    def __init__(self, net, optimizer, img_suffix, lab_suffix, dropout_rate, 
                loss_functions={LF.CrossEntropy(): 0.5, LF.SoftDice(): 0.5}, 
                device=None):
        super().__init__(net, optimizer, device)
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        self.dropout_rate = dropout_rate
        self.loss_function = loss_functions

    def predict(self, batch):
        img = batch[self.img_suffix].to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(img)
        pred = F.softmax(logits, dim=1)
        pred = pred.cpu().numpy()
        return pred

    # TODO: must be defined for training. 
    # network forward and backward need to be done here.
    def train_step(self, batch, epoch):
        img = batch[self.img_suffix]
        img = img.to(device=self.device)

        # switch to train model, zero gradient for optimizer
        self.net.train()
        self.optimizer.zero_grad()

        # forward + backward + optimize
        logits = self.net(img, self.dropout_rate)
        loss = self.get_loss(batch, logits)   # calculate loss
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().numpy() # return loss for logging

    # loss calculation, can be modified as required
    def get_loss(self, batch, logits):
        lab = batch[self.lab_suffix]
        lab = lab.to(device=self.device)
        data_dict = {'logits': logits, 'target': lab}

        loss = 0
        for lf in self.loss_function:
            loss += lf(data_dict) * self.loss_function[lf]

        return loss

    # TODO: must be defined for evaluation.
    # evaluate and return results dict {metric1: value, metric2: [value] ...}
    def eval_step(self, batch, **kwargs):
        self.net.eval() # switch to evaluation model

        img = batch[self.img_suffix]
        lab = batch[self.lab_suffix]
        img_g = img.to(device=self.device)
        lab_g = lab.to(device=self.device)

        with torch.no_grad():
            logits = self.net(img_g)
            loss = self.get_loss(batch, logits).cpu().numpy()

        pred = F.softmax(logits, dim=1)
        pred = pred.argmax(dim=1)
        pred = torch.zeros_like(lab_g).scatter_(1, pred.unsqueeze(1), 1.)
        dice = EF.dice_coefficient(pred, lab_g, ignore_background=True).cpu().numpy()
        eval_results = {'loss': loss,
                        'dice': dice}
        
        # generate image results for logger and visulization
        # the key of image results must be 'image' or 'img'
        log_image = kwargs.get('log_image', False)
        if log_image:
            # tensor to numpy
            img = img.numpy()
            lab = np.argmax(lab.numpy(), 1, keepdims=True)
            pred = np.argmax(pred.cpu().numpy(), 1, keepdims=True)
            img = self.get_imgs_eval([img, lab, pred]) # create drawable image, see more details inside the function
            eval_results.update({'image': img})

        # TODO: custom argument for trainer.test(), for example:
        # in main code,trainer.test(..., need_logits=True) was called
        # this file add:
        # need_logits = kwargs.get('need_logits', False)
        # if need_logits: 
        #     eval_results.update({'logits': logits.cpu().numpy()})

        return eval_results
        

    def get_imgs_eval(self, imgs):
        # check the images are 2d or 3d
        if len(np.shape(imgs[0])) == 4:
            return self.get_imgs_eval_2d(imgs)
        elif len(np.shape(imgs[0])) == 5:
            return self.get_imgs_eval_3d(imgs)
        else:
            raise('Error: image dimension should be 4 (2d) or 5 (3d)!')

    # TODO: modify as required
    # return dict {'name': [drawable image]}
    def get_imgs_eval_2d(self, imgs):
        img, lab, pred = imgs
        combine_lab = U.gray_img_blend(lab, pred, channel_first=True) # blend prediction and ground truth image
        # combine images to one image, if channel_first, image shape should be [B,C,H,W], otherise [B,H,W,C] (batch, height, width, channel) 
        img = U.combine_2d_imgs_from_tensor([img, lab, pred, combine_lab], channel_first=True)
        img_dict = {'img-lab-pred-combined': img}
        return img_dict

    def get_imgs_eval_3d(self, imgs):
        img, lab, pred = imgs
        # for 3d image, only visulize [vis_slices] slices, image shape should be [B,C,H,W,D], otherwise [B,H,W,D,C] with channel_first=False
        img = U.combine_3d_imgs_from_tensor([img, lab, pred], vis_slices=6, channel_first=True)
        img_dict = {'img-lab-pred': img}
        return img_dict



class ClsModel(BaseModel):

    def __init__(self, net, optimizer, device, img_suffix, lab_suffix, dropout_rate, loss_functions={LF.CrossEntropy(): 0.5}):
        super().__init__(net, optimizer, device)
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        self.dropout_rate = dropout_rate
        self.loss_function = loss_functions


    def train_step(self, batch, epoch):
        img = batch[self.img_suffix]
        img = img.to(device=self.device)

        self.net.train()
        self.optimizer.zero_grad()

        logits = self.net(img, self.dropout_rate)
        loss = self.get_loss(batch, logits) 
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().numpy()

    def get_loss(self, batch, logits):
        lab = batch[self.lab_suffix]
        lab = lab.to(device=self.device)
        data_dict = {'logits': logits, 'target': lab}

        loss = 0
        for lf in self.loss_function:
            loss += lf(data_dict) * self.loss_function[lf]

        return loss

    def eval_step(self, batch, **kwargs):
        self.net.eval()

        img = batch[self.img_suffix]
        lab = batch[self.lab_suffix]
        img_g = img.to(device=self.device)
        lab_g = lab.to(device=self.device)

        with torch.no_grad():
            logits = self.net(img_g)
            loss = self.get_loss(batch, logits).cpu().numpy()

        pred = F.softmax(logits, dim=1)
        acc = EF.accuracy(pred, lab_g)
        eval_results = {'loss': loss,
                        'acc': acc}

        return eval_results

