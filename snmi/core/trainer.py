import os
import time
import numpy as np
from tqdm.auto import tqdm

import torch

from ..utils import utils as U
from ..utils import LogWriterTensorboard, LogWriterFile


tqdm_setting = {'unit': ' data', 
                'ascii': ' >=', 
                'bar_format': '{l_bar}{bar:10}{r_bar}'}


class Trainer:
    def __init__(self, model, log_writer=LogWriterTensorboard):
        self.model = model
        self.cur_epoch = 0
        self.log_writer = log_writer

    def train(self, 
            train_loader, 
            validation_loader, 
            epochs,   
            output_path,
            eval_frequency=1,
            save_frequency=None,
            val_loss_key='loss',
            early_stop_patience=None,
            learning_rate_schedule=None,
            log_train_image=False,
            log_validation_image=False,
            log_graph_input=None,
            print_tag=''):

        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_path = f'{output_path}/log/{timestr}'
        logwriter_train = self.log_writer(f'{log_path}/train')
        logwriter_valid = self.log_writer(f'{log_path}/valid')

        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        optimizers = self.model.optimizer if type(self.model.optimizer) is list else [self.model.optimizer]
        if log_graph_input is not None and isinstance(self.log_writer, LogWriterTensorboard):
            for i, net in enumerate(nets):
                logwriter_model = self.log_writer(f'{log_path}/net{i}')
                logwriter_model.write_graph(net, log_graph_input)
            logwriter_model.writer.close()

        best_val_loss = None
        early_stop_count = 0
        start_ep = self.cur_epoch
        for ep in range(start_ep, epochs):
            with tqdm(total=len(train_loader.dataset), desc=f'{print_tag}Epoch: {ep}/{epochs}, Training ', **tqdm_setting) as pbar:
                self.cur_epoch = ep
                # train and evaluation on training dataset
                ep_train_loss = []
                for batch in train_loader:
                    train_loss = self.model.train_step(batch, ep)
                    train_loss = [train_loss] if type(train_loss) is not list else train_loss
                    ep_train_loss.append(train_loss)
                    pbar.update(train_loader.batch_size)
                    
                    pbar.set_postfix_str(f'\tbatch - loss: {["{0:.4f}".format(x) for x in train_loss]}')
                
                lrs = [o.param_groups[0]['lr'] for o in optimizers]
                pbar.set_postfix_str(f'\ttotal - loss: {["{0:.4f}".format(x) for x in np.mean(ep_train_loss, 0)]}, lr: {lrs}')
                if learning_rate_schedule is not None:
                    learning_rate_schedule.step()

            # evaluation on validation dataset
            if eval_frequency is not None and (eval_frequency < 1 or ep % eval_frequency == 0 or ep == epochs - 1):
                # training set evaluation
                eval_train_results = self.eval(train_loader, desc=f'    Evaluation: training data   ', log_image=log_train_image)
                if log_train_image and 'image' in eval_train_results:
                    eval_train_image = eval_train_results.pop('image')
                    logwriter_train.write_image(eval_train_image, ep)
                logwriter_train.write_scalar(U.dict_mean(eval_train_results), ep)

                # validation set evaluation
                if validation_loader is not None:
                    eval_valid_results = self.eval(validation_loader, desc=f'    Evaluation: evaluation data ', log_image=log_validation_image)

                    if log_validation_image and 'image' in eval_valid_results:
                        eval_valid_image = eval_valid_results.pop('image')
                        logwriter_valid.write_image(eval_valid_image, ep)

                
                    logwriter_valid.write_scalar(U.dict_mean(eval_valid_results), ep)

                    # save best ckpt
                    if val_loss_key is not None and val_loss_key in eval_valid_results:
                        cur_val_loss = np.mean(eval_valid_results[val_loss_key])
                        if best_val_loss is None or cur_val_loss < best_val_loss:
                            self.save(f'{output_path}/ckpt/model_best.pt')
                            best_val_loss = cur_val_loss

                        # early stop
                        if early_stop_patience is not None:
                            if cur_val_loss > best_val_loss:
                                early_stop_count += 1
                            else:
                                early_stop_count = 0
                            if early_stop_count > early_stop_patience:
                                break

            # save checkpoint
            if save_frequency is not None and (save_frequency < 1 or ep % save_frequency == 0):
                self.save(f'{output_path}/ckpt/model_{ep}.pt')
            self.save(f'{output_path}/ckpt/model_final.pt')

        self.save(f'{output_path}/ckpt/model_final.pt')

    def eval(self, data_loader, **kwargs):
        desc = kwargs.get('desc', 'Evaluation: ')
        log_image = kwargs.get('log_image', False)

        with tqdm(total=len(data_loader.dataset), desc=desc, **tqdm_setting) as pbar:
            all_results = {}
            all_imgs = {}
            for batch in data_loader:
                results = self.model.eval_step(batch, **kwargs)
                if log_image and 'image' in results:
                    imgs = results.pop('image')
                    all_imgs = U.dict_concat(all_imgs, imgs)
                all_results = U.dict_concat(all_results, results)
                pbar.update(data_loader.batch_size)
                pbar.set_postfix_str('\tbatch - ' + U.dict_to_str(results))
            pbar.set_postfix_str('\ttotal - ' + U.dict_to_str(all_results))

        if log_image:
            all_results.update({'image': all_imgs})
        return all_results
        
    def save(self, ckpt_path):
        cur_state = {'epoch': self.cur_epoch}
        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        [cur_state.update({f'net{i}': nets[i].state_dict()}) for i in range(len(nets))]

        optimizers = self.model.optimizer if type(self.model.optimizer) is list else [self.model.optimizer]
        [cur_state.update({f'optimizer{i}': optimizers[i].state_dict()}) for i in range(len(optimizers))]

        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save(cur_state, ckpt_path)

    def restore(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.model.device)
        self.cur_epoch = ckpt['epoch']

        optimizers = self.model.optimizer if type(self.model.optimizer) is list else [self.model.optimizer]
        [optimizers[i].load_state_dict(ckpt[f'optimizer{i}']) for i in range(len(optimizers))]

        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        [nets[i].load_state_dict(ckpt[f'net{i}']) for i in range(len(nets))]

    def test(self, data_loader, output_path, **kwargs):


        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_name = kwargs.get('log_name', 'test')
        logwriter_test = self.log_writer(f'{output_path}/log/{timestr}/{log_name}')

        log_image = kwargs.get('log_image', False)
        if log_image:
            logwriterfile = LogWriterFile(f'{output_path}/test_images')
        desc = kwargs.get('desc', 'Evaluation: test data ')

        with tqdm(total=len(data_loader.dataset), desc=desc, **tqdm_setting) as pbar:
            test_results = {}
            i = 0
            for batch in data_loader:
                results = self.model.eval_step(batch, **kwargs)
                if log_image and 'image' in results:
                    imgs = results.pop('image')
                    logwriter_test.write_image(imgs, i)
                    for key in imgs:
                        logwriterfile.write_image({key:imgs[key]}, epoch=i)
                    

                test_results = U.dict_concat(test_results, results)
                pbar.update(data_loader.batch_size)
                pbar.set_postfix_str('\tbatch - ' + U.dict_to_str(results))
                i += 1
            str_test_results = U.dict_to_str(test_results)
            pbar.set_postfix_str('\ttotal - ' + str_test_results)

        logwriter_test.write_text(str_test_results, self.cur_epoch)

        return test_results


        

