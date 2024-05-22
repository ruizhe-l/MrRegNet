import os
import cv2
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class LogWriterTensorboard():
    def __init__(self, log_path):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.writer = SummaryWriter(log_path)

    def write_scalar(self, eval_dict, epoch, tag=''):
        for key in eval_dict:
            self.writer.add_scalar(os.path.join(key, tag), eval_dict[key], epoch)

    def write_image(self, img_dict, epoch, tag=''):
        for key in img_dict:
            self.writer.add_image(os.path.join(key, tag), img_dict[key], epoch, dataformats='HWC')

    def write_text(self, str, epoch, tag=''):
        self.writer.add_text(tag, str, epoch)

    def write_graph(self, net, input):
        self.writer.add_graph(net ,input)


class LogWriterFile():
    def __init__(self, log_path):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_path = log_path

    def write_scalar(self, eval_dict, epoch, tag=''):
        sub_path = f'{self.log_path}/plot'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        for key in eval_dict:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(eval_dict[key])
            fig.savefig(f'{sub_path}/{epoch}_{key}_{tag}.jpg', bbox_inches='tight')   # save the figure to file
            plt.close(fig)

    def write_image(self, img_dict, epoch, tag=''):
        sub_path = f'{self.log_path}/image'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        for key in img_dict:
            img = img_dict[key]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{sub_path}/{epoch}_{key}_{tag}.jpg', img)

    def write_text(self, str, epoch, tag=''):
        sub_path = f'{self.log_path}/txt'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        with open(f'{sub_path}/{epoch}_{tag}.txt', 'a+') as f:
            f.write(f'{epoch}-{tag}: {str}')
