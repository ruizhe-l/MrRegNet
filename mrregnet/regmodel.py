from snmi.models.base_model import *
from mrregnet.layers import SpatialTransformer, ResizeTransform
from mrregnet.regis_tools import *

class RegModel(BaseModel):
    def __init__(self, net, optimizer, img_suffix, source_key, target_key, 
                lab_suffix=None, weight_suffix=None, device=None, up_displacement=False,
                loss_functions={LF.NCC(win=3): 1}, 
                disp_loss_functions={LF.Grad(): 10},
                lab_loss_functions={LF.SoftDice(): 1.0}):
        super().__init__(net, optimizer, device)
        self.net = net
        self.source_img_suffix = source_key + img_suffix
        self.source_lab_suffix = source_key + lab_suffix
        self.target_img_suffix = target_key + img_suffix
        self.target_lab_suffix = target_key + lab_suffix if lab_suffix is not None else None
        self.target_weight_suffix = target_key + weight_suffix if weight_suffix is not None else None
        self.loss_function = loss_functions
        self.disp_loss_function = disp_loss_functions
        self.lab_loss_function = lab_loss_functions
        self.up_displacement = up_displacement

    def train_step(self, batch, epoch):
        source_img = batch[self.source_img_suffix].to(device=self.device)
        target_img = batch[self.target_img_suffix].to(device=self.device)

        self.net.train()
        self.optimizer.zero_grad()

        logits = self.net(source_img, target_img)
        loss, _, sim_loss, lab_loss, disp_loss = self.get_loss(batch, logits)
 
        loss.backward()
        self.optimizer.step()

        loss = loss.cpu().detach().numpy()
        sim_loss = sim_loss.cpu().detach().numpy() if type(sim_loss) not in [float] else sim_loss
        lab_loss = lab_loss.cpu().detach().numpy() if type(lab_loss) not in [float] else lab_loss
        disp_loss = disp_loss.cpu().detach().numpy() if type(disp_loss) not in [int, float] else disp_loss
        return [loss, sim_loss, lab_loss, disp_loss]

    def get_loss(self, batch, logits):
        source_img = batch[self.source_img_suffix].to(device=self.device)
        target_img = batch[self.target_img_suffix].to(device=self.device)
        disps, disps_res = logits

        sim_loss = 0
        lab_loss = 0
        count_layer = 0
        for i, cur_d in enumerate(disps):
            # count used layers
            any_loss = 0
            for lf in self.loss_function:
                cur_w = self.loss_function[lf][i] if type(self.loss_function[lf]) is list else self.loss_function[lf]
                any_loss += cur_w
            if any_loss == 0:
                continue
            count_layer += 1

            if not self.up_displacement:
                cur_source_img = F.interpolate(source_img, size=cur_d[0].shape[2:], mode='nearest')
                cur_target_img = F.interpolate(target_img, size=cur_d[0].shape[2:], mode='nearest') 
                wrapped_img = SpatialTransformer(size=cur_d[0].shape[2:])(cur_source_img, cur_d[0])
                data_dict = {'target': cur_target_img, 'logits': wrapped_img}
            else:
                cur_scale = [cur_d[0].shape[shapei]/source_img.shape[shapei] for shapei in range(2,cur_d[0].dim())]
                cur_rescaler = ResizeTransform(cur_scale, ndims=cur_d[0].dim()-2)
                cur_disp = cur_rescaler(cur_d[0])
                wrapped_img = SpatialTransformer(size=cur_disp.shape[2:])(source_img, cur_disp)
                data_dict = {'target': target_img, 'logits': wrapped_img}
            for lf in self.loss_function:
                cur_w = self.loss_function[lf][i] if type(self.loss_function[lf]) is list else self.loss_function[lf]
                sim_loss += lf(data_dict) * cur_w

            if self.lab_loss_function is not None:
                source_lab = batch[self.source_lab_suffix].to(device=self.device)
                target_lab = batch[self.target_lab_suffix].to(device=self.device)
                
                if not self.up_displacement:
                    cur_source_lab = F.interpolate(source_lab, size=cur_d[0].shape[2:], mode='nearest')
                    cur_target_lab = F.interpolate(target_lab, size=cur_d[0].shape[2:], mode='nearest')
                    wrapped_lab = SpatialTransformer(size=cur_d[0].shape[2:])(cur_source_lab, cur_d[0])
                    data_dict = {'target': cur_target_lab, 'logits': wrapped_lab}
                else:
                    wrapped_lab = SpatialTransformer(size=cur_disp.shape[2:])(source_lab, cur_disp)
                    data_dict = {'target': target_lab, 'logits': wrapped_lab}

                # if self.target_weight_suffix is not None:
                #     weight_map = batch[self.target_weight_suffix].to(device=self.device)
                #     cur_weight_lab = F.interpolate(weight_map, size=cur_d[0].shape[2:], mode='nearest')
                #     data_dict.update({'weight':cur_weight_lab})
                
                for llf in self.lab_loss_function:
                    cur_w = self.lab_loss_function[llf][i] if type(self.lab_loss_function[llf]) is list else self.lab_loss_function[llf]
                    lab_loss += llf(data_dict) * cur_w

        sim_loss /= count_layer
        lab_loss /= count_layer

        disp_loss = 0
        count_layer = 0
        for i, cur_dres in enumerate(disps_res):
            data_dict = {'logits': cur_dres[0]}
            for lf in self.disp_loss_function:
                cur_w = self.disp_loss_function[lf][i] if type(self.disp_loss_function[lf]) is list else self.disp_loss_function[lf]
                if cur_w > 0:
                    disp_loss += lf(data_dict) * cur_w
                    count_layer += 1
        disp_loss /= count_layer
        loss = sim_loss + lab_loss + disp_loss

        return loss, wrapped_img, sim_loss, lab_loss, disp_loss


    def eval_step(self, batch, **kwargs):
        self.net.eval() # switch to evaluation model

        source_img = batch[self.source_img_suffix].to(device=self.device)
        target_img = batch[self.target_img_suffix].to(device=self.device)
        source_lab = batch[self.source_lab_suffix].to(device=self.device)
        target_lab = batch[self.target_lab_suffix].to(device=self.device)
        

        with torch.no_grad():
            logits = self.net(source_img, target_img)

        loss, wrapped_img, sim_loss, lab_loss, disp_loss = self.get_loss(batch, logits)
        wrapped_lab = SpatialTransformer(size=source_lab.shape[2:], mode='nearest')(source_lab, logits[0][-1][0])
    
        loss = loss.cpu().numpy()
        sim_loss = sim_loss.cpu().numpy() if type(sim_loss) not in [float] else sim_loss
        disp_loss = disp_loss.cpu().numpy() if type(disp_loss) not in [int, float] else disp_loss
        lab_loss = lab_loss.cpu().numpy() if type(lab_loss) not in [float] else lab_loss
        jac = JacboianDetSitk(logits[0][-1][0].cpu().numpy())

        gncc = EF.gncc(target_img, wrapped_img).cpu().numpy()
        gncc_baseline = EF.gncc(target_img, source_img).cpu().numpy()
        eval_results = {'loss': loss,
                        # 'smooth': disp_loss,
                        # 'lab_loss': lab_loss,
                        'jac_0': np.sum(np.sum(jac < 0) / np.prod(jac.shape)),
                        'jac_std': np.std(jac),
                        'gncc': gncc.mean(),
                        'gncc_baseline': gncc_baseline.mean()
                        }
        # if np.isnan(loss):
        #     print('non')
        if self.source_lab_suffix is not None:
            dice_baseline = EF.dice_coefficient(target_lab, source_lab, ignore_background=True).cpu().numpy()
            dice = EF.dice_coefficient(target_lab, wrapped_lab, ignore_background=True).cpu().numpy()
            eval_results.update({'dice': dice, 'dice_baseline': dice_baseline})

        log_disp = kwargs.get('log_disp', False)
        if log_disp: 
            disp = logits[0][-1][0].cpu().numpy()
            eval_results.update({'disp': disp})

        log_image = kwargs.get('log_image', False)

        if log_image:
            source_img = source_img.cpu().numpy()
            target_img = target_img.cpu().numpy()
            wrapped_img = wrapped_img.cpu().numpy()
            img_in = [source_img, target_img, wrapped_img]
            if self.source_lab_suffix is not None:
                
                source_lab = source_lab.cpu().numpy().argmax(1)
                target_lab = target_lab.cpu().numpy().argmax(1)
                wrapped_lab = wrapped_lab.cpu().numpy().argmax(1)
                img_in = img_in + [source_lab, target_lab, wrapped_lab]
            if self.target_weight_suffix is not None:
                    weight_map = batch[self.target_weight_suffix].numpy()
                    img_in = img_in + [weight_map]

            if len(source_img.shape) == 5:
                imgs = self.get_imgs_eval(img_in, tag='source-target-wrapped & lab')
            # else:
            #     heatmap_res_img = [vis_heatmap(x[0].cpu().numpy(), source_img.shape[2:]) for x in logits[1]]
            #     heatmap_img = [vis_heatmap(x[0].cpu().numpy(), source_img.shape[2:]) for x in logits[0]]
            #     base_grids = [draw_grid()]
            #     base_grid = torch.stack([grid_] * 5).unsqueeze(1)
            #     a = SpatialTransformer(size=logits[0][-1][0].shape[2:], mode='nearest')(grid_.to(device=self.device), F.interpolate(logits[0][-3][0], size=(256,256), mode='nearest')*2*2)
            #     b = a[0][0].cpu().numpy()
            #     cv2.imwrite('_test_fig/z_grid.jpg', cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
            #     img = self.get_imgs_eval(img_in, 
            #                             tag='source-target-wrapped & lab') 
            #     img1 = self.get_imgs_eval(heatmap_res_img, 
            #                             tag='disps_res_heatmap') 
            #     img2 = self.get_imgs_eval(disps_res_grid, 
            #                             tag='disps_res_grid') 
            #     img3 = self.get_imgs_eval(heatmap_img, 
            #                             tag='disps_heatmap') 
            #     img4 = self.get_imgs_eval(disps_grid, 
            #                             tag='disps_grid') 
            #     imgs = img
            #     imgs.update(img1), imgs.update(img2), imgs.update(img3), imgs.update(img4)
            else:
                
                disps_res_vmax = np.max([(torch.sqrt(x[0][:,0,...] ** 2 + x[0][:,1,...] ** 2)).view([x[0].shape[0],-1]).max(1)[0].cpu().numpy() * (source_img.shape[2] / x[0].shape[2]) for x in logits[1]], 0)
                disps_res_img = [vis_disps(x[0].cpu().numpy(), source_img.shape[2:], vmin=(0,) * source_img.shape[0], vmax=disps_res_vmax) for x in logits[1]]
                disps_vmax = np.max([(torch.sqrt(x[0][:,0,...] ** 2 + x[0][:,1,...] ** 2)).view([x[0].shape[0],-1]).max(1)[0].cpu().numpy() * (source_img.shape[2] / x[0].shape[2]) for x in logits[0]], 0)
                disps_img = [vis_disps(x[0].cpu().numpy(), source_img.shape[2:], vmin=(0,) * source_img.shape[0], vmax=disps_vmax) for x in logits[0]]
                disps_res_quiver = [x[0] for x in disps_res_img]
                disps_res_grid = [x[1] for x in disps_res_img]
                disps_quiver = [x[0] for x in disps_img]
                disps_grid = [x[1] for x in disps_img]

                img = self.get_imgs_eval(img_in, 
                                        tag='source-target-wrapped & lab') 
                img1 = self.get_imgs_eval(disps_res_quiver, 
                                        tag='disps_res_quiver') 
                img2 = self.get_imgs_eval(disps_res_grid, 
                                        tag='disps_res_grid') 
                img3 = self.get_imgs_eval(disps_quiver, 
                                        tag='disps_quiver') 
                img4 = self.get_imgs_eval(disps_grid, 
                                        tag='disps_grid') 
                imgs = img
                imgs.update(img1), imgs.update(img2), imgs.update(img3), imgs.update(img4)
            eval_results.update({'image': imgs})

        return eval_results


    def predict(self, batch, need_org=False):
        source_img = batch[self.source_img_suffix].to(self.device)
        target_img = batch[self.target_img_suffix].to(self.device)
        source_lab = batch[self.source_lab_suffix].to(self.device)

        self.net.eval()
        with torch.no_grad():
            disp = self.net(source_img, target_img)[0][-1][0]
        wrapped_lab = SpatialTransformer(size=disp.shape[2:])(source_lab, disp)
        wrapped_lab = wrapped_lab.cpu().numpy()
        if not need_org:
            return wrapped_lab
        else:
            wrapped_img = SpatialTransformer(size=disp.shape[2:])(source_img, disp)
            wrapped_img = wrapped_img.cpu().numpy()
            return wrapped_img, wrapped_lab

 