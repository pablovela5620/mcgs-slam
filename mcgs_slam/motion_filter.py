import cv2
import torch
import lietorch
import torch.nn.functional as F

import geom.projective_ops as pops
from modules.corr import CorrBlock

from torchvision import transforms


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, args=None, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.args = args
        self.device = device

        self.count = 0
        
        self.metric3d_model = None

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        # operator at 1/8 resolution
        ht = video.ht // 8
        wd = video.wd // 8
        self.coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        x = self.cnet(image)
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)
    
    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def __prior_extractor(self, im_tensor, intrinsic):
        
        if self.metric3d_model is None:
            self.metric3d_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True, trust_repo=True)
            self.metric3d_model.cuda().eval()
            
        image_size = (616, 1064)
        h, w = im_tensor.shape[-2:]
        scale = min(image_size[0] / h, image_size[1] / w)
        intrinsic_scaled = intrinsic * scale

        trans_totensor = transforms.Compose(
            [
                transforms.Resize((int(h * scale), int(w * scale))),
            ]
        )
        im_tensor = trans_totensor(im_tensor).cuda()

        pad_h, pad_w = image_size[0] - int(h * scale), image_size[1] - int(w * scale)
        pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
        im_tensor = transforms.functional.pad(
            im_tensor,
            (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half),
            padding_mode="constant",
            fill=0.0,
        )

        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        pred_depth, _, output_dict = self.metric3d_model.inference({"input": im_tensor})
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            :,
            pad_info[0] : pred_depth.shape[1] - pad_info[1],
            pad_info[2] : pred_depth.shape[2] - pad_info[3],
        ]
        pred_depth = F.interpolate(
            pred_depth[:, None, :, :], (h, w), mode="bicubic"
        ).squeeze()
        
        fx_scaled = intrinsic_scaled[:, 0]
        canonical_to_real_scale = fx_scaled / 1000.0
        pred_depth = pred_depth.cpu() * canonical_to_real_scale.view(-1, 1, 1).cpu()
        depth = torch.clamp(pred_depth, 0, 300).float().squeeze().cpu()
        
        normal = output_dict['prediction_normal'][:, :3, :, :].squeeze()
        normal = normal[
            :, 
            :, 
            pad_info[0] : normal.shape[2] - pad_info[1], 
            pad_info[2] : normal.shape[3] - pad_info[3]
        ]
        normal = F.interpolate(normal, size=(h, w), mode='bicubic').float().squeeze().cpu()
        
        return depth, normal

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, t, tstamp, image, intrinsics, measurement_depth=None):
        """ main update operation - run on every frame in video """
        # skip features of stereo right image
        indices = list(range(len(image)))
        del indices[1]

        Id = lietorch.SE3.Identity(1,).data.squeeze()

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            prior_depth, normal = self.__prior_extractor(inputs[0], intrinsics)
            net, inp = self.__context_encoder(inputs[:,indices])
            self.net, self.inp, self.fmap = net, inp, gmap
            intrinsics[:, :4] /= 8.0
            
            if self.args.rgbd:
                self.video.append(t, tstamp, image[indices][:, [2,1,0]], Id, 1.0, measurement_depth[indices], normal[indices], intrinsics, gmap, net, inp)
            elif self.args.prgbd:
                self.video.append(t, tstamp, image[indices][:, [2,1,0]], Id, 1.0, prior_depth[indices], normal[indices], intrinsics, gmap, net, inp)
            else:
                self.video.append(t, tstamp, image[indices][:, [2,1,0]], Id, 1.0, None, normal[indices], intrinsics, gmap, net, inp)

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(self.coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None,[0]], self.inp[None,[0]], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh or (tstamp - self.video.kf_stamps[self.video.counter.value-1]) > 3:
                self.count = 0
                prior_depth, normal = self.__prior_extractor(inputs[0], intrinsics)
                net, inp = self.__context_encoder(inputs[:,indices])
                self.net, self.inp, self.fmap = net, inp, gmap
                intrinsics[:, :4] /=8.0
                
                if self.args.rgbd:
                    self.video.append(t, tstamp, image[indices][:, [2,1,0]], None, None, measurement_depth[indices], normal[indices], intrinsics, gmap, net, inp)
                elif self.args.prgbd:
                    self.video.append(t, tstamp, image[indices][:, [2,1,0]], None, None, prior_depth[indices], normal[indices], intrinsics, gmap, net, inp)
                else:
                    self.video.append(t, tstamp, image[indices][:, [2,1,0]], None, None, None, normal[indices], intrinsics, gmap, net, inp)

            else:
                self.count += 1
