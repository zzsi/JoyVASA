# coding: utf-8

"""
Pipeline of LivePortrait with Lip Animation Generation
"""

import torch
torch.backends.cudnn.benchmark = True

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import os
import os.path as osp
 
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.video import images2video, add_audio_to_video
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, resize_to_limit
from .utils.helper import mkdir, basename, dct2device, is_image
from .utils.rprint import rlog as log
from .live_portrait_wmg_wrapper import LivePortraitWrapper

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

class LivePortraitLipPipeline(object):
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)
    
    def execute(self, args: ArgumentConfig):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## load reference image ########
        if is_image(args.reference):
            img_rgb = load_image_rgb(args.reference)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            log(f"Load reference image from {args.reference}")
            source_rgb_lst = [img_rgb]
        else:
            raise Exception(f"Unknown reference image format: {args.reference}")

        ######## generate motion sequence ########
        driving_template_dct = self.live_portrait_wrapper.gen_motion_sequence(args)
        n_frames = driving_template_dct['n_frames']

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")
        I_p_lst = []
        x_d_0_info = None

        ######## process source info ########
        if inf_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
            if crop_info is None:
                raise Exception("No face detected in the source image!")
            source_lmk = crop_info['lmk_crop']
            img_crop_256x256 = crop_info['img_crop_256x256']
        else:
            source_lmk = self.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
            img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))
        
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

        ######## animate ########
        for i in track(range(n_frames), description='🚀Animating Lip Region...', total=n_frames):
            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            
            if i == 0:  # cache the first frame
                x_d_0_info = x_d_i_info.copy()

            # Only modify lip-related keypoints
            delta_new = x_s_info['exp'].clone()
            for lip_idx in [6, 12, 14, 17, 19, 20]:  # Lip keypoint indices
                delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]

            # Keep original scale and translation
            scale_new = x_s_info['scale']
            t_new = x_s_info['t']
            t_new[..., 2].fill_(0)  # zero tz

            # Generate new keypoints
            x_d_i_new = scale_new * (x_c_s + delta_new) + t_new

            # Apply stitching if enabled
            if inf_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            # Generate output
            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        # save the animated result
        mkdir(args.output_dir)
        temp_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}_lip_temp.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            images2video(I_p_pstbk_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        else:
            images2video(I_p_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        final_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}_lip.mp4')
        add_audio_to_video(temp_video, args.audio, final_video, remove_temp=False)
        return final_video
