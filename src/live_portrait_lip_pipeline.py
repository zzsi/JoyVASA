# coding: utf-8

"""
Pipeline of LivePortrait with Lip Animation Generation
"""

import torch
torch.backends.cudnn.benchmark = True

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import os
import os.path as osp
import numpy as np
import pickle
 
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.video import images2video, add_audio_to_video
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, resize_to_limit
from .utils.helper import mkdir, basename, dct2device, is_image, calc_motion_multiplier
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
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        lip_delta_before_animation = None

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

        # let lip-open scalar to be 0 at first
        if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

        ######## animate ########
        # Store lip features for each frame
        lip_features = []
        
        # Define lip keypoint indices
        lip_indices = [6, 12, 14, 17, 19, 20]  # Lip keypoint indices
        
        for i in track(range(n_frames), description='🚀Animating Lip Region...', total=n_frames):
            x_d_i_info = driving_template_dct['motion'][i]
            x_d_i_info = dct2device(x_d_i_info, device)
            
            if i == 0:  # cache the first frame
                x_d_0_info = x_d_i_info.copy()

            # Only modify lip-related keypoints
            delta_new = x_s_info['exp'].clone()
            
            # Apply stronger lip movement by directly using the driving expression for lip keypoints
            # This makes the lip movements more pronounced
            for lip_idx in lip_indices:
                # Use the driving expression directly for lip keypoints instead of relative motion
                delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]

            # Keep original scale and translation
            scale_new = x_s_info['scale']
            t_new = x_s_info['t']
            t_new[..., 2].fill_(0)  # zero tz

            # Generate new keypoints
            x_d_i_new = scale_new * (x_c_s + delta_new) + t_new

            # Apply motion multiplier for relative motion
            if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly":
                if i == 0:
                    x_d_0_new = x_d_i_new
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
                x_d_i_new = x_d_diff + x_s

            # Apply lip normalization if enabled
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new += lip_delta_before_animation

            # Apply stitching if enabled
            if inf_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            # Apply a stronger driving multiplier for lip movements
            # This amplifies the lip animation
            lip_multiplier = inf_cfg.driving_multiplier * 1.5  # Increase lip movement by 50%
            
            # Apply different multipliers for lip and non-lip keypoints
            x_d_i_new_lip = x_s.clone()
            x_d_i_new_non_lip = x_s.clone()
            
            # Apply stronger multiplier to lip keypoints
            for lip_idx in lip_indices:
                x_d_i_new_lip[:, lip_idx, :] = x_s[:, lip_idx, :] + (x_d_i_new[:, lip_idx, :] - x_s[:, lip_idx, :]) * lip_multiplier
            
            # Apply normal multiplier to non-lip keypoints
            for idx in range(x_s.shape[1]):
                if idx not in lip_indices:
                    x_d_i_new_non_lip[:, idx, :] = x_s[:, idx, :] + (x_d_i_new[:, idx, :] - x_s[:, idx, :]) * inf_cfg.driving_multiplier
            
            # Combine the keypoints
            x_d_i_new = x_d_i_new_non_lip.clone()
            for lip_idx in lip_indices:
                x_d_i_new[:, lip_idx, :] = x_d_i_new_lip[:, lip_idx, :]
            
            # Generate output
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            # Store lip features for this frame
            lip_features.append({
                'keypoints': x_d_i_new.cpu().numpy(),
                'delta': delta_new.cpu().numpy(),
                'scale': scale_new.cpu().numpy(),
                'translation': t_new.cpu().numpy()
            })

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
        add_audio_to_video(temp_video, args.audio, final_video, remove_temp=True)

        # Save lip features
        features_path = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}_lip_features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump({
                'fps': inf_cfg.output_fps,
                'n_frames': n_frames,
                'features': lip_features
            }, f)

        return final_video, features_path
