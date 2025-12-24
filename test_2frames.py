# new_two_frame_processor_with_histmatch.py
# -*- coding: utf-8 -*-
"""
Processes two images to create motion flow and residual,
then reconstructs the second frame from the first, flow, and residual,
using a simplified VideoCodec workflow.
Includes histogram matching of the reconstructed frame to the original.
"""

# ==============================================================================
# Imports
# ==============================================================================
import io
import os
import sys
import traceback
from pathlib import Path
import time

# Third-party Libraries
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast # For RAFT mixed precision
from torchvision import transforms
from torchvision.transforms import functional as TF_tv

# CompressAI - must be installed
try:
    import compressai
    from compressai.entropy_models import EntropyBottleneck
    print(f"Using compressai version: {compressai.__version__}")
except ImportError:
    print("ERROR: compressai library not found. Please run: pip install compressai")
    sys.exit(1)
except Exception as e:
    print(f"ERROR importing compressai: {e}")
    sys.exit(1)

# Scikit-image for histogram matching and metrics
try:
    import skimage
    from skimage.exposure import match_histograms as skimage_match_histograms
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    print(f"Using scikit-image version: {skimage.__version__}")
except ImportError:
    print("ERROR: scikit-image library not found. Please run: pip install scikit-image")
    sys.exit(1)


# RAFT - Availability Checks
_TORCHVISION_RAFT_NEW_AVAILABLE = False
_TV_RAFT_NEW_IMPL = None
_TV_RAFT_NEW_WEIGHTS_ENUM = None

try:
    from torchvision.models.optical_flow import raft_large as _tv_raft_large_new, Raft_Large_Weights as _Raft_Large_Weights_New
    _TORCHVISION_RAFT_NEW_AVAILABLE = True
    _TV_RAFT_NEW_IMPL = _tv_raft_large_new
    _TV_RAFT_NEW_WEIGHTS_ENUM = _Raft_Large_Weights_New
    print("Found RAFT from torchvision.models.optical_flow (new API with Weights)")
except ImportError:
    print("RAFT from torchvision.models.optical_flow (new API with Weights) not found.")

if not _TORCHVISION_RAFT_NEW_AVAILABLE:
    print("\nCRITICAL ERROR: torchvision RAFT implementation could not be imported.\n"
          "Please ensure torchvision is up-to-date.")
    sys.exit(1)


# ==============================================================================
# Helper Network Modules (Copied from codec.py)
# ==============================================================================
def get_activation(name="leaky_relu"):
    name_lower = name.lower() if name else "none"
    if name_lower == "none": return nn.Identity()
    elif name_lower == "relu": return nn.ReLU(inplace=True)
    elif name_lower == "leaky_relu": return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name_lower == "gelu": return nn.GELU()
    elif name_lower == "sigmoid": return nn.Sigmoid()
    elif name_lower == "tanh": return nn.Tanh()
    else: raise ValueError(f"Unknown activation function: {name}")

class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True), bias=False):
        actual_padding = padding
        if isinstance(padding, str) and padding.lower() == 'same':
            if stride == 1: actual_padding = kernel_size // 2
            else:
                if kernel_size % 2 == 0: raise ValueError("kernel_size must be odd for 'same' padding with stride=1")
                actual_padding = kernel_size // 2
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=actual_padding, bias=bias))
        if norm_layer is not None: self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None: self.add_module("act", act_layer)

class ConvTransposeNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True), bias=False):
        super().__init__()
        self.add_module("conv_transpose", nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
        if norm_layer is not None: self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None: self.add_module("act", act_layer)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size, stride=1, padding='same', norm_layer=norm_layer, act_layer=act_layer),
            ConvNormAct(channels, channels, kernel_size, stride=1, padding='same', norm_layer=norm_layer, act_layer=None)
        )
        self.final_act = act_layer if act_layer is not None else nn.Identity()
    def forward(self, x): return self.final_act(self.block(x) + x)

class Encoder(nn.Module):
    def __init__(self, input_channels, base_channels=64, latent_channels=128, num_downsample_layers=3, num_res_blocks=2):
        super().__init__()
        layers = [ConvNormAct(input_channels, base_channels, kernel_size=5, stride=1, padding=2)]
        current_channels = base_channels
        for _ in range(num_downsample_layers):
            out_ch = current_channels * 2
            layers.append(ConvNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1))
            current_channels = out_ch
        for _ in range(num_res_blocks): layers.append(ResidualBlock(current_channels))
        layers.append(nn.Conv2d(current_channels, latent_channels, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*layers)
        self.output_channels = latent_channels
    def forward(self, x): return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, output_channels, base_channels=64, latent_channels=128, num_upsample_layers=3, num_res_blocks=2, final_activation=None):
        super().__init__()
        channels_before_upsample = base_channels * (2**num_upsample_layers)
        layers = [ConvNormAct(latent_channels, channels_before_upsample, kernel_size=3, stride=1, padding=1)]
        current_channels = channels_before_upsample
        for _ in range(num_res_blocks): layers.append(ResidualBlock(current_channels))
        for _ in range(num_upsample_layers):
            out_ch = current_channels // 2
            layers.append(ConvTransposeNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_channels = out_ch
        layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=5, stride=1, padding=2))
        if final_activation: layers.append(get_activation(final_activation))
        self.decoder = nn.Sequential(*layers)
        self.input_channels = latent_channels
    def forward(self, x): return self.decoder(x)

class WarpingLayer(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, flow):
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
                                        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
        norm_flow_x = (flow[:, 0, :, :] / ((W - 1) / 2.0) if W > 1 else torch.zeros_like(flow[:, 0, :, :]))
        norm_flow_y = (flow[:, 1, :, :] / ((H - 1) / 2.0) if H > 1 else torch.zeros_like(flow[:, 1, :, :]))
        norm_flow = torch.stack((norm_flow_x, norm_flow_y), dim=3)
        sampling_grid = grid + norm_flow
        return F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

class MotionCompensationNetwork(nn.Module):
    def __init__(self, input_channels=3 + 2 + 3, output_channels=3, base_channels=32, num_res_blocks=3):
        super().__init__()
        layers = [ConvNormAct(input_channels, base_channels, kernel_size=5, padding=2)]
        for _ in range(num_res_blocks): layers.append(ResidualBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, output_channels, kernel_size=5, padding=2))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    def forward(self, warped_ref, flow, ref_frame):
        mcn_input = torch.cat([warped_ref, flow, ref_frame], dim=1)
        return warped_ref * self.network(mcn_input)

# ==============================================================================
# VideoCodec Model (Copied and simplified interface from codec.py)
# ==============================================================================
class VideoCodec(nn.Module):
    def __init__(self, motion_latent_channels=128, residual_latent_channels=192, mcn_base_channels=32,
                 encoder_base_channels=64, encoder_res_blocks=2, encoder_downsample_layers=3,
                 decoder_res_blocks=2, decoder_upsample_layers=3):
        super().__init__()
        self.motion_encoder = Encoder(2, encoder_base_channels // 2, motion_latent_channels, encoder_downsample_layers, encoder_res_blocks)
        self.motion_entropy_bottleneck = EntropyBottleneck(motion_latent_channels)
        self.motion_decoder = Decoder(2, encoder_base_channels // 2, motion_latent_channels, decoder_upsample_layers, decoder_res_blocks, None)
        self.residual_encoder = Encoder(3, encoder_base_channels, residual_latent_channels, encoder_downsample_layers, encoder_res_blocks)
        self.residual_entropy_bottleneck = EntropyBottleneck(residual_latent_channels)
        self.residual_decoder = Decoder(3, encoder_base_channels, residual_latent_channels, decoder_upsample_layers, decoder_res_blocks, None)
        self.warping_layer = WarpingLayer()
        self.motion_compensation_net = MotionCompensationNetwork(3 + 2 + 3, 3, mcn_base_channels, 3)

    @torch.no_grad()
    def init_entropy_bottleneck_buffers(self):
        print("DEBUG: Updating entropy bottleneck buffers with force=True...")
        try:
            self.motion_entropy_bottleneck.update(force=True)
            self.residual_entropy_bottleneck.update(force=True)
            print("DEBUG: Entropy bottleneck buffers updated successfully.")
        except Exception as e:
            print(f"WARNING: Could not initialize/update entropy bottleneck buffers: {e}"); traceback.print_exc()

    def _compress_latent(self, bottleneck, latent_tensor, bottleneck_name=""):
        if latent_tensor.shape[0] != 1: latent_tensor = latent_tensor[0:1]
        try:
            strings = bottleneck.compress(latent_tensor)
            shape = latent_tensor.size()[-2:]
            return (strings[0] if isinstance(strings, list) else strings), tuple(shape)
        except RuntimeError as e:
            if "Entropy bottleneck must be updated" in str(e):
                print(f"RuntimeError: {bottleneck_name} needs update. Forcing update.")
                bottleneck.update(force=True)
                strings = bottleneck.compress(latent_tensor)
                return (strings[0] if isinstance(strings, list) else strings), tuple(latent_tensor.size()[-2:])
            raise e

    @torch.no_grad()
    def compress_simplified(self, flow_input_for_comp, residual_input_for_comp):
        self.eval()
        if flow_input_for_comp.shape[0] != 1: flow_input_for_comp = flow_input_for_comp[0:1]
        if residual_input_for_comp.shape[0] != 1: residual_input_for_comp = residual_input_for_comp[0:1]
        
        compressed_data = {}
        motion_latents = self.motion_encoder(flow_input_for_comp)
        compressed_data["motion"] = self._compress_latent(self.motion_entropy_bottleneck, motion_latents, "MotionEB")
        frame_residual_latents = self.residual_encoder(residual_input_for_comp)
        compressed_data["frame_residual"] = self._compress_latent(self.residual_entropy_bottleneck, frame_residual_latents, "ResidualEB")
        return compressed_data

    @torch.no_grad()
    def decompress_frame_simplified(self, previous_frame_tensor, motion_compressed_data,
                                   residual_compressed_data, target_frame_hw):
        self.eval()
        if previous_frame_tensor.shape[0] != 1: previous_frame_tensor = previous_frame_tensor[0:1]

        motion_strings, motion_latent_shape_hw = motion_compressed_data
        residual_strings, residual_latent_shape_hw = residual_compressed_data
        motion_latent_shape_hw = tuple(map(int, motion_latent_shape_hw))
        residual_latent_shape_hw = tuple(map(int, residual_latent_shape_hw))

        quantized_motion_latent = self.motion_entropy_bottleneck.decompress([motion_strings], motion_latent_shape_hw)
        flow_reconstructed_native_res = self.motion_decoder(quantized_motion_latent)

        H_orig, W_orig = target_frame_hw
        if flow_reconstructed_native_res.shape[-2:] != (H_orig, W_orig):
             flow_reconstructed_native_res = resize_flow(flow_reconstructed_native_res, target_hw=(H_orig, W_orig))
        if flow_reconstructed_native_res is None: raise RuntimeError("Flow reconstruction/resizing failed.")

        warped_prev_frame = self.warping_layer(previous_frame_tensor, flow_reconstructed_native_res)
        frame2_motion_compensated = self.motion_compensation_net(warped_prev_frame, flow_reconstructed_native_res, previous_frame_tensor)

        quantized_residual_latents = self.residual_entropy_bottleneck.decompress([residual_strings], residual_latent_shape_hw)
        residual_reconstructed_native_res = self.residual_decoder(quantized_residual_latents)

        if residual_reconstructed_native_res.shape[-2:] != (H_orig, W_orig) and H_orig > 0 and W_orig > 0:
            residual_reconstructed_native_res = TF_tv.resize(
                residual_reconstructed_native_res, [H_orig, W_orig],
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
            )
        if H_orig > 0 and W_orig > 0 and residual_reconstructed_native_res.shape[-2:] != (H_orig, W_orig):
            raise RuntimeError(f"Reconstructed residual shape mismatch.")

        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed_native_res
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)
        return (frame2_reconstructed, flow_reconstructed_native_res, warped_prev_frame,
                frame2_motion_compensated, residual_reconstructed_native_res)

# ==============================================================================
# Helper Functions (Copied/Adapted from codec.py)
# ==============================================================================
def preprocess_frame_raft(frame_np_rgb, resize_shape_hw, device):
    tensor = TF_tv.to_tensor(frame_np_rgb)
    resized_tensor = TF_tv.resize(tensor, list(resize_shape_hw), antialias=True)
    return resized_tensor.unsqueeze(0).to(device)

def preprocess_frame_codec(frame_np_rgb, device):
    tensor = TF_tv.to_tensor(frame_np_rgb)
    return tensor.unsqueeze(0).to(device)

def resize_flow(flow_tensor, target_hw):
    if flow_tensor is None: return None
    B, C, H_in, W_in = flow_tensor.shape
    if C != 2: raise ValueError(f"Flow tensor must have 2 channels, got {C}")
    H_out, W_out = target_hw
    if (H_in, W_in) == (H_out, W_out): return flow_tensor
    if H_in == 0 or W_in == 0: return torch.zeros(B, C, H_out, W_out, device=flow_tensor.device, dtype=flow_tensor.dtype)
    if H_out == 0 or W_out == 0: return torch.zeros(B, C, H_out, W_out, device=flow_tensor.device, dtype=flow_tensor.dtype)
    try:
        flow_resized = TF_tv.resize(flow_tensor, [H_out, W_out], interpolation=transforms.InterpolationMode.BILINEAR, antialias=False)
        scale_w = float(W_out) / W_in if W_in > 0 else 1.0
        scale_h = float(H_out) / H_in if H_in > 0 else 1.0
        flow_scaled = torch.zeros_like(flow_resized)
        flow_scaled[:, 0, :, :] = flow_resized[:, 0, :, :] * scale_w
        flow_scaled[:, 1, :, :] = flow_resized[:, 1, :, :] * scale_h
        return flow_scaled
    except Exception as e: print(f"Error resizing flow: {e}"); traceback.print_exc(); return None

def load_model_checkpoint(checkpoint_path, model, model_name="Model", device=None, strict_load=False):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"ERROR: {model_name} checkpoint path invalid: '{checkpoint_path}'"); return False
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} checkpoint: {checkpoint_path} to device: {device}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)))
        if not isinstance(state_dict, dict) or not state_dict :
            if all('.' in k or k.endswith(('_weight', '_bias', '_running_mean', '_running_var', '_num_batches_tracked')) for k in checkpoint.keys()):
                 state_dict = checkpoint
            else: raise KeyError("Could not find state_dict.")

        cleaned_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        state_dict = { (k[len('_orig_mod.'):] if '_orig_mod.' in k else k) : v for k, v in cleaned_state_dict.items() }
        state_dict = { (k[len('model.'):] if k.startswith('model.') else k) : v for k, v in state_dict.items() }

        if isinstance(model, VideoCodec):
            try: model.motion_entropy_bottleneck.update(force=True); model.residual_entropy_bottleneck.update(force=True)
            except Exception as e_pre_update: print(f"    WARNING: PRE-LOAD EB update failed: {e_pre_update}")

        load_result = model.load_state_dict(state_dict, strict=strict_load)
        
        if isinstance(model, VideoCodec):
            model.init_entropy_bottleneck_buffers()
            if model.motion_entropy_bottleneck._quantized_cdf is None or model.residual_entropy_bottleneck._quantized_cdf is None:
                 print("    CRITICAL WARNING: CDFs are still None after POST-LOAD update!")

        if not load_result.missing_keys and not load_result.unexpected_keys: print(f"  {model_name} loaded successfully.")
        else:
            print(f"  {model_name} loaded with mismatches (strict={strict_load}).")
            if load_result.missing_keys: print(f"    Missing Keys: {load_result.missing_keys}")
            if load_result.unexpected_keys: print(f"    Unexpected Keys: {load_result.unexpected_keys}")
        
        model.to(device); model.eval()
        del checkpoint, state_dict, cleaned_state_dict
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return True
    except Exception as e: print(f"ERROR loading {model_name}: {e}"); traceback.print_exc(); return False

def load_image_as_tensor(image_path, device):
    try:
        img_pil = Image.open(image_path).convert('RGB')
        return TF_tv.to_tensor(img_pil).unsqueeze(0).to(device)
    except Exception as e: print(f"Error loading image {image_path}: {e}"); return None

def tensor_to_cv2_bgr(tensor_bchw_01):
    if tensor_bchw_01 is None: return []
    if tensor_bchw_01.dim() == 3: tensor_bchw_01 = tensor_bchw_01.unsqueeze(0)
    if tensor_bchw_01.dim() != 4 or tensor_bchw_01.shape[1] not in [1, 3]: raise ValueError(f"Unsupported tensor shape {tensor_bchw_01.shape}")
    if tensor_bchw_01.shape[1] == 1: tensor_bchw_01 = tensor_bchw_01.repeat(1, 3, 1, 1)
    
    images_np = []
    for i in range(tensor_bchw_01.shape[0]):
        img_hwc_rgb = torch.clamp(tensor_bchw_01[i], 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
        img_hwc_bgr_uint8 = (img_hwc_rgb * 255).astype(np.uint8)
        if img_hwc_bgr_uint8.shape[2] == 3: img_hwc_bgr_uint8 = cv2.cvtColor(img_hwc_bgr_uint8, cv2.COLOR_RGB2BGR)
        images_np.append(img_hwc_bgr_uint8)
    return images_np

def save_tensor_as_image_vis(tensor_bchw, filepath: Path, drange=(0,1)):
    if tensor_bchw is None: print(f"Warning: Attempted to save None tensor to {filepath}"); return
    try:
        min_val, max_val = drange
        tensor_normalized = (tensor_bchw - min_val) / (max_val - min_val + 1e-6)
        img_np_list = tensor_to_cv2_bgr(tensor_normalized)
        if not img_np_list: print(f"Warning: Conversion to CV2 failed for {filepath}"); return
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(filepath), img_np_list[0]): print(f"Warning: Failed to write image to {filepath}")
    except Exception as e: print(f"ERROR saving tensor to image {filepath}: {e}"); traceback.print_exc()

def visualize_flow_hsv(flow_tensor_b2hw, filepath: Path, clip_norm=None):
    if flow_tensor_b2hw is None or flow_tensor_b2hw.shape[1] != 2: print(f"Invalid flow for viz: {flow_tensor_b2hw.shape if flow_tensor_b2hw is not None else 'None'}"); return
    try:
        flow_np_hw2 = flow_tensor_b2hw[0].detach().cpu().numpy().transpose(1, 2, 0)
        mag, ang_rad = cv2.cartToPolar(flow_np_hw2[..., 0], flow_np_hw2[..., 1])
        hsv = np.zeros((flow_np_hw2.shape[0], flow_np_hw2.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang_rad * 180 / np.pi / 2
        hsv[..., 1] = 255
        if clip_norm is not None: mag = np.clip(mag, 0, clip_norm)
        if np.any(mag): cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
        else: mag = np.zeros_like(mag)
        hsv[..., 2] = mag.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), bgr)
    except Exception as e: print(f"ERROR visualizing flow {filepath}: {e}"); traceback.print_exc()

def _match_histograms_cv(source_tensor_bchw, reference_tensor_bchw, device):
    """
    Matches histogram of source_tensor to reference_tensor using skimage.exposure.match_histograms
    by matching Y, Cr, Cb channels in YCrCb space.
    Assumes B=1 for both tensors.
    """
    if source_tensor_bchw.shape[0] != 1 or reference_tensor_bchw.shape[0] != 1:
        raise ValueError("Histogram matching currently supports B=1 only for source and reference.")
    if source_tensor_bchw.shape[1] != 3 or reference_tensor_bchw.shape[1] != 3:
        raise ValueError("Histogram matching requires 3-channel (RGB) tensors.")

    source_cv_bgr_list = tensor_to_cv2_bgr(source_tensor_bchw)
    ref_cv_bgr_list = tensor_to_cv2_bgr(reference_tensor_bchw)

    if not source_cv_bgr_list or not ref_cv_bgr_list:
        raise RuntimeError("Conversion to CV2 BGR failed for histogram matching input.")

    source_cv_bgr = source_cv_bgr_list[0] # uint8, HWC, BGR
    ref_cv_bgr = ref_cv_bgr_list[0]     # uint8, HWC, BGR

    source_ycrcb = cv2.cvtColor(source_cv_bgr, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(ref_cv_bgr, cv2.COLOR_BGR2YCrCb)

    source_y, source_cr, source_cb = cv2.split(source_ycrcb)
    ref_y, ref_cr, ref_cb = cv2.split(ref_ycrcb)

    # skimage_match_histograms expects images of the same dtype, usually uint8 or float
    # and can return float64. We need to manage dtype for cv2.
    matched_y_processed = skimage_match_histograms(source_y, ref_y, channel_axis=None) # channel_axis=None for 2D
    matched_cr_processed = skimage_match_histograms(source_cr, ref_cr, channel_axis=None)
    matched_cb_processed = skimage_match_histograms(source_cb, ref_cb, channel_axis=None)

    matched_y = np.clip(matched_y_processed, 0, 255).astype(np.uint8)
    matched_cr = np.clip(matched_cr_processed, 0, 255).astype(np.uint8)
    matched_cb = np.clip(matched_cb_processed, 0, 255).astype(np.uint8)

    matched_ycrcb = cv2.merge([matched_y, matched_cr, matched_cb])
    matched_bgr = cv2.cvtColor(matched_ycrcb, cv2.COLOR_YCrCb2BGR)
    matched_rgb_np = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)
    
    matched_tensor_chw = TF_tv.to_tensor(matched_rgb_np) # HWC [0,255] uint8 -> CHW [0,1] float
    return matched_tensor_chw.unsqueeze(0).to(device)


# ==============================================================================
# Main Two-Frame Processing Logic
# ==============================================================================
def process_two_frames(config_params):
    print("\n--- Starting Two-Frame Processing ---")
    im1_path = Path(config_params["im1_path"])
    im2_path = Path(config_params["im2_path"])
    codec_checkpoint_path = Path(config_params["codec_checkpoint_path"])
    output_dir = Path(config_params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device Setup ---
    if config_params.get("gpu_id") is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{config_params['gpu_id']}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    raft_amp_enabled = config_params.get("raft_mixed_precision", True) and device.type == 'cuda'
    print(f"RAFT Mixed Precision: {'ENABLED' if raft_amp_enabled else 'DISABLED'}")

    # --- Load RAFT Model (Torchvision) ---
    raft_model = None
    if _TORCHVISION_RAFT_NEW_AVAILABLE:
        print(f"Instantiating Torchvision RAFT (new API) with weights: {_TV_RAFT_NEW_WEIGHTS_ENUM.DEFAULT}")
        raft_model = _TV_RAFT_NEW_IMPL(weights=_TV_RAFT_NEW_WEIGHTS_ENUM.DEFAULT)
    else:
        print("FATAL: No suitable torchvision RAFT model found."); sys.exit(1)
    
    raft_model.to(device).eval()
    print("RAFT model loaded and in eval mode.")

    # --- Load VideoCodec Model ---
    print("Instantiating VideoCodec model...")
    video_codec = VideoCodec(
        motion_latent_channels=config_params["motion_latent_channels"],
        residual_latent_channels=config_params["residual_latent_channels"],
        mcn_base_channels=config_params["mcn_base_channels"],
        encoder_base_channels=config_params["encoder_base_channels"],
        encoder_res_blocks=config_params["encoder_res_blocks"],
        encoder_downsample_layers=config_params["encoder_downsample_layers"],
        decoder_res_blocks=config_params["decoder_res_blocks"],
        decoder_upsample_layers=config_params["decoder_upsample_layers"],
    )
    if not load_model_checkpoint(str(codec_checkpoint_path), video_codec, "VideoCodec", device, strict_load=False):
        print(f"FATAL: Failed to load VideoCodec checkpoint: {codec_checkpoint_path}"); sys.exit(1)

    # --- Load Images ---
    im1_tensor_native_b1chw = load_image_as_tensor(im1_path, device)
    im2_tensor_native_b1chw = load_image_as_tensor(im2_path, device) # This is our ground truth for im2
    if im1_tensor_native_b1chw is None or im2_tensor_native_b1chw is None:
        print("FATAL: Failed to load input images."); sys.exit(1)

    _b, _c, native_h, native_w = im1_tensor_native_b1chw.shape
    print(f"Images loaded. Native resolution: {native_w}x{native_h}")

    # --- 1. Estimate Optical Flow (RAFT) ---
    print("Estimating optical flow with RAFT...")
    im1_np_rgb = np.array(TF_tv.to_pil_image(im1_tensor_native_b1chw[0].cpu()))
    im2_np_rgb = np.array(TF_tv.to_pil_image(im2_tensor_native_b1chw[0].cpu()))

    raft_input_h, raft_input_w = config_params["raft_resize_height"], config_params["raft_resize_width"]
    im1_raft_input = preprocess_frame_raft(im1_np_rgb, (raft_input_h, raft_input_w), device)
    im2_raft_input = preprocess_frame_raft(im2_np_rgb, (raft_input_h, raft_input_w), device)

    with torch.no_grad(), autocast(device_type=device.type, enabled=raft_amp_enabled):
        flow_predictions_raft_res = raft_model(im1_raft_input, im2_raft_input, num_flow_updates=config_params["raft_iters"])
    flow_at_raft_res = flow_predictions_raft_res[-1]
    flow_native_uncompressed = resize_flow(flow_at_raft_res, target_hw=(native_h, native_w))
    if flow_native_uncompressed is None:
        print("FATAL: Failed to resize RAFT flow to native resolution."); sys.exit(1)
    visualize_flow_hsv(flow_native_uncompressed, output_dir / "flow_initial_native.png")
    print(f"Optical flow estimated and resized to native. Shape: {flow_native_uncompressed.shape}")

    # --- 2. Calculate Residual (at native resolution) ---
    print("Calculating residual at native resolution...")
    with torch.no_grad():
        warped_im1_native = video_codec.warping_layer(im1_tensor_native_b1chw, flow_native_uncompressed)
        mc_pred_native = video_codec.motion_compensation_net(warped_im1_native, flow_native_uncompressed, im1_tensor_native_b1chw)
        residual_native_uncompressed = im2_tensor_native_b1chw - mc_pred_native
        residual_native_uncompressed = torch.clamp(residual_native_uncompressed, -1.0, 1.0)
    save_tensor_as_image_vis(residual_native_uncompressed, output_dir / "residual_initial_native.png", drange=(-0.5, 0.5))
    print(f"Residual calculated at native. Shape: {residual_native_uncompressed.shape}")

    # --- 3. Compress Flow and Residual ---
    print("Compressing flow and residual...")
    start_comp_time = time.time()
    compressed_data = video_codec.compress_simplified(flow_native_uncompressed, residual_native_uncompressed)
    motion_payload, residual_payload = compressed_data["motion"], compressed_data["frame_residual"]
    end_comp_time = time.time()
    print(f"Compression done in {end_comp_time - start_comp_time:.3f}s. "
          f"Motion bytes: {len(motion_payload[0])}, Residual bytes: {len(residual_payload[0])}")

    # --- 4. Decompress and Reconstruct im2 ---
    print("Decompressing and reconstructing im2 (before histogram matching)...")
    start_decomp_time = time.time()
    (im2_reconstructed_raw, flow_reconstructed_final, 
     _warped_debug, _mc_pred_debug, residual_reconstructed_final) = \
        video_codec.decompress_frame_simplified(
            im1_tensor_native_b1chw,
            motion_payload,
            residual_payload,
            target_frame_hw=(native_h, native_w)
        )
    end_decomp_time = time.time()
    print(f"Decompression & raw reconstruction done in {end_decomp_time - start_decomp_time:.3f}s.")

    # --- 5. Histogram Matching ---
    print("Performing histogram matching...")
    start_hist_match_time = time.time()
    try:
        im2_reconstructed_hist_matched = _match_histograms_cv(
            im2_reconstructed_raw, 
            im2_tensor_native_b1chw, # Match to original im2
            device
        )
        end_hist_match_time = time.time()
        print(f"Histogram matching done in {end_hist_match_time - start_hist_match_time:.3f}s.")
    except Exception as e_hm:
        print(f"ERROR during histogram matching: {e_hm}. Skipping histogram matching.")
        traceback.print_exc()
        im2_reconstructed_hist_matched = im2_reconstructed_raw # Fallback to raw reconstruction

    # --- 6. Save Outputs ---
    print("Saving outputs...")
    save_tensor_as_image_vis(im2_reconstructed_raw, output_dir / "im2_reconstructed_raw.png")
    save_tensor_as_image_vis(im2_reconstructed_hist_matched, output_dir / "im2_reconstructed_hist_matched.png")
    visualize_flow_hsv(flow_reconstructed_final, output_dir / "flow_reconstructed_final.png")
    save_tensor_as_image_vis(residual_reconstructed_final, output_dir / "residual_reconstructed_final.png", drange=(-0.5, 0.5))
    save_tensor_as_image_vis(im2_tensor_native_b1chw, output_dir / "im2_original_for_comparison.png")

    # --- 7. Calculate Metrics ---
    print("Calculating quality metrics...")
    im2_gt_pil = TF_tv.to_pil_image(im2_tensor_native_b1chw.squeeze(0).cpu())
    im2_gt_np = np.array(im2_gt_pil)

    # Metrics for raw reconstruction
    im2_rec_raw_pil = TF_tv.to_pil_image(im2_reconstructed_raw.squeeze(0).cpu())
    im2_rec_raw_np = np.array(im2_rec_raw_pil)
    psnr_raw = peak_signal_noise_ratio(im2_gt_np, im2_rec_raw_np, data_range=255)
    # For SSIM, ensure dimensions are suitable for win_size (min_dim >= win_size)
    min_dim_raw = min(im2_gt_np.shape[0], im2_gt_np.shape[1])
    win_size_raw = min(7, min_dim_raw - (1 if min_dim_raw % 2 == 0 else 0) ) # Ensure odd and <= min_dim-1
    if win_size_raw < 3 : ssim_raw = float('nan') # skip if too small
    else: ssim_raw = structural_similarity(im2_gt_np, im2_rec_raw_np, data_range=255, channel_axis=-1, win_size=win_size_raw)
    print(f"  Metrics for RAW reconstruction (vs original im2):")
    print(f"    PSNR: {psnr_raw:.2f} dB")
    print(f"    SSIM: {ssim_raw:.4f} (win_size={win_size_raw})")

    # Metrics for histogram-matched reconstruction
    im2_rec_hm_pil = TF_tv.to_pil_image(im2_reconstructed_hist_matched.squeeze(0).cpu())
    im2_rec_hm_np = np.array(im2_rec_hm_pil)
    psnr_hm = peak_signal_noise_ratio(im2_gt_np, im2_rec_hm_np, data_range=255)
    min_dim_hm = min(im2_gt_np.shape[0], im2_gt_np.shape[1])
    win_size_hm = min(7, min_dim_hm - (1 if min_dim_hm % 2 == 0 else 0) )
    if win_size_hm < 3 : ssim_hm = float('nan')
    else: ssim_hm = structural_similarity(im2_gt_np, im2_rec_hm_np, data_range=255, channel_axis=-1, win_size=win_size_hm)

    print(f"  Metrics for HISTOGRAM-MATCHED reconstruction (vs original im2):")
    print(f"    PSNR: {psnr_hm:.2f} dB")
    print(f"    SSIM: {ssim_hm:.4f} (win_size={win_size_hm})")

    print(f"--- Processing Finished. Outputs saved in '{output_dir.resolve()}' ---")


if __name__ == "__main__":
    config = {
        "im1_path": "./im1.png",
        "im2_path": "./im2.png",
        "codec_checkpoint_path": "./codec_checkpoints_2phase_visual/latest_checkpoint_3phase.pth.tar",
        "output_dir": "./two_frame_output_histmatch",
        "gpu_id": 0,
        "raft_resize_height": 368,
        "raft_resize_width": 640,
        "raft_iters": 12,
        "raft_mixed_precision": True,
        "motion_latent_channels": 128,
        "residual_latent_channels": 192,
        "mcn_base_channels": 32,
        "encoder_base_channels": 64,
        "encoder_res_blocks": 2,
        "encoder_downsample_layers": 3,
        "decoder_res_blocks": 2,
        "decoder_upsample_layers": 3,
    }

    if not Path(config["im1_path"]).is_file() or not Path(config["im2_path"]).is_file():
        print(f"ERROR: Input images {config['im1_path']} or {config['im2_path']} not found.")
        # Create dummy images example (optional)
        if not Path(config["im1_path"]).is_file():
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(config["im1_path"])
            print(f"Created dummy {config['im1_path']}")
        if not Path(config["im2_path"]).is_file():
            dummy_img2_arr = np.zeros((256, 256, 3), dtype=np.uint8); dummy_img2_arr[100:150, 120:170, :] = 200
            Image.fromarray(dummy_img2_arr).save(config["im2_path"])
            print(f"Created dummy {config['im2_path']}")
        # sys.exit(1)

    if not Path(config["codec_checkpoint_path"]).is_file():
        print(f"ERROR: Codec checkpoint {config['codec_checkpoint_path']} not found."); sys.exit(1)

    process_two_frames(config)