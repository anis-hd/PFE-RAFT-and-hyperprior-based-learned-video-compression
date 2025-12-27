
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Keep for functional resize and dataset transforms
from PIL import Image
import os
import sys
import glob
import numpy as np
from pathlib import Path
import traceback # For detailed error logging
from tqdm import tqdm # For scanning progress and training loop
import math
import re
import argparse
import time
import random
import io # For potential in-memory file-like objects if needed, though bytes are sufficient
import logging # Using standard logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

# Make sure compressai is installed and importable
try:
    import compressai
    from compressai.entropy_models import EntropyBottleneck # Only need this
    from compressai.ops import quantize_ste
    compressai_available = True
    log.info("CompressAI library found and imported successfully.")
except ImportError:
    log.error("ERROR: compressai library not found or installation is incomplete.")
    log.error("Please install it: pip install compressai")
    sys.exit(1)

# Import AMP tools if available
try:
    from torch.amp import autocast, GradScaler
    amp_available = True
    log.info("torch.amp available. Automatic Mixed Precision (AMP) is enabled.")
except ImportError:
    log.warning("WARNING: torch.amp not available. Automatic Mixed Precision (AMP) will be disabled.")
    amp_available = False
    # Define dummy classes if AMP is not available to avoid errors later
    class autocast:
        def __init__(self, device_type, dtype=None, enabled=True):
             self._enabled = enabled
        def __enter__(self): pass
        def __exit__(self, *args): pass
        def is_enabled(self): return self._enabled

    class GradScaler:
        def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
             self._enabled = enabled
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def load_state_dict(self, state_dict): pass
        def state_dict(self): return {}
        def is_enabled(self): return self._enabled

# Import MS-SSIM if available
try:
    from pytorch_msssim import ms_ssim
    msssim_available = True
    log.info("MS-SSIM found and available (required for Phase 3).")
except ImportError:
    msssim_available = False
    log.warning("WARNING: pytorch_msssim not found. MS-SSIM related functionality (Phase 3) will be disabled.")


# Import plotting tools if available
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend
    import matplotlib.pyplot as plt
    plotting_available = True
    log.info("Matplotlib found. Visualization enabled.")
except ImportError:
    log.warning("Warning: Matplotlib not found. Visualization will be disabled.")
    plotting_available = False

# ==============================================================================
# MODULES (Building Blocks) - UNCHANGED
# ==============================================================================

# --- Helper Modules & Functions ---
def get_activation(name="leaky_relu"):
    """Returns the specified activation function."""
    if name is None or name.lower() == "none": return nn.Identity()
    elif name.lower() == "relu": return nn.ReLU(inplace=True)
    elif name.lower() == "leaky_relu": return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name.lower() == "gelu": return nn.GELU()
    elif name.lower() == "sigmoid": return nn.Sigmoid()
    elif name.lower() == "tanh": return nn.Tanh()
    elif name.lower() == "softplus": return nn.Softplus()
    else: raise ValueError(f"Unknown activation function: {name}")

class ConvNormAct(nn.Sequential):
    """Basic Convolution -> Normalization -> Activation block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True), bias=False):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if norm_layer is not None: self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None: self.add_module("act", act_layer)

class ConvTransposeNormAct(nn.Sequential):
    """Basic Transposed Convolution -> Normalization -> Activation block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True), bias=False):
        super().__init__()
        self.add_module("conv_transpose", nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
        if norm_layer is not None: self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None: self.add_module("act", act_layer)

class ResidualBlock(nn.Module):
    """Simple Residual Block."""
    def __init__(self, channels, kernel_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.block = nn.Sequential(ConvNormAct(channels, channels, kernel_size, stride=1, padding='same', norm_layer=norm_layer, act_layer=act_layer),
                                    ConvNormAct(channels, channels, kernel_size, stride=1, padding='same', norm_layer=norm_layer, act_layer=None))
        self.final_act = act_layer
    def forward(self, x):
        identity = x
        out = self.block(x)
        # Ensure identity addition works even if final_act is inplace
        if self.final_act:
            return self.final_act(out + identity)
        else:
            return out + identity


# --- Core Autoencoder Components ---
class Encoder(nn.Module):
    """Generic CNN Encoder with downsampling."""
    def __init__(self, input_channels, base_channels=64, latent_channels=128, num_downsample_layers=3, num_res_blocks=2):
        super().__init__()
        layers = [ConvNormAct(input_channels, base_channels, kernel_size=5, stride=1, padding='same')]
        current_channels = base_channels
        for _ in range(num_downsample_layers):
            out_ch = current_channels * 2
            layers.append(ConvNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1))
            current_channels = out_ch
        for _ in range(num_res_blocks): layers.append(ResidualBlock(current_channels))
        layers.append(nn.Conv2d(current_channels, latent_channels, kernel_size=3, stride=1, padding='same'))
        self.encoder = nn.Sequential(*layers)
    def forward(self, x): return self.encoder(x)

class Decoder(nn.Module):
    """Generic CNN Decoder with upsampling (symmetric to Encoder)."""
    def __init__(self, output_channels, base_channels=64, latent_channels=128, num_upsample_layers=3, num_res_blocks=2, final_activation=None):
        super().__init__()
        channels_before_upsample = base_channels * (2**num_upsample_layers)
        layers = [ConvNormAct(latent_channels, channels_before_upsample, kernel_size=3, stride=1, padding='same')]
        current_channels = channels_before_upsample
        for _ in range(num_res_blocks): layers.append(ResidualBlock(current_channels))
        for _ in range(num_upsample_layers):
            out_ch = current_channels // 2
            layers.append(ConvTransposeNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_channels = out_ch
        layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=5, stride=1, padding='same'))
        if final_activation: layers.append(get_activation(final_activation))
        self.decoder = nn.Sequential(*layers)
    def forward(self, x): return self.decoder(x)

# --- Motion Compensation & Warping (WarpingLayer needed, MCN used only in Phase 2 Residual Calc) ---
class WarpingLayer(nn.Module):
    """ Warps an image using optical flow using F.grid_sample. """
    def __init__(self): super().__init__()
    def forward(self, x, flow):
        B, C, H, W = x.size()
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
                                        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
                                        indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1) # B, H, W, 2

        # Normalize flow to [-1, 1] range relative to grid
        flow_permuted = flow.permute(0, 2, 3, 1) # B, H, W, 2
        # Handle potential division by zero if H or W is 1
        norm_flow_x = flow_permuted[..., 0] / ((W - 1) / 2) if W > 1 else torch.zeros_like(flow_permuted[..., 0])
        norm_flow_y = flow_permuted[..., 1] / ((H - 1) / 2) if H > 1 else torch.zeros_like(flow_permuted[..., 1])
        norm_flow = torch.stack((norm_flow_x, norm_flow_y), dim=3) # B, H, W, 2

        # Add normalized flow to grid
        sampling_grid = grid + norm_flow # B, H, W, 2

        # Sample pixels from input image `x` using the sampling grid
        warped_x = F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_x

class MotionCompensationNetwork(nn.Module):
    """ Refines the warped reference frame using a CNN. """
    def __init__(self, input_channels=3 + 2 + 3, output_channels=3, base_channels=32, num_res_blocks=3):
        super().__init__()
        layers = [ConvNormAct(input_channels, base_channels, kernel_size=5, padding='same')]
        for _ in range(num_res_blocks): layers.append(ResidualBlock(base_channels))
        # Output mask/refinement map, applied multiplicatively
        layers.extend([nn.Conv2d(base_channels, output_channels, kernel_size=5, padding='same'), nn.Sigmoid()]) # Sigmoid ensures mask is [0, 1]
        self.network = nn.Sequential(*layers)
    def forward(self, warped_ref, flow, ref_frame):
        if flow.dim() != 4 or flow.shape[1] != 2: raise ValueError(f"Expected flow shape (B, 2, H, W), got {flow.shape}")
        mcn_input = torch.cat([warped_ref, flow, ref_frame], dim=1)
        refinement_map = self.network(mcn_input)
        # Apply refinement map multiplicatively - interpretation might vary, additive is also common
        refined_frame = warped_ref * refinement_map # Or: warped_ref + refinement_map (if refinement learns a delta)
        return refined_frame


# --- VideoCodec Model ---
class VideoCodec(nn.Module):
    def __init__(self, motion_latent_channels=128, residual_latent_channels=192,
                 mcn_base_channels=32, encoder_base_channels=64,
                 encoder_res_blocks=2, encoder_downsample_layers=3,
                 decoder_res_blocks=2, decoder_upsample_layers=3,
                 likelihood_bound=1e-9):
        super().__init__()
        self.likelihood_bound = float(likelihood_bound)

        self.motion_encoder = Encoder(input_channels=2, base_channels=encoder_base_channels // 2, latent_channels=motion_latent_channels, num_downsample_layers=encoder_downsample_layers, num_res_blocks=encoder_res_blocks)
        self.motion_entropy_bottleneck = EntropyBottleneck(motion_latent_channels)
        self.motion_decoder = Decoder(output_channels=2, base_channels=encoder_base_channels // 2, latent_channels=motion_latent_channels, num_upsample_layers=decoder_upsample_layers, num_res_blocks=decoder_res_blocks, final_activation=None)

        self.warping_layer = WarpingLayer()
        self.motion_compensation_net = MotionCompensationNetwork(input_channels=3 + 2 + 3, output_channels=3, base_channels=mcn_base_channels)

        self.residual_encoder = Encoder(input_channels=3, base_channels=encoder_base_channels, latent_channels=residual_latent_channels, num_downsample_layers=encoder_downsample_layers, num_res_blocks=encoder_res_blocks)
        self.residual_entropy_bottleneck = EntropyBottleneck(residual_latent_channels)
        self.residual_decoder = Decoder(output_channels=3, base_channels=encoder_base_channels, latent_channels=residual_latent_channels, num_upsample_layers=decoder_upsample_layers, num_res_blocks=decoder_res_blocks, final_activation=None)

    def forward(self, frame1, frame2, flow_input, current_phase=2):
        """
        Forward pass for training. Behavior depends on the phase.
        `flow_input` is typically the ground truth optical flow.
        """
        # --- Motion Path ---
        # Motion AE always tries to reconstruct the `flow_input` (GT flow in P1/P2, or just runs frozen in P3)
        motion_latents = self.motion_encoder(flow_input)
        # In Phase 3, motion_entropy_bottleneck is in eval mode, so quantization is direct, likelihoods are from fixed CDFs.
        quantized_motion_latents, motion_likelihoods = self.motion_entropy_bottleneck(motion_latents)
        motion_likelihoods = torch.clamp(motion_likelihoods, min=self.likelihood_bound)
        rate_motion = -torch.log2(motion_likelihoods)
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        # --- Motion Compensation & Residual Computation ---
        frame2_motion_compensated = None
        if current_phase == 1:
            # Phase 1: Use GT flow (`flow_input`) for warping.
            # MCN is typically frozen (or not used for residual target if config.freeze_mcn_phase1).
            warped_frame1_p1 = self.warping_layer(frame1, flow_input)
            residual_computed = frame2 - warped_frame1_p1
            frame2_motion_compensated = warped_frame1_p1 # Predictor for frame2
        else: # Phase 2 or 3
            # Phase 2 & 3: Use reconstructed flow for warping, then MCN refines it.
            # In Phase 3, flow_reconstructed and MCN are outputs of frozen modules.
            warped_frame1_p23 = self.warping_layer(frame1, flow_reconstructed)
            frame2_motion_compensated = self.motion_compensation_net(warped_frame1_p23, flow_reconstructed, frame1)
            residual_computed = frame2 - frame2_motion_compensated

        # --- Residual Path ---
        # In Phase 3, this is the part being trained.
        residual_latents = self.residual_encoder(residual_computed)
        quantized_residual_latents, residual_likelihoods = self.residual_entropy_bottleneck(residual_latents)
        residual_likelihoods = torch.clamp(residual_likelihoods, min=self.likelihood_bound)
        rate_residual = -torch.log2(residual_likelihoods)
        residual_reconstructed = self.residual_decoder(quantized_residual_latents)

        # --- Final Frame Reconstruction ---
        # In Phase 3, frame2_motion_compensated is from frozen path, residual_reconstructed is from trained path.
        frame2_reconstructed_final = torch.clamp(frame2_motion_compensated + residual_reconstructed, 0.0, 1.0)

        return {
            'flow_reconstructed': flow_reconstructed,
            'residual_reconstructed': residual_reconstructed,
            'rate_motion': rate_motion,
            'rate_residual': rate_residual,
            'flow_input': flow_input,
            'residual_computed': residual_computed,
            'motion_latents': motion_latents,
            'residual_latents': residual_latents,
            'frame2_motion_compensated': frame2_motion_compensated,
            'frame2_reconstructed_final': frame2_reconstructed_final,
        }

    @torch.no_grad()
    def compress_frame(self, frame1, frame2, flow12):
        """Compresses the motion and residual information between two frames."""
        self.eval() # Ensure all submodules are in eval mode
        motion_latents = self.motion_encoder(flow12)
        motion_strings, motion_shape = self.motion_entropy_bottleneck.compress(motion_latents)

        quantized_motion_latents = self.motion_entropy_bottleneck.decompress(motion_strings, motion_shape)
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)
        residual = frame2 - frame2_motion_compensated
        residual_latents = self.residual_encoder(residual)
        residual_strings, residual_shape = self.residual_entropy_bottleneck.compress(residual_latents)

        return {"motion": (motion_strings, motion_shape), "residual": (residual_strings, residual_shape)}

    @torch.no_grad()
    def decompress_frame(self, frame1, compressed_frame_data):
        """Decompresses a frame given the reference frame and compressed data."""
        self.eval() # Ensure all submodules are in eval mode
        motion_strings, motion_shape = compressed_frame_data["motion"]
        quantized_motion_latents = self.motion_entropy_bottleneck.decompress(motion_strings, motion_shape)
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)

        residual_strings, residual_shape = compressed_frame_data["residual"]
        quantized_residual_latents = self.residual_entropy_bottleneck.decompress(residual_strings, residual_shape)
        residual_reconstructed = self.residual_decoder(quantized_residual_latents)

        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)
        return frame2_reconstructed

# ==============================================================================
# UTILS (Helper Functions)
# ==============================================================================
def compute_psnr(a, b, max_val=1.0):
    """Computes PSNR between two images or batches of images."""
    a = a.float(); b = b.float()
    if a.ndim == 4 and a.shape[0] == 1: a = a.squeeze(0)
    if b.ndim == 4 and b.shape[0] == 1: b = b.squeeze(0)
    if a.ndim == 3 and b.ndim == 3 and a.shape[0] == 1 and b.shape[0] == 1: # Should not happen with BxCxHxW
         a = a.squeeze(0); b = b.squeeze(0)
    a = torch.clamp(a, 0.0, max_val); b = torch.clamp(b, 0.0, max_val)
    mse = torch.mean((a - b) ** 2)
    if mse == 0: return float('inf')
    try:
        psnr_val = 20 * math.log10(max_val / math.sqrt(mse))
    except ValueError: # math domain error if mse is extremely small but not zero
        psnr_val = float('inf') if mse == 0 else 0.0 # Should be caught by mse==0
    return psnr_val

def save_checkpoint(state, checkpoint_dir, latest_filename="latest_checkpoint_3phase.pth.tar"):
    """Saves the training state directly (simplified, less atomic)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # If latest_filename is an absolute path, os.path.join might just return it.
    # This is intended if config.latest_checkpoint_file is absolute.
    if os.path.isabs(latest_filename):
        latest_filepath = latest_filename
    else:
        latest_filepath = os.path.join(checkpoint_dir, latest_filename)

    log.info(f"Attempting to save checkpoint to: {latest_filepath}")
    try:
        torch.save(state, latest_filepath)
        log.info(f"Checkpoint successfully saved to {latest_filepath}")
    except Exception as e:
        log.error(f"ERROR saving checkpoint to {latest_filepath}: {e}")
        log.error(f"Make sure the directory exists and is writable. Check disk space if on Kaggle/limited environment.")


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device=None):
    """Loads checkpoint state into model, optimizer, and scaler. (Simplified version)"""
    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint file not found at: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading checkpoint: {checkpoint_path} to device: {device}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
    except Exception as e:
        log.error(f"ERROR loading checkpoint file {checkpoint_path}: {e}")
        raise

    if 'state_dict' not in checkpoint: raise KeyError("'state_dict' missing from checkpoint.")
    state_dict = checkpoint['state_dict']

    # Handle potential 'module.' prefix from DataParallel saving
    if all(k.startswith('module.') for k in state_dict.keys()):
        log.info("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    # Handle potential '_orig_mod.' prefix from torch.compile
    if any('._orig_mod.' in k for k in state_dict.keys()):
         log.info("Removing '._orig_mod.' prefix from state_dict keys (likely from torch.compile).")
         try: # Try to clear hooks that might cause issues with torch.compile state dicts
            from torch.nn.modules.module import _global_parameter_registration_hooks
            _global_parameter_registration_hooks.clear()
         except (ImportError, AttributeError): pass
         state_dict = {re.sub(r'\._orig_mod', '', k) : v for k, v in state_dict.items()}

    # --- Entropy Bottleneck Update (Critical Step BEFORE loading state_dict) ---
    log.info("Updating entropy bottlenecks before loading state dict (calling .update(force=True))...")
    try:
        # Access EBs directly on the base model, even if wrapped by DataParallel
        temp_model_for_eb_update = model
        if isinstance(model, nn.DataParallel): temp_model_for_eb_update = model.module

        if hasattr(temp_model_for_eb_update, 'motion_entropy_bottleneck'):
            temp_model_for_eb_update.motion_entropy_bottleneck.update(force=True)
        if hasattr(temp_model_for_eb_update, 'residual_entropy_bottleneck'):
            temp_model_for_eb_update.residual_entropy_bottleneck.update(force=True)
        log.info("Entropy bottlenecks .update(force=True) called.")
    except Exception as e:
        log.warning(f"Warning during pre-load entropy bottleneck .update(force=True): {e}")

    load_result = model.load_state_dict(state_dict, strict=False)
    missing_non_buffer = [k for k in load_result.missing_keys if not k.endswith(('_quantized_cdf', '_offset', '_cdf_length'))]
    if missing_non_buffer:
        log.warning(f"  > Checkpoint Loading - Missing Keys (excluding EB buffers): {missing_non_buffer}")
    if load_result.unexpected_keys:
        log.warning(f"  > Checkpoint Loading - Unexpected Keys: {load_result.unexpected_keys}")

    if optimizer and 'optimizer' in checkpoint:
        log.info("Loading optimizer state...")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Move optimizer state to target device
            for state in optimizer.state.values():
                for k_opt, v_opt in state.items():
                    if isinstance(v_opt, torch.Tensor): state[k_opt] = v_opt.to(device)
            log.info("Optimizer state loaded successfully.")
        except RuntimeError as e_opt: # Common if model structure changed affecting params in optimizer
            log.warning(f"RUNTIME ERROR loading optimizer state: {e_opt}. Optimizer state might be incompatible.")
            log.warning("Optimizer may need to be re-initialized if training is unstable.")
        except Exception as e_opt_other:
            log.warning(f"Warning: Could not load optimizer state properly due to other error: {e_opt_other}")
    elif optimizer:
        log.warning("Optimizer state not found in checkpoint.")

    if scaler and scaler.is_enabled() and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        log.info("Loading GradScaler state...")
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            log.info("GradScaler state loaded successfully.")
        except Exception as e_scaler:
            log.warning(f"Warning: Could not load GradScaler state: {e_scaler}")
    elif scaler and scaler.is_enabled():
        log.warning("GradScaler state not found in checkpoint.")

    start_epoch = checkpoint.get('epoch', -1) + 1 # checkpoint stores 0-indexed epoch, resume with next
    optimizer_mode = checkpoint.get('optimizer_mode', 'full') # Default if not found
    log.info(f"Checkpoint loaded. Resuming from Epoch {start_epoch}. Optimizer mode was '{optimizer_mode}'.")

    del checkpoint # Free memory
    torch.cuda.empty_cache() # Clear any cached memory
    return start_epoch, optimizer_mode

# ==============================================================================
# DATASET
# ==============================================================================
def read_flo_file(filename):
    """Reads a .flo file (optical flow format). Returns None on error."""
    try:
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), np.float32, count=1)
            if not np.isclose(magic[0], 202021.25): return None # Invalid magic number
            width = np.frombuffer(f.read(4), np.int32, count=1)[0]
            height = np.frombuffer(f.read(4), np.int32, count=1)[0]
            if width <= 0 or height <= 0 or width > 10000 or height > 10000: return None # Unreasonable dimensions
            data_size = height * width * 2 # 2 channels (u, v)
            data = np.frombuffer(f.read(data_size * 4), np.float32, count=data_size) # 4 bytes per float
            if len(data) != data_size: return None # Incomplete data
            if np.isnan(data).any() or np.isinf(data).any(): return None # Corrupted data
            return data.reshape((height, width, 2))
    except FileNotFoundError:
        # log.debug(f"Flow file not found: {filename}") # Can be very verbose
        return None
    except Exception as e:
        log.warning(f"Error reading flow file {filename}: {e}")
        return None

class VideoFrameFlowDatasetNested(Dataset):
    def __init__(self, frame_base_dir, flow_base_dir, frame_prefix="im", frame_suffix=".png", transform=None):
        self.frame_base_path = Path(frame_base_dir).resolve()
        self.flow_base_path = Path(flow_base_dir).resolve()
        self.frame_prefix = frame_prefix
        self.frame_suffix = frame_suffix
        self.transform = transform if transform else transforms.ToTensor()
        self.pairs = []

        log.info(f"Scanning for frames in: {self.frame_base_path} with suffix {self.frame_suffix}")
        all_frames = list(self.frame_base_path.rglob(f"*{self.frame_suffix}"))
        log.info(f"Found {len(all_frames)} potential frame files.")

        frames_by_dir = {}
        log.info("Grouping frames by directory...")
        # Use disable=True for tqdm if running non-interactively or if too verbose
        for f_path in tqdm(all_frames, desc="Grouping Frames", leave=False, disable=True): # Can set disable=False for progress
            if not f_path.name.startswith(self.frame_prefix): continue # Filter by prefix
            dir_path = f_path.parent
            if dir_path not in frames_by_dir: frames_by_dir[dir_path] = []
            frames_by_dir[dir_path].append(f_path)

        count_good = 0
        count_total_pairs_checked = 0
        log.info(f"Checking {len(frames_by_dir)} directories for valid frame pairs and flow files...")
        for dir_path, frame_list in tqdm(frames_by_dir.items(), desc="Scanning Dirs", leave=False, disable=True): # Can set disable=False
            try: # Attempt numeric sort based on numbers in filename after prefix
                sorted_frames = sorted(frame_list, key=lambda p: int(re.findall(r'\d+', p.stem[len(self.frame_prefix):])[0]))
            except (IndexError, ValueError): # Fallback to alphabetical if numeric sort fails
                # log.debug(f"Numeric sort failed for dir {dir_path}, using alphabetical.")
                sorted_frames = sorted(frame_list)

            for i in range(len(sorted_frames) - 1):
                count_total_pairs_checked += 1
                f1_path, f2_path = sorted_frames[i], sorted_frames[i+1]
                # Ensure frames are consecutive (e.g., im1.png, im2.png)
                try:
                    n1 = int(re.findall(r'\d+', f1_path.stem[len(self.frame_prefix):])[0])
                    n2 = int(re.findall(r'\d+', f2_path.stem[len(self.frame_prefix):])[0])
                    if n2 != n1 + 1: continue # Skip if not consecutive
                except (IndexError, ValueError): continue # Skip if numbers can't be extracted

                # Construct expected flow file path relative to flow_base_path
                # Example: if f1_path = .../frames/seq1/im1.png
                # relative_frame_path = seq1/im1.png
                # flow_path = .../flow/seq1/im1.flo (using f1's name for the flow from f1 to f2)
                relative_frame_path = f1_path.relative_to(self.frame_base_path)
                flow_path = self.flow_base_path / relative_frame_path.with_suffix(".flo")

                if flow_path.is_file():
                    self.pairs.append((str(f1_path), str(f2_path), str(flow_path)))
                    count_good += 1
                # else:
                    # log.debug(f"Flow file not found for pair ({f1_path.name}, {f2_path.name}): {flow_path}")

        log.info(f"Dataset Scan Complete: Found {count_good} valid frame/flow pairs out of {count_total_pairs_checked} potential pairs checked.")
        if not self.pairs:
            log.error("ERROR: No valid frame/flow pairs found! Check frame/flow directories, prefixes, and file integrity.")
            log.error(f"Frame base: {self.frame_base_path}, Flow base: {self.flow_base_path}")
            sys.exit(1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1_p, f2_p, fl_p = self.pairs[idx]
        try:
            f1 = Image.open(f1_p).convert('RGB')
            f2 = Image.open(f2_p).convert('RGB')
            flow_np = read_flo_file(fl_p)
            if flow_np is None: raise RuntimeError(f"Failed to read or validate flow file: {fl_p}")

            f1_t = self.transform(f1)
            f2_t = self.transform(f2)
            flow_t = torch.from_numpy(flow_np.astype(np.float32)).permute(2, 0, 1) # HWC to CHW

            # Ensure flow map has same H, W as frames (after frame transform)
            _, H_f, W_f = f1_t.shape
            C_fl, H_fl, W_fl = flow_t.shape

            if H_f != H_fl or W_f != W_fl:
                # Resize flow and scale values
                # Using antialias=False for flow is generally preferred, or True if using newer torchvision.
                # For older torchvision, antialias=True might not be available for BILINEAR.
                # Let's assume False is safer for general compatibility for flow.
                flow_r = transforms.functional.resize(flow_t, [H_f, W_f], interpolation=transforms.InterpolationMode.BILINEAR, antialias=False)
                scale_W = float(W_f) / W_fl if W_fl > 0 else 1.0
                scale_H = float(H_f) / H_fl if H_fl > 0 else 1.0
                flow_final = torch.zeros_like(flow_r)
                flow_final[0, :, :] = flow_r[0, :, :] * scale_W # u component
                flow_final[1, :, :] = flow_r[1, :, :] * scale_H # v component
            else:
                flow_final = flow_t

            return f1_t, f2_t, flow_final
        except FileNotFoundError as e:
            log.error(f"ERROR (Dataset Get Item {idx}): File not found - {e}. Check dataset integrity.")
            # Re-raise as a critical error that DataLoader should handle (or stop execution)
            raise RuntimeError(f"File not found for index {idx}: {e}") from e
        except Exception as e:
            log.error(f"ERROR (Dataset Get Item {idx}): Failed to load item {idx} ({f1_p}, {f2_p}, {fl_p}): {e}\n{traceback.format_exc()}")
            # Re-raise to be caught by DataLoader or stop execution
            raise RuntimeError(f"Failed to process data for index {idx}") from e


# ==============================================================================
# TRAINING SCRIPT CONFIGURATION
# ==============================================================================
class TrainConfig:
    # --- Paths ---
    frame_base_dir: str = "./sequence"
    flow_base_dir: str = "./generated_flow"
    checkpoint_dir: str = "./codec_checkpoints_2phase_visual" # Changed dir name slightly
    vis_dir: str = "./codec_visualizations_3phase_resAE_vis"   # Changed dir name slightly
    log_file: str = "training_log_3phase_resAE_vis.txt"
    # This can be an absolute path to a specific checkpoint to load, or just a filename for the latest.
    # If it's just a filename, it will be joined with checkpoint_dir.
    # If it's an absolute path, that exact file will be loaded (if exists) and also used for saving.
    latest_checkpoint_file: str = "latest_checkpoint_3phase.pth.tar" # Example: relative filename
    # latest_checkpoint_file: str = r"C:\Users\Anis\Desktop\XX\codec_checkpoints_2phase_visual\latest_checkpoint_3phase.pth.tar" # User's example absolute path

    # --- Model Architecture ---
    motion_latent_channels: int = 128
    residual_latent_channels: int = 192
    mcn_base_channels: int = 32
    encoder_base_channels: int = 64
    encoder_res_blocks: int = 2
    encoder_downsample_layers: int = 3
    decoder_res_blocks: int = 2
    decoder_upsample_layers: int = 3

    # --- Training Hyperparameters ---
    epochs: int = 1000
    batch_size: int = 8
    learning_rate: float = 1e-4       # For Phase 1 & 2
    weight_decay: float = 1e-5
    clip_max_norm: float = 1.0        # Set to 0 to disable gradient clipping
    seed: int = 42
    num_workers: int = 2              # Adjust based on system capabilities
    use_amp: bool = True              # Enable Automatic Mixed Precision if available
    bitstream_calc_freq: int = 200    # Calculate real bitstream size every N batches (0 to disable)

    # --- Phase Settings ---
    phase1_epochs: int = 30           # Epochs for Phase 1 (1-indexed)
    phase3_start_epoch: int = 97    # Epoch when Phase 3 begins (1-indexed)
    # Lambdas for Phase 1 & 2 (and reused in Phase 3 if not overridden for monitoring)
    lambda_mse_flow: float = 10.0
    lambda_mse_residual: float = 15000.0 # High weight for residual reconstruction quality
    lambda_bpp_motion: float = 1.0
    lambda_bpp_residual: float = 5.0
    # Phase 3 specific (Residual AE training)
    lambda_msssim_phase3: float = 5.0      # Weight for MS-SSIM (1-MS_SSIM) in Phase 3 loss
    # lambda_mse_residual is reused for Phase 3 residual reconstruction.
    # lambda_bpp_residual is reused for Phase 3 residual BPP.
    learning_rate_phase3: float = 1e-5     # Potentially smaller LR for fine-tuning ResAE in Phase 3
    freeze_mcn_phase1: bool = True         # Whether to freeze MCN during Phase 1

    # --- System ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# TRAINING SCRIPT MAIN FUNCTION
# ==============================================================================
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Potentially add for full determinism, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    log.info(f"Random seed set to {seed}")

def visualize_epoch_end(epoch, save_dir, data_dict, avg_metrics, current_phase_numeric):
    """Generates and saves a visualization grid at the end of an epoch."""
    if not plotting_available or not data_dict:
        if not plotting_available: log.debug("Plotting disabled, skipping visualization.")
        if not data_dict: log.warning("No data available for visualization.")
        return
    os.makedirs(save_dir, exist_ok=True)
    try:
        idx_vis = 0 # Visualize the first sample in the last batch
        # Check for essential keys
        required_keys = ['frame1', 'frame2_orig', 'frame2_reconstructed_final',
                         'frame2_motion_compensated', 'flow_input', 'flow_reconstructed',
                         'residual_computed', 'residual_reconstructed']
        if not all(k in data_dict for k in required_keys):
            log.warning(f"Missing keys in data_dict for visualization (needs {required_keys}, got {list(data_dict.keys())}), skipping.")
            return

        def tensor_to_np(tensor): # Helper to convert tensor to displayable numpy image
            if tensor is None: return None
            # Ensure it's on CPU, permute C,H,W to H,W,C, convert to float numpy
            return tensor[idx_vis].detach().cpu().permute(1, 2, 0).float().numpy()

        frame1_np = tensor_to_np(data_dict['frame1'])
        frame2_orig_np = tensor_to_np(data_dict['frame2_orig'])
        frame2_recon_np = tensor_to_np(data_dict['frame2_reconstructed_final'])
        mc_image_np = tensor_to_np(data_dict['frame2_motion_compensated'])
        res_comp_np = tensor_to_np(data_dict['residual_computed'])
        res_rec_np = tensor_to_np(data_dict['residual_reconstructed'])

        def flow_to_mag_norm(flow_tensor): # Helper to convert flow to normalized magnitude map
            if flow_tensor is None: return None
            flow_np = flow_tensor[idx_vis].detach().cpu().float().numpy() # CHW
            mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            mag_norm = mag / (mag.max() + 1e-6) # Normalize to [0,1]
            return mag_norm

        flow_input_mag_norm = flow_to_mag_norm(data_dict['flow_input'])
        flow_rec_mag_norm = flow_to_mag_norm(data_dict['flow_reconstructed'])

        # Normalize residuals for visualization (typically centered around 0)
        res_comp_vis = np.clip(res_comp_np * 0.5 + 0.5, 0, 1) if res_comp_np is not None else None
        res_rec_vis = np.clip(res_rec_np * 0.5 + 0.5, 0, 1) if res_rec_np is not None else None

        # Per-item metrics for titles (using the visualized sample)
        psnr_frame_recon = compute_psnr(data_dict['frame2_reconstructed_final'][idx_vis], data_dict['frame2_orig'][idx_vis])
        item_mse_flow = F.mse_loss(data_dict['flow_reconstructed'][idx_vis], data_dict['flow_input'][idx_vis]).item() if flow_rec_mag_norm is not None else float('nan')
        item_mse_res = F.mse_loss(data_dict['residual_reconstructed'][idx_vis], data_dict['residual_computed'][idx_vis]).item() if res_rec_vis is not None else float('nan')

        fig, axs = plt.subplots(4, 2, figsize=(12, 18)) # Adjust figsize as needed
        fig.set_facecolor('white') # Set background to white directly on the fig object

        phase_str = f"Phase {current_phase_numeric}"
        avg_loss = avg_metrics.get('loss', float('nan'))
        title_metrics = []

        avg_mse_flow = avg_metrics.get('mse_flow', float('nan'))
        avg_mse_res = avg_metrics.get('mse_res', float('nan'))
        avg_bpp_m = avg_metrics.get('bpp_mot', float('nan'))
        avg_bpp_r = avg_metrics.get('bpp_res', float('nan'))

        if current_phase_numeric == 3:
            avg_msssim = avg_metrics.get('ms_ssim_val', float('nan'))
            title_metrics.extend([
                f"Avg L:{avg_loss:.4f} MS-SSIM:{avg_msssim:.4f}",
                f"MSE_F(mon):{avg_mse_flow:.6f} MSE_R(opt):{avg_mse_res:.6f}", # Clarify monitored vs optimized
                f"BPP_M(mon):{avg_bpp_m:.4f} BPP_R(opt):{avg_bpp_r:.4f}"
            ])
        else: # Phase 1 & 2
            title_metrics.extend([
                f"Avg L:{avg_loss:.4f} MSE_F(opt):{avg_mse_flow:.6f} MSE_R(opt):{avg_mse_res:.6f}",
                f"Avg BPP_M(opt):{avg_bpp_m:.4f} BPP_R(opt):{avg_bpp_r:.4f}"
            ])

        if avg_metrics.get('batches_with_kb_calc', 0) > 0:
             kb_m = avg_metrics.get('bitstream_kb_mot', float('nan'))
             kb_r = avg_metrics.get('bitstream_kb_res', float('nan'))
             title_metrics.append(f"Avg KB M(R):{kb_m:.2f} R(R):{kb_r:.2f}")

        fig.suptitle(f"Epoch {epoch} [{phase_str}] | {' | '.join(title_metrics)}", fontsize=11)

        axs[0, 0].imshow(np.clip(frame2_orig_np, 0, 1)); axs[0, 0].set_title("Original Frame 2"); axs[0, 0].axis("off")
        axs[0, 1].imshow(np.clip(frame2_recon_np, 0, 1)); axs[0, 1].set_title(f"Recon Frame 2 (PSNR: {psnr_frame_recon:.2f}dB)"); axs[0, 1].axis("off")

        mc_type_str = "Warped (GT Flow)"
        if current_phase_numeric == 2: mc_type_str = "MCN Output (Recon Flow)"
        elif current_phase_numeric == 3: mc_type_str = "MCN Output (Recon Flow, Frozen)"

        axs[1, 0].imshow(np.clip(mc_image_np, 0, 1)); axs[1, 0].set_title(f"Motion Comp. ({mc_type_str})"); axs[1, 0].axis("off")
        axs[1, 1].imshow(np.clip(frame1_np, 0, 1)); axs[1, 1].set_title("Reference Frame 1"); axs[1, 1].axis("off")

        if flow_input_mag_norm is not None: axs[2, 0].imshow(flow_input_mag_norm, cmap='viridis'); axs[2, 0].set_title("Input Flow Mag (Norm)"); axs[2, 0].axis("off")
        else: axs[2,0].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[2, 0].set_title("Input Flow Mag (Norm)"); axs[2, 0].axis("off")

        flow_recon_title = f"Recon Flow Mag (MSE: {item_mse_flow:.6f})"
        if current_phase_numeric == 3: flow_recon_title += " [Frozen]"
        if flow_rec_mag_norm is not None: axs[2, 1].imshow(flow_rec_mag_norm, cmap='viridis'); axs[2, 1].set_title(flow_recon_title); axs[2, 1].axis("off")
        else: axs[2,1].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[2, 1].set_title(flow_recon_title); axs[2, 1].axis("off")

        res_target_desc = "Target: F2 - Warp(F1, GTFlow)"
        if current_phase_numeric == 2: res_target_desc = "Target: F2 - MCN(Warp(F1, RecFlow))"
        elif current_phase_numeric == 3: res_target_desc = "Target: F2 - MCN(Warp(F1, RecFlow)) [MCN Frozen]"


        if res_comp_vis is not None: axs[3, 0].imshow(res_comp_vis); axs[3, 0].set_title(res_target_desc); axs[3, 0].axis("off")
        else: axs[3,0].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[3, 0].set_title(res_target_desc); axs[3, 0].axis("off")
        if res_rec_vis is not None: axs[3, 1].imshow(res_rec_vis); axs[3, 1].set_title(f"Recon Residual (MSE: {item_mse_res:.6f})"); axs[3, 1].axis("off")
        else: axs[3,1].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[3, 1].set_title(f"Recon Residual (MSE: {item_mse_res:.6f})"); axs[3, 1].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_phase{current_phase_numeric}_vis.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig) # Close the figure to free memory
        log.debug(f"Visualization saved to {save_path}")
    except Exception as e:
        log.error(f"ERROR during visualization for Epoch {epoch}: {e}\n{traceback.format_exc()}")
        if 'plt' in locals() and 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig) # Attempt to close figure on error

def main(config: TrainConfig):
    set_seed(config.seed)
    amp_enabled = config.use_amp and config.device == "cuda" and amp_available
    device = torch.device(config.device)
    log.info(f"Using device: {device}, AMP Enabled: {amp_enabled}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.vis_dir, exist_ok=True)
    log_path = os.path.join(config.checkpoint_dir, config.log_file) # Log file in checkpoint_dir
    file_handler = logging.FileHandler(log_path, mode='a') # Append mode
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    log.addHandler(file_handler)

    log.info("="*60 + "\nStarting Training Session (3-Phase - ResAE only in P3)\n" + "="*60)
    config_vars = {k: getattr(config, k) for k, v in vars(TrainConfig).items() if not k.startswith('__') and not callable(v)}
    log.info("Configuration:")
    for k, v in config_vars.items(): log.info(f"  {k}: {v}")
    log.info("-" * 60)

    log.info("Setting up dataset...")
    img_transform = transforms.Compose([transforms.ToTensor()]) # Basic transform
    try:
        train_dataset = VideoFrameFlowDatasetNested(config.frame_base_dir, config.flow_base_dir, transform=img_transform)
        if len(train_dataset) == 0:
             log.error("Dataset is empty after initialization. Exiting.")
             return
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=(device.type == 'cuda'), drop_last=True)
        log.info(f"Dataset loaded: {len(train_dataset)} pairs. DataLoader: {len(train_loader)} batches.")
    except Exception as e:
        log.error(f"FATAL: Failed to create Dataset or DataLoader: {e}\n{traceback.format_exc()}")
        return

    log.info("Initializing Video Codec model...")
    model = VideoCodec(
        motion_latent_channels=config.motion_latent_channels, residual_latent_channels=config.residual_latent_channels,
        mcn_base_channels=config.mcn_base_channels, encoder_base_channels=config.encoder_base_channels,
        encoder_res_blocks=config.encoder_res_blocks, encoder_downsample_layers=config.encoder_downsample_layers,
        decoder_res_blocks=config.decoder_res_blocks, decoder_upsample_layers=config.decoder_upsample_layers
    ).to(device)

    model_base = model # Handle for DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        log.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)
        model_base = model.module # Access original model methods/attributes
    else:
        log.info("Using a single GPU or CPU.")

    model_base._optimizer_mode = "initial_unknown_state" # Custom attribute to track optimizer state
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model initialized ({num_params:,} total parameters).")

    # Optimizer and scaler initialized before loading checkpoint or starting training loop
    # This initial optimizer is a placeholder; it will be properly configured based on phase/checkpoint.
    optimizer = optim.AdamW(model_base.parameters(), lr=config.learning_rate)
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0 # Default start epoch (0-indexed for loop, 1-indexed for display)
    # Determine checkpoint path (absolute or relative to checkpoint_dir)
    if os.path.isabs(config.latest_checkpoint_file):
        latest_checkpoint_path = config.latest_checkpoint_file
    else:
        latest_checkpoint_path = os.path.join(config.checkpoint_dir, config.latest_checkpoint_file)

    log.info(f"Checking for checkpoint at: {latest_checkpoint_path}")
    if os.path.exists(latest_checkpoint_path):
        log.info(f"Found potential checkpoint: {latest_checkpoint_path}")
        try:
            # Pass the placeholder optimizer and scaler to be populated by load_checkpoint
            start_epoch, resumed_optimizer_mode = load_checkpoint(latest_checkpoint_path, model_base, optimizer, scaler, device)
            model_base._optimizer_mode = resumed_optimizer_mode # Restore optimizer mode
            log.info(f"Resumed successfully. Training from Epoch {start_epoch + 1}. Optimizer mode was '{resumed_optimizer_mode}'.")
        except Exception as e:
            log.error(f"ERROR loading checkpoint: {e}\n{traceback.format_exc()}")
            log.warning("Starting training from scratch due to checkpoint load failure.")
            start_epoch = 0
            model_base._optimizer_mode = "initial_unknown_state" # Reset mode
            optimizer = optim.AdamW(model_base.parameters(), lr=config.learning_rate) # Re-init placeholder
            scaler = GradScaler(enabled=amp_enabled) # Re-init scaler
    else:
        log.info("No checkpoint found. Starting training from scratch.")
        model_base._optimizer_mode = "initial_unknown_state"

    model.to(device) # Ensure model is on the correct device after potential load_state_dict

    log.info(f"--- Starting Training Loop from Epoch {start_epoch + 1} up to {config.epochs} ---")
    total_batches = len(train_loader)
    if total_batches == 0:
        log.error("ERROR: DataLoader is empty! Cannot train.")
        return

    for epoch in range(start_epoch, config.epochs):
        model.train() # Set model to training mode (affects BatchNorm, Dropout, EBs' internal `training` flag)
        current_epoch_1idx = epoch + 1

        # Determine current training phase
        current_phase_numeric = 0
        if current_epoch_1idx >= config.phase3_start_epoch:
            current_phase_numeric = 3
            if not msssim_available and config.lambda_msssim_phase3 > 0: # Check if MS-SSIM needed and available
                log.error(f"Epoch {current_epoch_1idx}: MS-SSIM not available, but required for Phase 3. Stopping training.")
                break
        elif current_epoch_1idx <= config.phase1_epochs:
            current_phase_numeric = 1
        else:
            current_phase_numeric = 2

        # --- Phase-specific setup for requires_grad, module train/eval modes, LR, and optimizer mode string ---
        phase_description = ""
        target_lr = config.learning_rate
        target_optimizer_mode = ""

        if current_phase_numeric == 1:
            phase_description = f"Phase 1 (Train: MotionAE, ResAE(GTWarp), EBs | MCN Frozen: {config.freeze_mcn_phase1})"
            target_lr = config.learning_rate
            target_optimizer_mode = "phase1"
            
            for name, module_item in model_base.named_children():
                is_mcn = (name == 'motion_compensation_net')
                if is_mcn and config.freeze_mcn_phase1:
                    module_item.eval() # Set MCN to eval mode
                    for param in module_item.parameters(): param.requires_grad = False
                else:
                    module_item.train() # Ensure other modules are in train mode
                    for param in module_item.parameters(): param.requires_grad = True

        elif current_phase_numeric == 2:
            phase_description = "Phase 2 (Train: Full Model - MotionAE, ResAE(MCN), MCN, EBs)"
            target_lr = config.learning_rate
            target_optimizer_mode = "phase2_full_model"
            for module_item in model_base.children():
                module_item.train() # All modules in train mode
                for param in module_item.parameters(): param.requires_grad = True

        elif current_phase_numeric == 3:
            phase_description = "Phase 3 (Train: ResAE Only | MS-SSIM, MSE_Res, BPP_Res)"
            target_lr = config.learning_rate_phase3
            target_optimizer_mode = "phase3_residual_ae_only"

            modules_to_freeze_names = ['motion_encoder', 'motion_entropy_bottleneck', 'motion_decoder', 'motion_compensation_net']
            modules_to_train_names = ['residual_encoder', 'residual_entropy_bottleneck', 'residual_decoder']
            
            for name, module_item in model_base.named_children():
                if name in modules_to_freeze_names:
                    module_item.eval() # Set to eval mode (stops EB CDF updates, affects BN/Dropout)
                    for param in module_item.parameters(): param.requires_grad = False
                elif name in modules_to_train_names:
                    module_item.train() # Ensure in train mode
                    for param in module_item.parameters(): param.requires_grad = True
                elif name == 'warping_layer': # Passive layer
                    pass 
                # else: log.debug(f"Module {name} not explicitly handled in Phase 3 train/freeze logic.")

        log.info(f"\n{'-'*20} Starting Epoch {current_epoch_1idx}/{config.epochs} ({phase_description}) {'-'*20}")

        # --- Optimizer Configuration ---
        # Reconfigure optimizer if mode or LR changed, or if it's the very first setup
        new_optimizer_config_needed = False
        if model_base._optimizer_mode != target_optimizer_mode or \
           optimizer.param_groups[0]['lr'] != target_lr or \
           model_base._optimizer_mode == "initial_unknown_state":
            new_optimizer_config_needed = True

        if new_optimizer_config_needed:
            log.info(f"Configuring optimizer for mode '{target_optimizer_mode}' with LR {target_lr}.")
            active_params = [p for p in model_base.parameters() if p.requires_grad]
            if not active_params:
                log.error(f"CRITICAL: No trainable parameters found for phase {current_phase_numeric}. Training loop might fail or do nothing.")
                # Depending on severity, might `break` or `sys.exit(1)` here.
            optimizer = optim.AdamW(active_params, lr=target_lr, weight_decay=config.weight_decay)
            if amp_enabled: # Reset GradScaler if optimizer changes
                scaler = GradScaler(enabled=amp_enabled)
                log.info(f"Reset GradScaler for new optimizer configuration.")
            model_base._optimizer_mode = target_optimizer_mode
            trainable_params_count = sum(p.numel() for p in active_params)
            log.info(f"Optimizer configured. Trainable parameters: {trainable_params_count:,}")
        else:
            log.debug(f"Optimizer mode '{model_base._optimizer_mode}' and LR {optimizer.param_groups[0]['lr']:.1e} are consistent. No optimizer reconfiguration needed.")


        epoch_metrics = {'loss': 0.0, 'mse_flow': 0.0, 'mse_res': 0.0, 'bpp_mot': 0.0, 'bpp_res': 0.0,
                         'ms_ssim_val': 0.0, # Specific to Phase 3
                         'sum_kb_mot_calculated': 0.0, 'sum_kb_res_calculated': 0.0, 'batches_with_kb_calc': 0}
        processed_batches_count = 0
        epoch_start_time = time.time()
        batch_iter_desc = f"Epoch {current_epoch_1idx} [P{current_phase_numeric}, LR:{optimizer.param_groups[0]['lr']:.1e}]"
        batch_iter = tqdm(enumerate(train_loader), total=total_batches, desc=batch_iter_desc, leave=True, unit="batch")
        last_batch_data_for_viz = None

        for i, batch_data in batch_iter:
            try:
                frame1, frame2_orig, flow_input_gt = batch_data
                frame1 = frame1.to(device, non_blocking=True)
                frame2_orig = frame2_orig.to(device, non_blocking=True)
                flow_input_gt = flow_input_gt.to(device, non_blocking=True)

                B, _, H, W = frame1.shape
                num_pixels_total_batch = float(B * H * W)
                if B == 0 or num_pixels_total_batch <= 0:
                    log.warning(f"Skipping batch {i+1} due to zero batch size or zero pixels.")
                    continue

                optimizer.zero_grad(set_to_none=True) # More memory efficient

                with autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, enabled=amp_enabled):
                    outputs = model(frame1, frame2_orig, flow_input_gt, current_phase=current_phase_numeric)
                    loss = torch.tensor(0.0, device=device)

                    # Common distortion and rate terms (calculated for all phases, used selectively in loss)
                    mse_flow_val = F.mse_loss(outputs['flow_reconstructed'], outputs['flow_input'])
                    mse_residual_val = F.mse_loss(outputs['residual_reconstructed'], outputs['residual_computed'])
                    
                    # Sum over B,H,W for batch total bits, then divide by pixels for BPP
                    total_rate_motion_batch = outputs['rate_motion'].sum()
                    total_rate_residual_batch = outputs['rate_residual'].sum()

                    bpp_motion_term_val = total_rate_motion_batch / num_pixels_total_batch
                    bpp_residual_term_val = total_rate_residual_batch / num_pixels_total_batch

                    ms_ssim_value = torch.tensor(0.0, device=device) # Default for phases not using it

                    if current_phase_numeric == 3:
                        if not msssim_available: # Should have been caught earlier, but as a safeguard
                            raise RuntimeError("MS-SSIM is required for Phase 3 loss but not available.")
                        frame2_reconstructed_final_p3 = outputs['frame2_reconstructed_final']
                        ms_ssim_value = ms_ssim(frame2_reconstructed_final_p3.float(), frame2_orig.float(), data_range=1.0, size_average=True)
                        
                        loss_msssim_term = config.lambda_msssim_phase3 * (1.0 - ms_ssim_value)
                        loss_mse_residual_term = config.lambda_mse_residual * mse_residual_val
                        loss_bpp_residual_term = config.lambda_bpp_residual * bpp_residual_term_val
                        
                        loss = loss_msssim_term + loss_mse_residual_term + loss_bpp_residual_term
                        
                        epoch_metrics['ms_ssim_val'] += ms_ssim_value.item()
                    else: # Phase 1 & 2
                        loss = (config.lambda_mse_flow * mse_flow_val +
                                config.lambda_mse_residual * mse_residual_val +
                                config.lambda_bpp_motion * bpp_motion_term_val +
                                config.lambda_bpp_residual * bpp_residual_term_val)

                # --- Backward pass and optimization step ---
                scaler.scale(loss).backward()
                
                # Unscale before clipping, if clipping is enabled
                scaler.unscale_(optimizer)
                if config.clip_max_norm > 0:
                    # Clip gradients only for parameters that require grad and have a gradient
                    params_to_clip = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                    if params_to_clip: # Avoid error if no params to clip (e.g., all frozen)
                        torch.nn.utils.clip_grad_norm_(params_to_clip, config.clip_max_norm)
                
                scaler.step(optimizer)
                scaler.update()

                # Update Entropy Bottlenecks' CDFs
                # The EBs' internal .training flag (set by module.train()/eval()) controls if .update() has an effect.
                if model.training: # Check overall model training state
                     try:
                         if hasattr(model_base, 'motion_entropy_bottleneck'):
                            model_base.motion_entropy_bottleneck.update()
                         if hasattr(model_base, 'residual_entropy_bottleneck'):
                            model_base.residual_entropy_bottleneck.update()
                     except Exception as update_e: # Catch potential errors during EB update
                         log.warning(f"Warning: EntropyBottleneck update() failed at Batch {i+1}: {update_e}")

                # --- Accumulate metrics ---
                epoch_metrics['loss'] += loss.item()
                epoch_metrics['mse_flow'] += mse_flow_val.item()
                epoch_metrics['mse_res'] += mse_residual_val.item()
                epoch_metrics['bpp_mot'] += bpp_motion_term_val.item()
                epoch_metrics['bpp_res'] += bpp_residual_term_val.item()
                processed_batches_count += 1

                # --- Logging for TQDM postfix ---
                log_postfix = {'L': f"{loss.item():.4f}"}
                if current_phase_numeric == 3:
                    log_postfix.update({
                        'MS-SSIM*': f"{ms_ssim_value.item():.4f}", # Optimized
                        'MSE_R*': f"{mse_residual_val.item():.6f}",    # Optimized
                        'BPP_R*': f"{bpp_residual_term_val.item():.4f}", # Optimized
                        'MSE_F': f"{mse_flow_val.item():.6f}",     # Monitored
                        'BPP_M': f"{bpp_motion_term_val.item():.4f}"  # Monitored
                    })
                else: # Phase 1 & 2
                    log_postfix.update({
                        'MSE_F*': f"{mse_flow_val.item():.6f}", 
                        'MSE_R*': f"{mse_residual_val.item():.6f}",
                        'BPP_M*': f"{bpp_motion_term_val.item():.4f}", 
                        'BPP_R*': f"{bpp_residual_term_val.item():.4f}"
                    })
                
                # --- Optional: Calculate real bitstream size periodically ---
                calc_bitstream_this_batch = (config.bitstream_calc_freq > 0 and \
                                             (i + 1) % config.bitstream_calc_freq == 0 and \
                                             i > 0) # Avoid on first batch if freq=1
                if calc_bitstream_this_batch:
                    try:
                        with torch.no_grad(): # Ensure no gradients computed here
                            # Use latents from current forward pass outputs
                            motion_lat_for_compress = outputs['motion_latents']
                            residual_lat_for_compress = outputs['residual_latents']
                            
                            # Compress. Note: EBs should be in correct train/eval state from phase setup.
                            # .compress() itself calls .quantize(training=False) internally.
                            motion_strings, _ = model_base.motion_entropy_bottleneck.compress(motion_lat_for_compress)
                            residual_strings, _ = model_base.residual_entropy_bottleneck.compress(residual_lat_for_compress)
                            
                            batch_kb_motion = sum(len(s) for s in motion_strings) / 1024.0
                            batch_kb_residual = sum(len(s) for s in residual_strings) / 1024.0
                            
                            epoch_metrics['sum_kb_mot_calculated'] += batch_kb_motion
                            epoch_metrics['sum_kb_res_calculated'] += batch_kb_residual
                            epoch_metrics['batches_with_kb_calc'] += 1
                            log_postfix['KB_M(R)'] = f"{batch_kb_motion:.1f}" # (R) for Real bitstream
                            log_postfix['KB_R(R)'] = f"{batch_kb_residual:.1f}"
                    except Exception as compress_e:
                        log.warning(f"Warning: Bitstream calculation (compress) failed at Batch {i+1}: {compress_e}")
                batch_iter.set_postfix(log_postfix)

                # --- Store last batch data for visualization ---
                if i == total_batches - 1: # If it's the last batch of the epoch
                    with torch.no_grad():
                        last_batch_data_for_viz = {k: v.detach().cpu() for k, v in outputs.items() if isinstance(v, torch.Tensor)}
                        last_batch_data_for_viz['frame1'] = frame1.detach().cpu()
                        last_batch_data_for_viz['frame2_orig'] = frame2_orig.detach().cpu()

            except RuntimeError as e: # Catch runtime errors, especially OOM
                if "CUDA out of memory" in str(e):
                    log.error(f"CUDA OOM occurred at Epoch {current_epoch_1idx} Batch {i+1}. Skipping batch.")
                    torch.cuda.empty_cache() # Attempt to free memory
                    # Consider reducing batch size or model complexity if this persists.
                else:
                    log.error(f"RUNTIME ERROR E{current_epoch_1idx} B{i+1}: {e}\n{traceback.format_exc()}")
                # Ensure scaler state is consistent if an error occurs mid-batch with AMP
                if amp_enabled and scaler.is_enabled() and scaler._scale is not None: # Check if scaler was used
                    try: scaler.update() # Try to update scaler state, may do nothing if step wasn't called
                    except Exception as scaler_ex: log.warning(f"scaler.update() in error handler failed: {scaler_ex}")
                optimizer.zero_grad(set_to_none=True) # Clear gradients from partial backward pass
                continue # Skip to next batch
            except Exception as e: # Catch any other unexpected errors
                log.error(f"UNEXPECTED ERROR E{current_epoch_1idx} B{i+1}: {e}\n{traceback.format_exc()}")
                if amp_enabled and scaler.is_enabled() and scaler._scale is not None:
                    try: scaler.update()
                    except Exception as scaler_ex: log.warning(f"scaler.update() in error handler failed: {scaler_ex}")
                optimizer.zero_grad(set_to_none=True)
                continue

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        if processed_batches_count > 0:
            # Calculate average metrics for the epoch
            avg_metrics = {k: v / processed_batches_count for k, v in epoch_metrics.items()
                           if k not in ['sum_kb_mot_calculated', 'sum_kb_res_calculated', 'batches_with_kb_calc']}
            num_kb_batches = epoch_metrics['batches_with_kb_calc']
            avg_kb_motion = epoch_metrics['sum_kb_mot_calculated'] / num_kb_batches if num_kb_batches > 0 else 0.0
            avg_kb_residual = epoch_metrics['sum_kb_res_calculated'] / num_kb_batches if num_kb_batches > 0 else 0.0
            avg_metrics.update({'bitstream_kb_mot': avg_kb_motion, 
                                'bitstream_kb_res': avg_kb_residual, 
                                'batches_with_kb_calc': num_kb_batches})

            # Log epoch summary
            log.info("-" * 60)
            log.info(f"Epoch {current_epoch_1idx}/{config.epochs} Summary ({phase_description}) | Time: {epoch_duration:.2f}s")
            log.info(f"  Avg Loss: {avg_metrics['loss']:.5f}")
            if current_phase_numeric == 3:
                log.info(f"  Avg MS-SSIM (opt): {avg_metrics['ms_ssim_val']:.5f}")
                log.info(f"  Avg MSE_Flow (mon): {avg_metrics['mse_flow']:.6f} | MSE_Res (opt): {avg_metrics['mse_res']:.6f}")
                log.info(f"  Avg BPP_M(est,mon): {avg_metrics['bpp_mot']:.5f} | BPP_R(est,opt): {avg_metrics['bpp_res']:.5f}")
            else: # Phase 1 & 2
                log.info(f"  Avg MSE_Flow (opt): {avg_metrics['mse_flow']:.6f} | MSE_Res (opt): {avg_metrics['mse_res']:.6f}")
                log.info(f"  Avg BPP_M(est,opt): {avg_metrics['bpp_mot']:.5f} | BPP_R(est,opt): {avg_metrics['bpp_res']:.5f}")

            if num_kb_batches > 0:
                log.info(f"  Avg KB_M(real): {avg_kb_motion:.2f} | KB_R(real): {avg_kb_residual:.2f} (from {num_kb_batches} batches)")
            log.info("-" * 60)

            # Save checkpoint
            ckpt_state = {
                'epoch': epoch, # Save 0-indexed epoch
                'state_dict': model_base.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config_vars, # Save config used for this training run
                'scaler_state_dict': scaler.state_dict() if amp_enabled else None,
                'optimizer_mode': model_base._optimizer_mode # Save the current optimizer mode
            }
            # latest_filename_for_saving will be either the absolute path from config
            # or just the relative filename from config.
            latest_filename_for_saving = config.latest_checkpoint_file
            save_checkpoint(ckpt_state, config.checkpoint_dir, latest_filename=latest_filename_for_saving)

            # Visualize last batch
            if last_batch_data_for_viz and plotting_available:
                 visualize_epoch_end(epoch=current_epoch_1idx, save_dir=config.vis_dir, 
                                     data_dict=last_batch_data_for_viz,
                                     avg_metrics=avg_metrics, 
                                     current_phase_numeric=current_phase_numeric)
        else:
            log.warning(f"Epoch {current_epoch_1idx} completed with 0 successful batches. No metrics to log or checkpoint to save.")

        if device.type == 'cuda': torch.cuda.empty_cache() # Clear CUDA cache at end of epoch

    log.info("="*60 + "\n--- Training Finished ---")
    if file_handler in log.handlers: # Clean up file handler
        log.removeHandler(file_handler)
        file_handler.close()

if __name__ == "__main__":
    config = TrainConfig() # Instantiate default config
    
    # --- Optional: Argument Parsing to override TrainConfig ---
    # Example:
    # parser = argparse.ArgumentParser(description="Video Codec Training")
    # parser.add_argument("--epochs", type=int, help="Number of training epochs")
    # parser.add_argument("--lr", type=float, dest="learning_rate", help="Initial learning rate")
    # parser.add_argument("--batch_size", type=int, help="Batch size")
    # ... add more arguments for other config fields ...
    # args = parser.parse_args()
    # # Update config object with parsed arguments
    # for k, v in vars(args).items():
    #     if v is not None and hasattr(config, k):
    #         setattr(config, k, v)
    #         log.info(f"Overriding config: {k} = {v}")
    # --- End Optional Argument Parsing ---

    try:
        main(config)
    except KeyboardInterrupt:
        log.warning("\n--- Training Interrupted by User (KeyboardInterrupt) ---")
        # Ensure log file is closed properly if main_log_obj is used globally
        main_log_obj = logging.getLogger() 
        for handler_obj in list(main_log_obj.handlers): # Iterate over a copy
            if isinstance(handler_obj, logging.FileHandler):
                handler_obj.close()
                main_log_obj.removeHandler(handler_obj)
        sys.exit(0) # Exit gracefully
    except Exception as e_global:
        log.critical(f"\nFATAL UNHANDLED ERROR during execution: {e_global}\n{traceback.format_exc()}")
        main_log_obj = logging.getLogger()
        for handler_obj in list(main_log_obj.handlers):
            if isinstance(handler_obj, logging.FileHandler):
                handler_obj.close()
                main_log_obj.removeHandler(handler_obj)
        sys.exit(1) # Exit with error code
