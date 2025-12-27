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

    def forward(self, frame1, frame2, flow_input, current_phase=2): # current_phase can be 1, 2, or 3
        """
        Forward pass for training. Behavior depends on the phase.
        """
        # --- Motion Path ---
        motion_latents = self.motion_encoder(flow_input)
        quantized_motion_latents, motion_likelihoods = self.motion_entropy_bottleneck(motion_latents)
        motion_likelihoods = torch.clamp(motion_likelihoods, min=self.likelihood_bound)
        rate_motion = -torch.log2(motion_likelihoods).sum()
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        # --- Motion Compensation & Residual Computation ---
        frame2_motion_compensated = None
        if current_phase == 1:
            warped_frame1_p1 = self.warping_layer(frame1, flow_input) # Phase 1 uses GT flow for warping directly
            residual_computed = frame2 - warped_frame1_p1
            frame2_motion_compensated = warped_frame1_p1
        else: # Phase 2 or 3: Use reconstructed flow and MCN
            warped_frame1_p23 = self.warping_layer(frame1, flow_reconstructed)
            frame2_motion_compensated = self.motion_compensation_net(warped_frame1_p23, flow_reconstructed, frame1)
            residual_computed = frame2 - frame2_motion_compensated

        # --- Residual Path ---
        residual_latents = self.residual_encoder(residual_computed)
        quantized_residual_latents, residual_likelihoods = self.residual_entropy_bottleneck(residual_latents)
        residual_likelihoods = torch.clamp(residual_likelihoods, min=self.likelihood_bound)
        rate_residual = -torch.log2(residual_likelihoods).sum()
        residual_reconstructed = self.residual_decoder(quantized_residual_latents)

        # --- Final Frame Reconstruction ---
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
        self.eval()
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
        self.eval()
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
    if a.ndim == 3 and b.ndim == 3 and a.shape[0] == 1 and b.shape[0] == 1:
         a = a.squeeze(0); b = b.squeeze(0)
    a = torch.clamp(a, 0.0, max_val); b = torch.clamp(b, 0.0, max_val)
    mse = torch.mean((a - b) ** 2)
    if mse == 0: return float('inf')
    try:
        psnr_val = 20 * math.log10(max_val / math.sqrt(mse))
    except ValueError:
        psnr_val = float('inf') if mse == 0 else 0.0
    return psnr_val

def save_checkpoint(state, checkpoint_dir, latest_filename="latest_checkpoint_3phase.pth.tar"):
    """Saves the training state directly (simplified, less atomic)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        log.error(f"ERROR loading checkpoint file {checkpoint_path}: {e}")
        raise

    if 'state_dict' not in checkpoint: raise KeyError("'state_dict' missing from checkpoint.")
    state_dict = checkpoint['state_dict']

    if all(k.startswith('module.') for k in state_dict.keys()):
        log.info("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    if any('._orig_mod.' in k for k in state_dict.keys()):
         log.info("Removing '._orig_mod.' prefix from state_dict keys (likely from torch.compile).")
         try:
            from torch.nn.modules.module import _global_parameter_registration_hooks
            _global_parameter_registration_hooks.clear()
         except (ImportError, AttributeError): pass
         state_dict = {re.sub(r'\._orig_mod', '', k) : v for k, v in state_dict.items()}

    log.info("Updating entropy bottlenecks before loading state dict (calling .update(force=True))...")
    try:
        if hasattr(model, 'motion_entropy_bottleneck'):
            model.motion_entropy_bottleneck.update(force=True)
        if hasattr(model, 'residual_entropy_bottleneck'):
            model.residual_entropy_bottleneck.update(force=True)
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
            for state_val in optimizer.state.values(): # Renamed 'state' to 'state_val'
                for k_opt, v_opt in state_val.items():
                    if isinstance(v_opt, torch.Tensor): state_val[k_opt] = v_opt.to(device)
            log.info("Optimizer state loaded successfully.")
        except RuntimeError as e_opt:
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

    start_epoch_val = checkpoint.get('epoch', -1) + 1 # Renamed 'start_epoch' to 'start_epoch_val'
    optimizer_mode = checkpoint.get('optimizer_mode', 'full')
    log.info(f"Checkpoint loaded. Resuming from Epoch {start_epoch_val}. Optimizer mode was '{optimizer_mode}'.")

    del checkpoint
    torch.cuda.empty_cache()
    return start_epoch_val, optimizer_mode

# ==============================================================================
# DATASET
# ==============================================================================
def read_flo_file(filename):
    """Reads a .flo file (optical flow format). Returns None on error."""
    try:
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), np.float32, count=1)
            if not np.isclose(magic[0], 202021.25): return None
            width = np.frombuffer(f.read(4), np.int32, count=1)[0]
            height = np.frombuffer(f.read(4), np.int32, count=1)[0]
            if width <= 0 or height <= 0 or width > 10000 or height > 10000: return None
            data_size = height * width * 2
            data = np.frombuffer(f.read(data_size * 4), np.float32, count=data_size)
            if len(data) != data_size: return None
            if np.isnan(data).any() or np.isinf(data).any(): return None
            return data.reshape((height, width, 2))
    except FileNotFoundError: return None
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
        for f_path in tqdm(all_frames, desc="Grouping Frames", leave=False, disable=True):
            if not f_path.name.startswith(self.frame_prefix): continue
            dir_path = f_path.parent
            if dir_path not in frames_by_dir: frames_by_dir[dir_path] = []
            frames_by_dir[dir_path].append(f_path)

        count_good = 0
        count_total_pairs_checked = 0
        log.info(f"Checking {len(frames_by_dir)} directories for valid frame pairs and flow files...")
        for dir_path, frame_list in tqdm(frames_by_dir.items(), desc="Scanning Dirs", leave=False, disable=True):
            try:
                sorted_frames = sorted(frame_list, key=lambda p: int(re.findall(r'\d+', p.stem[len(self.frame_prefix):])[0]))
            except (IndexError, ValueError):
                log.debug(f"Numeric sort failed for dir {dir_path}, using alphabetical.")
                sorted_frames = sorted(frame_list)

            for i in range(len(sorted_frames) - 1):
                count_total_pairs_checked += 1
                f1_path, f2_path = sorted_frames[i], sorted_frames[i+1]
                try:
                    n1 = int(re.findall(r'\d+', f1_path.stem[len(self.frame_prefix):])[0])
                    n2 = int(re.findall(r'\d+', f2_path.stem[len(self.frame_prefix):])[0])
                    if n2 != n1 + 1: continue
                except (IndexError, ValueError): continue

                relative_frame_path = f1_path.relative_to(self.frame_base_path)
                flow_path = self.flow_base_path / relative_frame_path.with_suffix(".flo")

                if flow_path.is_file():
                    self.pairs.append((str(f1_path), str(f2_path), str(flow_path)))
                    count_good += 1

        log.info(f"Dataset Scan Complete: Found {count_good} valid frame/flow pairs out of {count_total_pairs_checked} potential pairs checked.")
        if not self.pairs:
            log.error("ERROR: No valid frame/flow pairs found! Check frame/flow directories, prefixes, and file integrity.")
            sys.exit(1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1_p, f2_p, fl_p = self.pairs[idx]
        try:
            f1 = Image.open(f1_p).convert('RGB')
            f2 = Image.open(f2_p).convert('RGB')
            flow_np = read_flo_file(fl_p)
            if flow_np is None: raise RuntimeError(f"Failed read flow: {fl_p}")

            f1_t = self.transform(f1)
            f2_t = self.transform(f2)
            flow_t = torch.from_numpy(flow_np.astype(np.float32)).permute(2, 0, 1)

            _, H_f, W_f = f1_t.shape
            C_fl, H_fl, W_fl = flow_t.shape

            if H_f != H_fl or W_f != W_fl:
                flow_r = transforms.functional.resize(flow_t, [H_f, W_f], interpolation=transforms.InterpolationMode.BILINEAR, antialias=False)
                scale_W = float(W_f) / W_fl if W_fl > 0 else 1.0
                scale_H = float(H_f) / H_fl if H_fl > 0 else 1.0
                flow_final = torch.zeros_like(flow_r)
                flow_final[0, :, :] = flow_r[0, :, :] * scale_W
                flow_final[1, :, :] = flow_r[1, :, :] * scale_H
            else:
                flow_final = flow_t

            return f1_t, f2_t, flow_final
        except FileNotFoundError as e:
            log.error(f"ERROR (Dataset Get Item {idx}): File not found - {e}. Check dataset integrity.")
            raise RuntimeError(f"File not found for index {idx}: {e}") from e
        except Exception as e:
            log.error(f"ERROR (Dataset Get Item {idx}): Failed to load item {idx} ({f1_p}, {f2_p}, {fl_p}): {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to process data for index {idx}") from e


# ==============================================================================
# TRAINING SCRIPT CONFIGURATION
# ==============================================================================
class TrainConfig:
    # --- Path to load a specific checkpoint from (set to None or empty string to disable) ---
    load_specific_checkpoint: str = "C:\\Users\\Anis\\Desktop\\XX\\codec_checkpoints_2phase_visual\\latest_checkpoint_3phase.pth.tar" # USER SPECIFIED

    # --- Paths for saving NEW checkpoints, visualizations, and logs ---
    checkpoint_dir: str = "./codec_checkpoints_3phase_CONTINUED" # New save location
    vis_dir: str = "./codec_visualizations_3phase_CONTINUED"    # New vis location
    log_file: str = "training_log_3phase_CONTINUED.txt"        # New log name
    latest_checkpoint_file: str = "latest_checkpoint_3phase_CONTINUED.pth.tar" # New "latest" name

    # --- Dataset Paths ---
    frame_base_dir: str = "./sequence" # Update if your data is elsewhere
    flow_base_dir: str = "./generated_flow" # Update if your data is elsewhere


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
    epochs: int = 1000 # Total epochs for the entire run
    batch_size: int = 4
    learning_rate: float = 1e-4 # For Phase 1 & 2
    weight_decay: float = 1e-5
    clip_max_norm: float = 1.0
    seed: int = 42
    num_workers: int = 2
    use_amp: bool = True
    bitstream_calc_freq: int = 200

    # --- Phase Settings ---
    phase1_epochs: int = 30 # P1 runs for epochs [1, phase1_epochs]
    phase3_start_epoch: int = 90 # P3 starts at this epoch (1-indexed). P2 is between P1_end and P3_start. USER SPECIFIED
    
    # Lambdas for Phase 1 & 2
    lambda_mse_flow: float = 10.0
    lambda_mse_residual: float = 1000.0
    lambda_bpp_motion: float = 1.0
    lambda_bpp_residual: float = 1.0
    
    # Lambdas and LR for Phase 3
    lambda_msssim_phase3: float = 5.0
    lambda_mse_reconstruction_phase3: float = 100.0
    learning_rate_phase3: float = 1e-5
    
    freeze_mcn_phase1: bool = True

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
    log.info(f"Random seed set to {seed}")

def visualize_epoch_end(epoch, save_dir, data_dict, avg_metrics, current_phase_numeric):
    """Generates and saves a visualization grid at the end of an epoch."""
    if not plotting_available or not data_dict:
        if not plotting_available: log.debug("Plotting disabled, skipping visualization.")
        if not data_dict: log.warning("No data available for visualization.")
        return
    os.makedirs(save_dir, exist_ok=True)
    try:
        idx_vis = 0
        # Ensure all necessary keys are present
        required_keys = ['frame1', 'frame2_orig', 'frame2_reconstructed_final',
                         'motion_compensated_image', 'flow_input', 'flow_reconstructed',
                         'residual_computed', 'residual_reconstructed']
        if not all(k in data_dict for k in required_keys):
            missing = [k for k in required_keys if k not in data_dict]
            log.warning(f"Missing keys in data_dict for visualization: {missing}, skipping.")
            return

        def tensor_to_np(tensor):
            if tensor is None: return np.zeros((64,64,3)) # Placeholder if None
            # Ensure tensor is on CPU before converting to numpy
            return tensor[idx_vis].detach().cpu().permute(1, 2, 0).float().numpy()

        frame1_np = tensor_to_np(data_dict['frame1'])
        frame2_orig_np = tensor_to_np(data_dict['frame2_orig'])
        frame2_recon_np = tensor_to_np(data_dict['frame2_reconstructed_final'])
        mc_image_np = tensor_to_np(data_dict['motion_compensated_image'])
        res_comp_np = tensor_to_np(data_dict['residual_computed'])
        res_rec_np = tensor_to_np(data_dict['residual_reconstructed'])

        def flow_to_mag_norm(flow_tensor):
            if flow_tensor is None: return None
            flow_np = flow_tensor[idx_vis].detach().cpu().float().numpy() # Ensure CPU
            mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            mag_norm = mag / (mag.max() + 1e-6)
            return mag_norm

        flow_input_mag_norm = flow_to_mag_norm(data_dict['flow_input'])
        flow_rec_mag_norm = flow_to_mag_norm(data_dict['flow_reconstructed'])

        res_comp_vis = np.clip(res_comp_np * 0.5 + 0.5, 0, 1) if res_comp_np is not None else None
        res_rec_vis = np.clip(res_rec_np * 0.5 + 0.5, 0, 1) if res_rec_np is not None else None
        
        # Detach and move to CPU for PSNR calculation
        psnr_frame_recon = compute_psnr(data_dict['frame2_reconstructed_final'][idx_vis].cpu(), data_dict['frame2_orig'][idx_vis].cpu())
        item_mse_flow = F.mse_loss(data_dict['flow_reconstructed'][idx_vis].cpu(), data_dict['flow_input'][idx_vis].cpu()).item() if flow_rec_mag_norm is not None else float('nan')
        item_mse_res = F.mse_loss(data_dict['residual_reconstructed'][idx_vis].cpu(), data_dict['residual_computed'][idx_vis].cpu()).item() if res_rec_vis is not None else float('nan')


        fig, axs = plt.subplots(4, 2, figsize=(12, 18))
        fig. GCF().set_facecolor('white')

        phase_str = f"Phase {current_phase_numeric}"
        avg_loss = avg_metrics.get('loss', float('nan'))
        title_metrics = []
        if current_phase_numeric == 3:
            avg_msssim = avg_metrics.get('ms_ssim_val', float('nan'))
            avg_psnr = avg_metrics.get('psnr_val', float('nan'))
            avg_mse_rec_p3 = avg_metrics.get('mse_reconstruction_p3', float('nan'))
            title_metrics.append(f"Avg L(MSS+MSE):{avg_loss:.4f} MS-SSIM:{avg_msssim:.4f} PSNR:{avg_psnr:.2f}dB MSE_Rec:{avg_mse_rec_p3:.6f}")
        else:
            avg_mse_flow = avg_metrics.get('mse_flow', float('nan'))
            avg_mse_res = avg_metrics.get('mse_res', float('nan'))
            avg_bpp_m = avg_metrics.get('bpp_mot', float('nan'))
            avg_bpp_r = avg_metrics.get('bpp_res', float('nan'))
            title_metrics.extend([
                f"Avg L:{avg_loss:.4f} MSE_F:{avg_mse_flow:.6f} MSE_R:{avg_mse_res:.6f}",
                f"Avg BPP M(E):{avg_bpp_m:.4f} R(E):{avg_bpp_r:.4f}"
            ])
        if avg_metrics.get('batches_with_kb_calc', 0) > 0:
                kb_m = avg_metrics.get('bitstream_kb_mot', float('nan'))
                kb_r = avg_metrics.get('bitstream_kb_res', float('nan'))
                title_metrics.append(f"Avg KB M(R):{kb_m:.2f} R(R):{kb_r:.2f}")


        fig.suptitle(f"Epoch {epoch} [{phase_str}] | {' | '.join(title_metrics)}", fontsize=12)

        axs[0, 0].imshow(np.clip(frame2_orig_np, 0, 1)); axs[0, 0].set_title("Original Frame 2"); axs[0, 0].axis("off")
        axs[0, 1].imshow(np.clip(frame2_recon_np, 0, 1)); axs[0, 1].set_title(f"Recon Frame 2 (PSNR: {psnr_frame_recon:.2f}dB)"); axs[0, 1].axis("off")

        mc_type_str = "Warped (GT Flow)" if current_phase_numeric == 1 else "MCN Output (Recon Flow)"
        if current_phase_numeric == 3: mc_type_str += " [AEs+MCN Tuned]"
        axs[1, 0].imshow(np.clip(mc_image_np, 0, 1)); axs[1, 0].set_title(f"Motion Comp. ({mc_type_str})"); axs[1, 0].axis("off")
        axs[1, 1].imshow(np.clip(frame1_np, 0, 1)); axs[1, 1].set_title("Reference Frame 1"); axs[1, 1].axis("off")

        if flow_input_mag_norm is not None: axs[2, 0].imshow(flow_input_mag_norm, cmap='viridis'); axs[2, 0].set_title("Input Flow Mag (Norm)"); axs[2, 0].axis("off")
        else: axs[2,0].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[2, 0].set_title("Input Flow Mag (Norm)"); axs[2, 0].axis("off")
        if flow_rec_mag_norm is not None: axs[2, 1].imshow(flow_rec_mag_norm, cmap='viridis'); axs[2, 1].set_title(f"Recon Flow Mag (MSE: {item_mse_flow:.6f})"); axs[2, 1].axis("off")
        else: axs[2,1].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[2, 1].set_title(f"Recon Flow Mag (MSE: {item_mse_flow:.6f})"); axs[2, 1].axis("off")

        res_target_desc = "Target: F2 - Warp(F1, GTFlow)" if current_phase_numeric == 1 else "Target: F2 - MCN(Warp(F1, RecFlow))"
        if current_phase_numeric == 3: res_target_desc += " [Frozen EBs]"
        if res_comp_vis is not None: axs[3, 0].imshow(res_comp_vis); axs[3, 0].set_title(res_target_desc); axs[3, 0].axis("off")
        else: axs[3,0].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[3, 0].set_title(res_target_desc); axs[3, 0].axis("off")
        if res_rec_vis is not None: axs[3, 1].imshow(res_rec_vis); axs[3, 1].set_title(f"Recon Residual (MSE: {item_mse_res:.6f})"); axs[3, 1].axis("off")
        else: axs[3,1].text(0.5, 0.5, 'N/A', ha='center', va='center'); axs[3, 1].set_title(f"Recon Residual (MSE: {item_mse_res:.6f})"); axs[3, 1].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_phase{current_phase_numeric}_vis.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        log.debug(f"Visualization saved to {save_path}")
    except Exception as e:
        log.error(f"ERROR during visualization for Epoch {epoch}: {e}\n{traceback.format_exc()}")
        if 'plt' in locals() and 'fig' in locals() and hasattr(fig, 'number') and plt.fignum_exists(fig.number):
            plt.close(fig)

def main(config: TrainConfig):
    set_seed(config.seed)
    amp_enabled = config.use_amp and config.device == "cuda" and amp_available
    device = torch.device(config.device)
    log.info(f"Using device: {device}, AMP Enabled: {amp_enabled}")

    # Ensure NEW checkpoint/vis/log directories exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.vis_dir, exist_ok=True)
    log_path = os.path.join(config.checkpoint_dir, config.log_file) # Log in new checkpoint dir
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    log.addHandler(file_handler)

    log.info("="*60 + "\nStarting Training Session (3-Phase - Specific Load & Continue)\n" + "="*60)
    log.info(f"Attempting to load specific checkpoint from: {config.load_specific_checkpoint if config.load_specific_checkpoint else 'Not specified'}")
    log.info(f"NEW checkpoints will be saved in: {config.checkpoint_dir}")
    log.info(f"NEW visualizations will be saved in: {config.vis_dir}")
    log.info(f"NEW log file: {log_path}")
    config_vars = {k: getattr(config, k) for k, v in vars(TrainConfig).items() if not k.startswith('__') and not callable(v)}
    log.info("Configuration:")
    for k, v in config_vars.items(): log.info(f"  {k}: {v}")
    log.info("-" * 60)

    log.info("Setting up dataset...")
    img_transform = transforms.Compose([transforms.ToTensor()])
    try:
        train_dataset = VideoFrameFlowDatasetNested(config.frame_base_dir, config.flow_base_dir, transform=img_transform)
        if len(train_dataset) == 0:
             log.error("Dataset is empty after initialization. Exiting.")
             return
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=(device.type == 'cuda'),
                                  drop_last=True)
        log.info(f"Dataset loaded: {len(train_dataset)} pairs.")
        log.info(f"DataLoader created: {len(train_loader)} batches per epoch (Batch Size: {config.batch_size}, Drop Last: True).")
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

    model_base = model
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)
        model_base = model.module
    else:
        log.info("Using a single GPU or CPU.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model initialized ({num_params:,} total trainable parameters initially).")

    # Optimizer and scaler will be potentially loaded/re-initialized based on checkpoint
    optimizer = optim.AdamW(model_base.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=amp_enabled)
    model_base._optimizer_mode = "full" # Default mode, will be updated if checkpoint loaded

    # --- Checkpoint Loading ---
    start_epoch = 0
    resumed_optimizer_mode = "full"

    checkpoint_to_load_path = None
    if config.load_specific_checkpoint and os.path.exists(config.load_specific_checkpoint):
        checkpoint_to_load_path = config.load_specific_checkpoint
        log.info(f"Selected specific checkpoint for loading: {checkpoint_to_load_path}")
    else:
        # Fallback to loading "latest" from the *new* checkpoint_dir if specific one not provided/found
        # This is useful if the script was stopped and is being restarted with the same *new* save directory.
        potential_latest_path_in_new_dir = os.path.join(config.checkpoint_dir, config.latest_checkpoint_file)
        if os.path.exists(potential_latest_path_in_new_dir):
            checkpoint_to_load_path = potential_latest_path_in_new_dir
            log.info(f"Specific checkpoint '{config.load_specific_checkpoint}' not specified or not found. "
                     f"Attempting to load latest from NEW save dir: {checkpoint_to_load_path}")
        elif config.load_specific_checkpoint : # If it was specified but not found
             log.warning(f"Specified checkpoint {config.load_specific_checkpoint} was NOT found. "
                         "Will check for latest in new save dir or start fresh.")


    if checkpoint_to_load_path and os.path.exists(checkpoint_to_load_path):
        log.info(f"Attempting to load checkpoint: {checkpoint_to_load_path}")
        try:
            # Pass the current optimizer and scaler to be potentially updated by load_checkpoint
            start_epoch, resumed_optimizer_mode = load_checkpoint(checkpoint_to_load_path, model_base, optimizer, scaler, device)
            model_base._optimizer_mode = resumed_optimizer_mode # Crucial: restore optimizer mode
            log.info(f"Resumed successfully from checkpoint '{checkpoint_to_load_path}'. Training will start from Epoch {start_epoch + 1}. Optimizer mode was '{resumed_optimizer_mode}'.")
        except FileNotFoundError:
            log.info(f"Checkpoint file '{checkpoint_to_load_path}' reported existing but was not found during load. Starting from scratch.")
            start_epoch = 0
            model_base._optimizer_mode = "full"
            optimizer = optim.AdamW(model_base.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay) # Re-init
            scaler = GradScaler(enabled=amp_enabled) # Re-init
        except Exception as e:
            log.error(f"ERROR loading checkpoint '{checkpoint_to_load_path}': {e}\n{traceback.format_exc()}")
            log.warning("Starting training from scratch due to checkpoint load failure.")
            start_epoch = 0
            model_base._optimizer_mode = "full"
            optimizer = optim.AdamW(model_base.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay) # Re-init
            scaler = GradScaler(enabled=amp_enabled) # Re-init
    else:
        if config.load_specific_checkpoint and not os.path.exists(config.load_specific_checkpoint) :
            log.warning(f"Specified checkpoint '{config.load_specific_checkpoint}' not found.")
        log.info("No existing checkpoint found to load (neither specific nor latest in new save dir). Starting training from scratch.")
        start_epoch = 0
        model_base._optimizer_mode = "full"
        # Optimizer and scaler are already initialized for starting fresh

    model.to(device) # Ensure model is on device after potential load_checkpoint map_location='cpu'

    log.info(f"--- Starting Training Loop from Epoch {start_epoch + 1} up to {config.epochs} ---")
    total_batches = len(train_loader)
    if total_batches == 0:
        log.error("ERROR: DataLoader is empty! Cannot train.")
        return

    # --- Main Training Loop ---
    for epoch in range(start_epoch, config.epochs):
        model.train()
        current_epoch_1idx = epoch + 1

        current_phase_numeric = 0
        if current_epoch_1idx >= config.phase3_start_epoch:
            current_phase_numeric = 3
            if not msssim_available:
                log.error(f"Epoch {current_epoch_1idx}: MS-SSIM not available (pytorch-msssim not installed). Stopping training as it's required for Phase 3.")
                break
        elif current_epoch_1idx <= config.phase1_epochs:
            current_phase_numeric = 1
        else: # Between phase1_epochs and phase3_start_epoch
            current_phase_numeric = 2

        phase_description = f"Phase {current_phase_numeric}"
        if current_phase_numeric == 1: phase_description += " (Train: MotionAE, ResAE(GTWarp) | Frozen: MCN)"
        elif current_phase_numeric == 2: phase_description += " (Train: MotionAE, ResAE(MCN), MCN | All Trainable)"
        elif current_phase_numeric == 3: phase_description += " (Train: Encoders, Decoders, MCN | Frozen: EBs | Loss: MS-SSIM+MSE)"
        log.info(f"\n{'-'*20} Starting Epoch {current_epoch_1idx}/{config.epochs} ({phase_description}) {'-'*20}")

        phase3_optimizer_mode_name = "phase3_ae_mcn_ft"
        current_lr_for_phase = config.learning_rate # Default for P1/P2

        if current_phase_numeric == 3:
            current_lr_for_phase = config.learning_rate_phase3
            if model_base._optimizer_mode != phase3_optimizer_mode_name:
                log.info(f"Configuring optimizer for Phase 3 mode ('{phase3_optimizer_mode_name}', LR: {config.learning_rate_phase3}).")
                for param in model_base.parameters(): param.requires_grad = False
                trainable_modules_p3 = [model_base.motion_encoder, model_base.motion_decoder, model_base.residual_encoder, model_base.residual_decoder, model_base.motion_compensation_net]
                for module_to_train in trainable_modules_p3:
                    for param in module_to_train.parameters(): param.requires_grad = True
                for param in model_base.motion_entropy_bottleneck.parameters(): param.requires_grad = False
                for param in model_base.residual_entropy_bottleneck.parameters(): param.requires_grad = False
                
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_base.parameters()), lr=config.learning_rate_phase3, weight_decay=config.weight_decay)
                if amp_enabled: scaler = GradScaler(enabled=amp_enabled); log.info("Reset GradScaler for Phase 3.")
                model_base._optimizer_mode = phase3_optimizer_mode_name
                num_trainable_p3 = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
                log.info(f"Optimizer configured for Phase 3. Trainable params: {num_trainable_p3:,}")
        else: # Phase 1 or 2
            current_lr_for_phase = config.learning_rate
            if model_base._optimizer_mode != "full":
                log.info(f"Switching optimizer to Phase 1/2 mode ('full' model, LR: {config.learning_rate}).")
                for param in model_base.parameters(): param.requires_grad = True
                optimizer = optim.AdamW(model_base.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                if amp_enabled: scaler = GradScaler(enabled=amp_enabled); log.info("Reset GradScaler for Phase 1/2.")
                model_base._optimizer_mode = "full"
            
            for param in model_base.parameters(): param.requires_grad = True # Default for P1/P2
            if current_phase_numeric == 1 and config.freeze_mcn_phase1:
                # log.info("Phase 1: Freezing MCN parameters.") # Can be verbose
                for param in model_base.motion_compensation_net.parameters(): param.requires_grad = False
            elif current_phase_numeric == 2:
                 for param in model_base.motion_compensation_net.parameters(): param.requires_grad = True
        
        # Ensure optimizer's LR is correct for the current phase, especially after loading checkpoint
        for param_group in optimizer.param_groups:
            if abs(param_group['lr'] - current_lr_for_phase) > 1e-7:
                log.info(f"Updating optimizer LR for Epoch {current_epoch_1idx} (Phase {current_phase_numeric}) from {param_group['lr']:.2e} to {current_lr_for_phase:.2e}")
                param_group['lr'] = current_lr_for_phase


        epoch_metrics = {'loss': 0.0, 'mse_flow': 0.0, 'mse_res': 0.0, 'bpp_mot': 0.0, 'bpp_res': 0.0,
                         'ms_ssim_val': 0.0, 'psnr_val': 0.0, 'mse_reconstruction_p3': 0.0,
                         'sum_kb_mot_calculated': 0.0, 'sum_kb_res_calculated': 0.0, 'batches_with_kb_calc': 0}
        processed_batches_count = 0
        epoch_start_time = time.time()
        batch_iter_desc = f"Epoch {current_epoch_1idx} [P{current_phase_numeric}]"
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
                if B == 0 or num_pixels_total_batch <= 0: continue

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16, enabled=amp_enabled):
                    outputs = model(frame1, frame2_orig, flow_input_gt, current_phase=current_phase_numeric)
                    loss_val = torch.tensor(0.0, device=device) # Renamed 'loss' to 'loss_val'
                    
                    if current_phase_numeric == 3:
                        frame2_reconstructed_final_p3 = outputs['frame2_reconstructed_final']
                        ms_ssim_value = ms_ssim(frame2_reconstructed_final_p3.float(), frame2_orig.float(), data_range=1.0, size_average=True)
                        loss_msssim = config.lambda_msssim_phase3 * (1.0 - ms_ssim_value)
                        mse_reconstruction = F.mse_loss(frame2_reconstructed_final_p3, frame2_orig)
                        loss_mse_rec = config.lambda_mse_reconstruction_phase3 * mse_reconstruction
                        loss_val = loss_msssim + loss_mse_rec
                        
                        with torch.no_grad():
                            psnr_val_batch = compute_psnr(frame2_reconstructed_final_p3, frame2_orig)
                        
                        epoch_metrics['ms_ssim_val'] += ms_ssim_value.item()
                        epoch_metrics['psnr_val'] += psnr_val_batch
                        epoch_metrics['mse_reconstruction_p3'] += mse_reconstruction.item()
                    else: # Phase 1 & 2
                        mse_flow = F.mse_loss(outputs['flow_reconstructed'], outputs['flow_input'])
                        mse_residual = F.mse_loss(outputs['residual_reconstructed'], outputs['residual_computed'])
                        total_rate_motion = outputs['rate_motion'].sum()
                        total_rate_residual = outputs['rate_residual'].sum()
                        bpp_motion_term = total_rate_motion / num_pixels_total_batch
                        bpp_residual_term = total_rate_residual / num_pixels_total_batch

                        loss_val = (config.lambda_mse_flow * mse_flow +
                                config.lambda_mse_residual * mse_residual +
                                config.lambda_bpp_motion * bpp_motion_term +
                                config.lambda_bpp_residual * bpp_residual_term)

                        epoch_metrics['mse_flow'] += mse_flow.item()
                        epoch_metrics['mse_res'] += mse_residual.item()
                        epoch_metrics['bpp_mot'] += bpp_motion_term.item()
                        epoch_metrics['bpp_res'] += bpp_residual_term.item()

                scaler.scale(loss_val).backward()
                scaler.unscale_(optimizer)
                trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if config.clip_max_norm > 0 and trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.clip_max_norm)
                scaler.step(optimizer)
                scaler.update()

                if model.training and current_phase_numeric != 3: # No EB update in P3
                     try:
                         if hasattr(model_base, 'motion_entropy_bottleneck') and model_base.motion_entropy_bottleneck.training:
                            model_base.motion_entropy_bottleneck.update()
                         if hasattr(model_base, 'residual_entropy_bottleneck') and model_base.residual_entropy_bottleneck.training:
                            model_base.residual_entropy_bottleneck.update()
                     except AttributeError as ae:
                         log.error(f"AttributeError during EB update: {ae}. Check model_base structure.")
                     except Exception as update_e:
                         log.warning(f"Warning: EB update() failed at B{i+1}: {update_e}")

                epoch_metrics['loss'] += loss_val.item()
                processed_batches_count += 1

                log_postfix = {'L': f"{loss_val.item():.4f}"}
                if current_phase_numeric == 3:
                    log_postfix['MS-SSIM'] = f"{ms_ssim_value.item():.4f}"
                    log_postfix['PSNR'] = f"{psnr_val_batch:.2f}"
                    log_postfix['MSE_Rec'] = f"{mse_reconstruction.item():.6f}"
                else:
                    bpp_m_val_item = bpp_motion_term.item() if 'bpp_motion_term' in locals() else float('nan')
                    bpp_r_val_item = bpp_residual_term.item() if 'bpp_residual_term' in locals() else float('nan')
                    mse_flow_item = mse_flow.item() if 'mse_flow' in locals() else float('nan')
                    mse_res_item = mse_residual.item() if 'mse_residual' in locals() else float('nan')
                    log_postfix.update({'MSE_F': f"{mse_flow_item:.6f}", 'MSE_R': f"{mse_res_item:.6f}",
                                       'BPP_M': f"{bpp_m_val_item:.4f}", 'BPP_R': f"{bpp_r_val_item:.4f}"})
                
                calc_bitstream_this_batch = (config.bitstream_calc_freq > 0 and (i + 1) % config.bitstream_calc_freq == 0 and i > 0)
                if calc_bitstream_this_batch:
                    try:
                        with torch.no_grad():
                            motion_strings, _ = model_base.motion_entropy_bottleneck.compress(outputs['motion_latents'].detach())
                            residual_strings, _ = model_base.residual_entropy_bottleneck.compress(outputs['residual_latents'].detach())
                            batch_kb_motion = sum(len(s) for s in motion_strings) / 1024.0
                            batch_kb_residual = sum(len(s) for s in residual_strings) / 1024.0
                            epoch_metrics['sum_kb_mot_calculated'] += batch_kb_motion
                            epoch_metrics['sum_kb_res_calculated'] += batch_kb_residual
                            epoch_metrics['batches_with_kb_calc'] += 1
                            log_postfix['KB_M(R)'] = f"{batch_kb_motion:.1f}"
                            log_postfix['KB_R(R)'] = f"{batch_kb_residual:.1f}"
                    except Exception as compress_e:
                        log.warning(f"Warning: Bitstream calculation failed at B{i+1} (E{current_epoch_1idx}): {compress_e}")

                batch_iter.set_postfix(log_postfix)

                if i == total_batches - 1:
                    with torch.no_grad():
                        last_batch_data_for_viz = {k: v.detach().cpu() for k, v in outputs.items() if isinstance(v, torch.Tensor)}
                        last_batch_data_for_viz['frame1'] = frame1.detach().cpu()
                        last_batch_data_for_viz['frame2_orig'] = frame2_orig.detach().cpu()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    log.error(f"CUDA OOM occurred at Epoch {current_epoch_1idx} Batch {i+1}. Skipping batch.")
                    torch.cuda.empty_cache()
                else:
                    log.error(f"RUNTIME ERROR E{current_epoch_1idx} B{i+1}: {e}\n{traceback.format_exc()}")
                if amp_enabled and scaler.is_enabled():
                    try: scaler.update()
                    except Exception as scaler_ex: log.warning(f"scaler.update() in RuntimeError handler failed: {scaler_ex}")
                optimizer.zero_grad(set_to_none=True)
                continue
            except Exception as e:
                log.error(f"UNEXPECTED ERROR E{current_epoch_1idx} B{i+1}: {e}\n{traceback.format_exc()}")
                if amp_enabled and scaler.is_enabled():
                    try: scaler.update()
                    except Exception as scaler_ex: log.warning(f"scaler.update() in Exception handler failed: {scaler_ex}")
                optimizer.zero_grad(set_to_none=True)
                continue

        epoch_duration = time.time() - epoch_start_time
        if processed_batches_count > 0:
            avg_metrics = {k: v / processed_batches_count for k, v in epoch_metrics.items()
                           if k not in ['sum_kb_mot_calculated', 'sum_kb_res_calculated', 'batches_with_kb_calc']}
            num_kb_batches = epoch_metrics['batches_with_kb_calc']
            avg_kb_motion = epoch_metrics['sum_kb_mot_calculated'] / num_kb_batches if num_kb_batches > 0 else 0.0
            avg_kb_residual = epoch_metrics['sum_kb_res_calculated'] / num_kb_batches if num_kb_batches > 0 else 0.0
            avg_metrics.update({'bitstream_kb_mot': avg_kb_motion, 'bitstream_kb_res': avg_kb_residual, 'batches_with_kb_calc': num_kb_batches})

            log.info("-" * 60)
            log.info(f"Epoch {current_epoch_1idx}/{config.epochs} Summary ({phase_description}) | Time: {epoch_duration:.2f}s")
            if current_phase_numeric == 3:
                log.info(f"  Avg Loss (MS-SSIM+MSE): {avg_metrics['loss']:.5f} | Avg MS-SSIM: {avg_metrics['ms_ssim_val']:.5f} | Avg PSNR: {avg_metrics['psnr_val']:.2f}dB | Avg MSE_Rec(P3): {avg_metrics['mse_reconstruction_p3']:.6f}")
            else:
                log.info(f"  Avg Loss: {avg_metrics['loss']:.5f} | MSE_Flow: {avg_metrics['mse_flow']:.6f} | MSE_Res: {avg_metrics['mse_res']:.6f}")
                log.info(f"  Avg BPP_M(est): {avg_metrics['bpp_mot']:.5f} | BPP_R(est): {avg_metrics['bpp_res']:.5f}")
            if num_kb_batches > 0:
                log.info(f"  Avg KB_M(real): {avg_kb_motion:.2f} | KB_R(real): {avg_kb_residual:.2f} (from {num_kb_batches} batches)")
            log.info("-" * 60)

            # Save checkpoint to the NEW directory
            ckpt_state = {
                'epoch': epoch, 'state_dict': model_base.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': config_vars,
                'scaler_state_dict': scaler.state_dict() if amp_enabled else None,
                'optimizer_mode': model_base._optimizer_mode
            }
            save_checkpoint(ckpt_state, config.checkpoint_dir, latest_filename=config.latest_checkpoint_file)

            if last_batch_data_for_viz and plotting_available:
                 visualize_epoch_end(epoch=current_epoch_1idx, save_dir=config.vis_dir, data_dict=last_batch_data_for_viz,
                                     avg_metrics=avg_metrics, current_phase_numeric=current_phase_numeric)
        else:
            log.warning(f"Epoch {current_epoch_1idx} completed with 0 successful batches. No metrics to report or checkpoint to save.")

        if device.type == 'cuda': torch.cuda.empty_cache()

    log.info("="*60 + "\n--- Training Finished ---")
    if file_handler in log.handlers:
        log.removeHandler(file_handler)
        file_handler.close()

if __name__ == "__main__":
    config = TrainConfig()
    # Example: Override config from command line if needed
    # parser = argparse.ArgumentParser(description='Train 3-Phase Video Codec')
    # parser.add_argument('--load_specific_checkpoint', type=str, help='Path to a specific checkpoint to load.')
    # # Add other relevant arguments from TrainConfig
    # args = parser.parse_args()
    # if args.load_specific_checkpoint: config.load_specific_checkpoint = args.load_specific_checkpoint
    # # ... and so on for other arguments

    try:
        main(config)
    except KeyboardInterrupt:
        log.warning("\n--- Training Interrupted by User (KeyboardInterrupt) ---")
        main_log_instance = logging.getLogger()
        for handler in list(main_log_instance.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                main_log_instance.removeHandler(handler)
        sys.exit(0)
    except Exception as e:
        log.critical(f"\nFATAL UNHANDLED ERROR during execution: {e}\n{traceback.format_exc()}")
        main_log_instance = logging.getLogger()
        for handler in list(main_log_instance.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                main_log_instance.removeHandler(handler)
        sys.exit(1)
