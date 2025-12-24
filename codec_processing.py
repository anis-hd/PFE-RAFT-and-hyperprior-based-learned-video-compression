# codec.py
# -*- coding: utf-8 -*-
"""
Unified script for video encoding and decoding using RAFT optical flow
and a learned video codec.

Supports two modes via command-line arguments:
  --encode: Encodes an input video (MP4 or YUV) into a .rdvc file.
  --decode: Decodes a previously compressed .rdvc video file.
"""

# ==============================================================================
# Imports
# ==============================================================================
import argparse
import io
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
import struct # Added for RDVC file format
from skimage.exposure import match_histograms as skimage_match_histograms
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
from tqdm import tqdm

# CompressAI - must be installed
try:
    import compressai
    from compressai.entropy_models import EntropyBottleneck
    # from compressai.ops import quantize_ste # Not explicitly used in this simplified script, but good to have if extending
    print(f"Using compressai version: {compressai.__version__}")
except ImportError:
    print("ERROR: compressai library not found. Please run: pip install compressai")
    sys.exit(1)
except Exception as e:
    print(f"ERROR importing compressai: {e}")
    sys.exit(1)

# RAFT - Availability Checks
_TORCHVISION_RAFT_NEW_AVAILABLE = False
_TORCHVISION_RAFT_OLD_AVAILABLE = False
_LOCAL_RAFT_CORE_AVAILABLE = False

_TV_RAFT_NEW_IMPL = None
_TV_RAFT_NEW_WEIGHTS_ENUM = None
_TV_RAFT_OLD_IMPL = None
_LOCAL_RAFT_CORE_IMPL = None

try:
    from torchvision.models.optical_flow import raft_large as _tv_raft_large_new, Raft_Large_Weights as _Raft_Large_Weights_New
    _TORCHVISION_RAFT_NEW_AVAILABLE = True
    _TV_RAFT_NEW_IMPL = _tv_raft_large_new
    _TV_RAFT_NEW_WEIGHTS_ENUM = _Raft_Large_Weights_New
    print("Found RAFT from torchvision.models.optical_flow (new API with Weights)")
except ImportError:
    print("RAFT from torchvision.models.optical_flow (new API with Weights) not found.")
    try:
        from torchvision.models.raft import raft_large as _tv_raft_large_old # Older torchvision path
        _TORCHVISION_RAFT_OLD_AVAILABLE = True
        _TV_RAFT_OLD_IMPL = _tv_raft_large_old
        print("Found RAFT from torchvision.models.raft (old API)")
    except ImportError:
        print("RAFT from torchvision.models.raft (old API) not found.")

try:
    # Attempt to import from a local 'core' directory (common for RAFT repos)
    if 'core' not in sys.path and os.path.isdir('core'):
        print("Adding local 'core' directory to sys.path for RAFT.")
        sys.path.insert(0, os.path.abspath('core'))
    from raft import RAFT as RAFT_Core_Local # Assuming the class is named RAFT in raft.py
    _LOCAL_RAFT_CORE_AVAILABLE = True
    _LOCAL_RAFT_CORE_IMPL = RAFT_Core_Local
    print("Found RAFT from local 'core' module")
except ImportError:
    print("RAFT from local 'core' module not found.")
except Exception as e:
    print(f"ERROR importing local RAFT: {e}")


if not (_TORCHVISION_RAFT_NEW_AVAILABLE or _TORCHVISION_RAFT_OLD_AVAILABLE or _LOCAL_RAFT_CORE_AVAILABLE):
    print("\nCRITICAL ERROR: No RAFT implementation (torchvision or local) could be imported.\n"
          "Please ensure torchvision is up-to-date or RAFT is available in 'core' directory and its dependencies are met.")
    sys.exit(1)


# ==============================================================================
# RDVC File Format Constants
# ==============================================================================
RDVC_METADATA_MARKER = b"RDVCMETA"
RDVC_FRAME_MARKER = b"RDVCFRME"
RDVC_EOF_MARKER = b"RDVCEND_"

# Struct formats for packing/unpacking integers (Big-endian)
UINT8_FORMAT = ">B"  # Unsigned 1-byte integer
UINT32_FORMAT = ">I" # Unsigned 4-byte integer
INT32_FORMAT = ">i"  # Signed 4-byte integer
UINT64_FORMAT = ">Q" # Unsigned 8-byte integer

# ==============================================================================
# Helper Network Modules (Consistent for Encoder & Decoder)
# ==============================================================================
def get_activation(name="leaky_relu"):
    """Returns the specified activation layer instance."""
    name_lower = name.lower() if name else "none"
    if name_lower == "none":
        return nn.Identity()
    elif name_lower == "relu":
        return nn.ReLU(inplace=True)
    elif name_lower == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name_lower == "gelu": return nn.GELU()
    elif name_lower == "sigmoid": return nn.Sigmoid()
    elif name_lower == "tanh": return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")

class ConvNormAct(nn.Sequential):
    """Convolution -> Norm -> Activation Block."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding='same', # if stride=1, padding=kernel_size//2. if stride > 1, padding needs to be int
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.LeakyReLU(0.2, inplace=True),
        bias=False, # BatchNorm has affine params, so conv bias often False
    ):
        actual_padding = padding
        if isinstance(padding, str) and padding.lower() == 'same':
            if stride == 1:
                actual_padding = kernel_size // 2
            else:
                # For stride > 1, 'same' padding in PyTorch is complex.
                # This calculation is a common way to achieve something similar to TensorFlow's 'SAME'.
                # However, exact 'same' for stride > 1 is typically handled by explicit padding values.
                # For simplicity, we'll use a common formula, but be aware it might not be 'SAME' in all cases.
                if kernel_size % 2 == 0 : raise ValueError("kernel_size must be odd for 'same' padding with stride=1")
                actual_padding = kernel_size // 2 # This is a common approach but might need adjustment for specific stride/kernel combos

        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=actual_padding,
                bias=bias,
            ),
        )
        if norm_layer is not None:
            self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None:
            self.add_module("act", act_layer)

class ConvTransposeNormAct(nn.Sequential):
    """Transposed Convolution -> Norm -> Activation Block."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.LeakyReLU(0.2, inplace=True),
        bias=False,
    ):
        super().__init__()
        self.add_module(
            "conv_transpose",
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
        )
        if norm_layer is not None:
            self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None:
            self.add_module("act", act_layer)

class ResidualBlock(nn.Module):
    """Simple Residual Block: ConvNormAct -> ConvNorm -> Add -> Act."""
    def __init__(
        self,
        channels,
        kernel_size=3,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.LeakyReLU(0.2, inplace=True),
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                channels, channels, kernel_size, stride=1, padding='same',
                norm_layer=norm_layer, act_layer=act_layer
            ),
            ConvNormAct(
                channels, channels, kernel_size, stride=1, padding='same',
                norm_layer=norm_layer, act_layer=None # No activation before adding residual
            ),
        )
        self.final_act = act_layer if act_layer is not None else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.final_act(out)
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        input_channels,
        base_channels=64,
        latent_channels=128,
        num_downsample_layers=3,
        num_res_blocks=2,
    ):
        super().__init__()
        layers = []
        # Initial convolution: using kernel 5, padding 2 to maintain size if stride=1
        layers.append(
            ConvNormAct(
                input_channels, base_channels,
                kernel_size=5, stride=1, padding=2 # 'same' effectively
            )
        )

        current_channels = base_channels
        for _ in range(num_downsample_layers):
            out_ch = current_channels * 2
            layers.append(
                ConvNormAct(
                    current_channels, out_ch,
                    kernel_size=3, stride=2, padding=1 # Standard downsampling block
                )
            )
            current_channels = out_ch

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_channels))

        # Final convolution to latent space
        layers.append(
            nn.Conv2d(
                current_channels, latent_channels,
                kernel_size=3, stride=1, padding=1 # 'same' effectively
            )
        )
        self.encoder = nn.Sequential(*layers)
        self.output_channels = latent_channels

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(
        self,
        output_channels,
        base_channels=64, # Should match encoder's base_channels for symmetry
        latent_channels=128,
        num_upsample_layers=3, # Should match encoder's num_downsample_layers
        num_res_blocks=2,
        final_activation=None, # e.g., "sigmoid" or "tanh" if output is image-like
    ):
        super().__init__()
        layers = []

        # Calculate the number of channels before upsampling starts
        # This should be base_channels * (2^num_upsample_layers)
        channels_before_upsample = base_channels * (2**num_upsample_layers)

        # Initial convolution from latent space
        layers.append(
            ConvNormAct(
                latent_channels, channels_before_upsample,
                kernel_size=3, stride=1, padding=1 # 'same' effectively
            )
        )

        current_channels = channels_before_upsample
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_channels))

        for _ in range(num_upsample_layers):
            out_ch = current_channels // 2
            layers.append(
                ConvTransposeNormAct(
                    current_channels, out_ch,
                    kernel_size=3, stride=2, padding=1, output_padding=1 # Standard upsampling block
                )
            )
            current_channels = out_ch
            # After upsampling, current_channels should eventually become base_channels

        # Final convolution to output channels
        layers.append(
            nn.Conv2d(
                current_channels, output_channels,
                kernel_size=5, stride=1, padding=2 # 'same' effectively, matches initial encoder conv
            )
        )

        if final_activation:
            layers.append(get_activation(final_activation))

        self.decoder = nn.Sequential(*layers)
        self.input_channels = latent_channels

    def forward(self, x):
        return self.decoder(x)

class WarpingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, flow):
        """
        Warps an image x using optical flow.
        Args:
            x (torch.Tensor): Image tensor (B, C, H, W).
            flow (torch.Tensor): Optical flow tensor (B, 2, H, W), where flow[:, 0, :, :] is dx (horizontal)
                                 and flow[:, 1, :, :] is dy (vertical).
        Returns:
            torch.Tensor: Warped image tensor.
        """
        B, C, H, W = x.size()
        if flow.size()[-2:] != (H, W) or flow.size()[1] != 2:
            raise ValueError(
                f"Input image ({B},{C},{H},{W}) and flow ({flow.shape}) shape/channel mismatch."
            )

        # Create a regular grid: range [-1, 1] for x and y
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing='ij', # Important for (H, W) order
        )
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)

        # Normalize flow to be in [-1, 1] range relative to grid
        # flow dx is horizontal, corresponds to grid_x (dim 0 of last dim in 'grid')
        # flow dy is vertical, corresponds to grid_y (dim 1 of last dim in 'grid')
        norm_flow_x = (flow[:, 0, :, :] / ((W - 1) / 2.0)
                       if W > 1 else torch.zeros_like(flow[:, 0, :, :])) # dx
        norm_flow_y = (flow[:, 1, :, :] / ((H - 1) / 2.0)
                       if H > 1 else torch.zeros_like(flow[:, 1, :, :])) # dy
        norm_flow = torch.stack((norm_flow_x, norm_flow_y), dim=3) # (B, H, W, 2)

        # Add normalized flow to the grid. grid_sample expects (x,y) order for flow.
        sampling_grid = grid + norm_flow # New sample locations

        # Sample from x using the new grid
        warped_x = F.grid_sample(
            x, sampling_grid, mode='bilinear', padding_mode='border',
            align_corners=True # Consistent with many optical flow papers
        )
        return warped_x

class MotionCompensationNetwork(nn.Module):
    def __init__(
        self,
        input_channels=3 + 2 + 3, # warped_ref (3), flow (2), ref_frame (3)
        output_channels=3, # Refinement map, typically 3 channels for RGB image
        base_channels=32,
        num_res_blocks=3,
    ):
        super().__init__()
        layers = []
        layers.append(ConvNormAct(input_channels, base_channels, kernel_size=5, padding=2)) # 'same'
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, output_channels, kernel_size=5, padding=2)) # 'same'
        layers.append(nn.Sigmoid()) # Output is a multiplicative mask (0 to 1)
        self.network = nn.Sequential(*layers)

    def forward(self, warped_ref, flow, ref_frame):
        """
        Refines the warped reference frame.
        Args:
            warped_ref (torch.Tensor): Warped reference frame (B, 3, H, W).
            flow (torch.Tensor): Optical flow used for warping (B, 2, H, W).
            ref_frame (torch.Tensor): Original reference frame (B, 3, H, W).
        Returns:
            torch.Tensor: Refined/compensated frame (B, 3, H, W).
        """
        if not (warped_ref.size() == ref_frame.size()
                and warped_ref.size()[-2:] == flow.size()[-2:]):
            raise ValueError("Input sizes mismatch in MotionCompensationNetwork. "
                             f"Warped: {warped_ref.shape}, Flow: {flow.shape}, Ref: {ref_frame.shape}")
        if flow.dim() != 4 or flow.shape[1] != 2:
            raise ValueError(f"Expected flow shape (B, 2, H, W), got {flow.shape}")

        mcn_input = torch.cat([warped_ref, flow, ref_frame], dim=1)
        refinement_map = self.network(mcn_input)
        refined_frame = warped_ref * refinement_map # Multiplicative refinement
        return refined_frame

# ==============================================================================
# VideoCodec Model
# ==============================================================================
class VideoCodec(nn.Module):
    def __init__(
        self,
        motion_latent_channels=128,
        residual_latent_channels=192,
        mcn_base_channels=32,
        encoder_base_channels=64,
        encoder_res_blocks=2,
        encoder_downsample_layers=3,
        decoder_res_blocks=2,
        decoder_upsample_layers=3,
    ):
        super().__init__()
        self._config = locals() # Store init args for potential saving/loading
        del self._config['self'], self._config['__class__']

        # Motion Path (Flow)
        self.motion_encoder = Encoder(
            input_channels=2, base_channels=encoder_base_channels // 2, # Flow has 2 channels
            latent_channels=motion_latent_channels,
            num_downsample_layers=encoder_downsample_layers, num_res_blocks=encoder_res_blocks
        )
        self.motion_entropy_bottleneck = EntropyBottleneck(motion_latent_channels)
        self.motion_decoder = Decoder(
            output_channels=2, base_channels=encoder_base_channels // 2, # Output is flow
            latent_channels=motion_latent_channels,
            num_upsample_layers=decoder_upsample_layers, num_res_blocks=decoder_res_blocks,
            final_activation=None # Flow values are not restricted to [0,1]
        )

        # Residual Path (Frame Residual)
        self.residual_encoder = Encoder(
            input_channels=3, base_channels=encoder_base_channels, # Residual is image-like
            latent_channels=residual_latent_channels,
            num_downsample_layers=encoder_downsample_layers, num_res_blocks=encoder_res_blocks
        )
        self.residual_entropy_bottleneck = EntropyBottleneck(residual_latent_channels)
        self.residual_decoder = Decoder(
            output_channels=3, base_channels=encoder_base_channels, # Output is image-like residual
            latent_channels=residual_latent_channels,
            num_upsample_layers=decoder_upsample_layers, num_res_blocks=decoder_res_blocks,
            final_activation=None # Residuals can be +/-; clamping happens after adding to MC frame
        )

        # Shared Components
        self.warping_layer = WarpingLayer()
        self.motion_compensation_net = MotionCompensationNetwork(
            input_channels=3 + 2 + 3, output_channels=3, # warped_ref, flow, ref_frame
            base_channels=mcn_base_channels, num_res_blocks=3 # Example values
        )

    @torch.no_grad()
    def init_entropy_bottleneck_buffers(self):
        """Initializes or updates entropy bottleneck buffers. Essential before compression/decompression."""
        print("DEBUG: Updating entropy bottleneck buffers with force=True...")
        try:
            # It's crucial that these are updated, especially after loading a checkpoint
            # or before the first compression/decompression pass.
            self.motion_entropy_bottleneck.update(force=True)
            print("  DEBUG: Motion EB updated.")
            self.residual_entropy_bottleneck.update(force=True)
            print("  DEBUG: Residual EB updated.")
            print("DEBUG: Entropy bottleneck buffers updated successfully.")
        except Exception as e:
            print(f"WARNING: Could not initialize/update entropy bottleneck buffers: {e}")
            traceback.print_exc() # Provide more details on the error

    def _compress_latent(self, bottleneck, latent_tensor, bottleneck_name=""):
        """Helper to compress a latent tensor using specified bottleneck."""
        # CompressAI's compress expects B=1 or a list of B=1 tensors.
        # If B > 1, handle it or raise error. For simplicity, assume B=1 or take first.
        if latent_tensor.shape[0] != 1:
            # This might indicate an issue upstream if B>1 is not intended for this stage.
            print(f"Warning: _compress_latent for {bottleneck_name} expects B=1, got {latent_tensor.shape[0]}. Taking first element.")
            latent_tensor = latent_tensor[0:1] # Process only the first item in the batch

        try:
            strings = bottleneck.compress(latent_tensor) # Returns a list of strings
            shape = latent_tensor.size()[-2:] # H, W of the latent space
            # Ensure we return a single bytestring if only one item was compressed
            actual_string = strings[0] if isinstance(strings, list) else strings
            return actual_string, tuple(shape)
        except RuntimeError as e:
            if "Entropy bottleneck must be updated" in str(e):
                print(f"RuntimeError: {bottleneck.__class__.__name__} ({bottleneck_name}) needs update. Forcing update NOW during compress.")
                bottleneck.update(force=True) # Attempt to fix on the fly
                strings = bottleneck.compress(latent_tensor)
                shape = latent_tensor.size()[-2:]
                actual_string = strings[0] if isinstance(strings, list) else strings
                return actual_string, tuple(shape)
            print(f"FATAL during compress {bottleneck_name}: {e}")
            raise e
        except Exception as e: # Catch other potential errors
            print(f"FATAL during compress {bottleneck_name}: {e}")
            raise e


    @torch.no_grad()
    def compress_simplified(self, flow_input_for_comp, residual_input_for_comp):
        """
        Simplified compression for one P-frame.
        Assumes flow and residual inputs are already at the target compression resolution.
        """
        self.eval() # Ensure model is in evaluation mode

        # Ensure inputs are B=1, C, H, W
        if flow_input_for_comp.shape[0] != 1:
            flow_input_for_comp = flow_input_for_comp[0:1]
        if residual_input_for_comp.shape[0] != 1:
            residual_input_for_comp = residual_input_for_comp[0:1]

        compressed_data = {}

        # Compress Motion (Flow)
        motion_latents = self.motion_encoder(flow_input_for_comp)
        compressed_data["motion"] = self._compress_latent(
            self.motion_entropy_bottleneck, motion_latents, "MotionEB"
        )

        # Compress Residual
        frame_residual_latents = self.residual_encoder(residual_input_for_comp)
        compressed_data["frame_residual"] = self._compress_latent(
            self.residual_entropy_bottleneck, frame_residual_latents, "ResidualEB"
        )

        return compressed_data

    @torch.no_grad()
    def decompress_frame_simplified(
        self,
        previous_frame_tensor, # B, C, H_orig, W_orig (original resolution)
        motion_compressed_data, # (bitstream, (latent_H_m, latent_W_m))
        residual_compressed_data, # (bitstream, (latent_H_r, latent_W_r))
        target_frame_hw # (H_orig, W_orig) for upscaling outputs
    ):
        """
        Simplified decompression for one P-frame.
        Handles upscaling of decoded flow and residual to target_frame_hw.
        """
        self.eval() # Ensure model is in evaluation mode

        # Ensure previous_frame_tensor is B=1
        if previous_frame_tensor.shape[0] != 1:
             previous_frame_tensor = previous_frame_tensor[0:1]

        # 1. Decompress Motion (Flow)
        motion_strings, motion_latent_shape_hw = motion_compressed_data
        residual_strings, residual_latent_shape_hw = residual_compressed_data
        
        # Ensure shapes are tuples of ints for decompress
        motion_latent_shape_hw = tuple(map(int, motion_latent_shape_hw))
        residual_latent_shape_hw = tuple(map(int, residual_latent_shape_hw))
        
        quantized_motion_latent = self.motion_entropy_bottleneck.decompress(
            [motion_strings], motion_latent_shape_hw # decompress expects a list of strings
        )
        flow_reconstructed_low_res = self.motion_decoder(quantized_motion_latent) # B, 2, H_m_comp, W_m_comp

        # Upscale flow to original frame resolution
        H_orig, W_orig = target_frame_hw
        flow_reconstructed_upscaled = resize_flow(
            flow_reconstructed_low_res, target_hw=(H_orig, W_orig)
        )
        if flow_reconstructed_upscaled is None or flow_reconstructed_upscaled.shape[-2:] != (H_orig, W_orig):
            # This check is important as resize_flow can return None on error
            raise RuntimeError(f"Flow upscaling failed or produced wrong shape. Expected {(H_orig, W_orig)}, got {flow_reconstructed_upscaled.shape[-2:] if flow_reconstructed_upscaled is not None else 'None'}")


        # 2. Warp previous frame using reconstructed upscaled flow
        warped_prev_frame = self.warping_layer(previous_frame_tensor, flow_reconstructed_upscaled)

        # 3. Motion Compensation (Refinement)
        # MCN takes warped_prev, upscaled_flow, and original prev_frame (all at original res)
        frame2_motion_compensated = self.motion_compensation_net(
            warped_prev_frame, flow_reconstructed_upscaled, previous_frame_tensor
        )

        # 4. Decompress Residual
        quantized_residual_latents = self.residual_entropy_bottleneck.decompress(
            [residual_strings], residual_latent_shape_hw
        )
        residual_reconstructed_low_res = self.residual_decoder(quantized_residual_latents) # B, 3, H_r_comp, W_r_comp

        # Upscale residual to original frame resolution if needed
        if residual_reconstructed_low_res.shape[-2:] == (H_orig, W_orig):
            residual_reconstructed_upscaled = residual_reconstructed_low_res
        elif H_orig > 0 and W_orig > 0: # Avoid resize if target is 0x0
            residual_reconstructed_upscaled = TF_tv.resize(
                residual_reconstructed_low_res, [H_orig, W_orig], # resize expects [H, W]
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
            )
        else: # If H_orig or W_orig is 0, don't attempt resize. This case should be rare.
             residual_reconstructed_upscaled = residual_reconstructed_low_res


        if H_orig > 0 and W_orig > 0 and residual_reconstructed_upscaled.shape[-2:] != (H_orig, W_orig):
             raise RuntimeError(f"Reconstructed residual shape {residual_reconstructed_upscaled.shape[-2:]} mismatch target {(H_orig, W_orig)}.")

        # 5. Combine MC frame with reconstructed residual
        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed_upscaled
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0) # Final clamp for image range

        return (
            frame2_reconstructed,
            flow_reconstructed_upscaled, # For debugging or potential use
            warped_prev_frame,           # For debugging
            frame2_motion_compensated,   # For debugging
            residual_reconstructed_upscaled # For debugging
        )


# ==============================================================================
# Configuration Class
# ==============================================================================
class CodecConfig:
    def __init__(self):
        # Common paths and settings
        self.codec_checkpoint_path: str = "./codec_checkpoints_2phase_visual/latest_checkpoint_3phase.pth.tar"
        self.gpu: int | None = 0 # GPU ID or None for CPU. -1 in args also maps to None.

        # Encoder specific
        self.input_file_path: str = './input.yuv' # or .mp4, .avi etc.
        self.output_rdvc_file: str = './compressed_video.rdvc'
        self.iframe_interval: int = 5 # How often to insert an I-Frame
        self.iframe_jpeg_quality: int = 70 # JPEG quality for I-Frames (1-100)
        self.flow_compress_height: int = 1080 # Target height for flow before compression
        self.residual_compress_height: int = 1080 # Target height for residual before compression
        
        # YUV specific inputs (only if input_file_path is .yuv)
        self.input_yuv_width: int | None = 1920
        self.input_yuv_height: int | None = 1080
        self.input_yuv_pixel_format: str = "yuv420p" # Currently only yuv420p supported by reader
        self.input_yuv_fps: float | None = 30.0

        # RAFT specific (for encoder)
        self.raft_backend: str = "auto" # 'auto', 'torchvision', 'local'
        self.raft_checkpoint_dir: str = r'./raft_checkpoints_amp_epe' # For finding local RAFT checkpoint if path not given
        self.raft_checkpoint_path: str | None = None # Specific path to a RAFT model (local or torchvision custom)
        self.raft_resize_height: int = 368 # RAFT input height
        self.raft_resize_width: int = 640  # RAFT input width
        self.raft_iters: int = 12 # RAFT iterations
        self.raft_dropout: float = 0.0 # Dropout for local RAFT instantiation
        self.raft_mixed_precision: bool = True # AMP for RAFT if on CUDA

        # Decoder specific
        self.input_rdvc_file: str = './compressed_video.rdvc'
        self.output_video_path_decode: str = './reconstructed_video.mp4'
        self.debug_frames_dir_decode: str = './debug_frames_decoder' # Directory for debug frames
        self.debug_frame_interval_decode: int = 10 # Save debug frames every N frames
        self.low_motion_replacement_threshold: float = 0 # Pixels. 0.0 or negative disables P-frame region replacement with I-frame.
        # Post-processing for decoder (added parameters)
        self.temporal_filter_alpha: float = 0 # For moving average filter

        # Model architecture (must match the loaded checkpoint)
        self.motion_latent_channels: int = 128
        self.residual_latent_channels: int = 192
        self.mcn_base_channels: int = 32
        self.encoder_base_channels: int = 64
        self.encoder_res_blocks: int = 2
        self.encoder_downsample_layers: int = 3
        self.decoder_res_blocks: int = 2
        self.decoder_upsample_layers: int = 3


    def update_from_args(self, args):
        """Updates config attributes from parsed command-line arguments if they exist."""
        if hasattr(args, 'gpu') and args.gpu is not None:
            if args.gpu == -1: # Explicit CPU request
                self.gpu = None
                self.raft_mixed_precision = False # AMP not for CPU
            else:
                self.gpu = args.gpu
        
        # If GPU is ultimately None (either by default or by arg), disable AMP
        if self.gpu is None:
            self.raft_mixed_precision = False
        
        if hasattr(args, 'raft_backend') and args.raft_backend is not None: # Check if provided
            self.raft_backend = args.raft_backend
        # Other args can be added here if needed
        if hasattr(args, 'temporal_filter_alpha') and args.temporal_filter_alpha is not None:
            self.temporal_filter_alpha = args.temporal_filter_alpha


# ==============================================================================
# Helper Functions
# ==============================================================================
def find_latest_checkpoint_file(checkpoint_dir, prefix='raft_epoch_', suffix='.pth'):
    """Finds the latest checkpoint file in a directory, preferring epoch number."""
    if not os.path.isdir(checkpoint_dir):
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return None

    checkpoints = list(Path(checkpoint_dir).glob(f'{prefix}*{suffix}'))
    # If specific prefix yields no results, try a more generic suffix search
    if not checkpoints:
        checkpoints = list(Path(checkpoint_dir).glob(f'*{suffix}')) # e.g. *.pth

    if not checkpoints:
        print(f"Warning: No checkpoint files matching pattern found in {checkpoint_dir}")
        return None

    latest_checkpoint, max_epoch, found_by_epoch = None, -1, False
    
    # Try to find by epoch number in filename (e.g., raft_epoch_100.pth)
    pattern_text = re.escape(prefix) + r'(\d+)' + (re.escape(suffix) if suffix else '')
    for ckpt_path in checkpoints:
        match = re.search(pattern_text, ckpt_path.name)
        if match:
            try:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_checkpoint = str(ckpt_path)
                    found_by_epoch = True
            except ValueError:
                pass # Filename part wasn't a number

    if found_by_epoch:
        print(f"Found latest checkpoint by epoch: {os.path.basename(latest_checkpoint)}")
        return latest_checkpoint

    # If not found by epoch, fall back to modification time
    print("Warning: Could not determine checkpoint by epoch number, using mtime.")
    latest_mtime = 0
    for ckpt_path in checkpoints:
        try:
            mtime = ckpt_path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_checkpoint = str(ckpt_path)
        except OSError as e:
            print(f"Warning: Could not get mtime for {ckpt_path}: {e}")

    if latest_checkpoint:
        print(f"Found latest checkpoint by mtime: {os.path.basename(latest_checkpoint)}")
    else:
        print(f"Warning: Could not find any valid checkpoint file in {checkpoint_dir}")
    return latest_checkpoint

def preprocess_frame_raft(frame_np_rgb, resize_shape_hw, device):
    """Preprocesses a single NumPy RGB frame for RAFT input."""
    try:
        # Convert to tensor, normalize to [0,1] implicitly by to_tensor if uint8
        tensor = TF_tv.to_tensor(frame_np_rgb) # C, H, W
        # Resize
        resized_tensor = TF_tv.resize(tensor, list(resize_shape_hw), antialias=True) # C, H_new, W_new
        # Add batch dimension and send to device
        return resized_tensor.unsqueeze(0).to(device) # B, C, H_new, W_new
    except Exception as e:
        print(f"Error preprocessing frame for RAFT: {e}"); return None

def preprocess_frame_codec(frame_np_rgb, device):
    """Preprocesses a single NumPy RGB frame for Codec input (typically original resolution)."""
    try:
        tensor = TF_tv.to_tensor(frame_np_rgb) # C, H, W
        return tensor.unsqueeze(0).to(device) # B, C, H, W
    except Exception as e:
        print(f"Error preprocessing frame for Codec: {e}"); return None


def resize_flow(flow_tensor, target_hw):
    """
    Resizes an optical flow tensor and scales its values.
    Args:
        flow_tensor (torch.Tensor): Flow tensor (B, 2, H_in, W_in).
        target_hw (tuple): Target (Height, Width).
    Returns:
        torch.Tensor or None: Resized and scaled flow tensor, or None on error.
    """
    if flow_tensor is None: return None
    B, C, H_in, W_in = flow_tensor.shape
    if C != 2: raise ValueError(f"Flow tensor must have 2 channels, got {C}")

    H_out, W_out = target_hw

    if (H_in, W_in) == (H_out, W_out):
        return flow_tensor # No resize needed

    # Handle zero dimensions to prevent division by zero or invalid resize ops
    if H_in == 0 or W_in == 0 : # Cannot resize from zero area
        # Return zero flow at target resolution if sensible, or handle error
        # print(f"Warning: Attempting to resize flow from zero dimensions ({H_in}x{W_in}). Returning zeros at target.")
        return torch.zeros(B, C, H_out, W_out, device=flow_tensor.device, dtype=flow_tensor.dtype)
    if H_out == 0 or W_out == 0: # Target is zero area
        return torch.zeros(B, C, H_out, W_out, device=flow_tensor.device, dtype=flow_tensor.dtype)


    try:
        # Interpolation mode for flow is typically bilinear. Antialias can be True or False.
        # Some practices use antialias=False for flow.
        flow_resized = TF_tv.resize(flow_tensor, [H_out, W_out],
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    antialias=False) # Common choice for flow

        # Scale flow values
        scale_w = float(W_out) / W_in if W_in > 0 else 1.0
        scale_h = float(H_out) / H_in if H_in > 0 else 1.0

        flow_scaled = torch.zeros_like(flow_resized)
        flow_scaled[:, 0, :, :] = flow_resized[:, 0, :, :] * scale_w # dx component
        flow_scaled[:, 1, :, :] = flow_resized[:, 1, :, :] * scale_h # dy component

        return flow_scaled
    except Exception as e:
        print(f"Error resizing flow from {H_in}x{W_in} to {H_out}x{W_out}: {e}")
        traceback.print_exc()
        return None


def load_model_checkpoint(checkpoint_path, model, model_name="Model", device=None, strict_load=False):
    """Loads a model checkpoint, handling potential 'module.' prefix and other common variations."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"ERROR: {model_name} checkpoint path invalid or file not found: '{checkpoint_path}'")
        return False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {model_name} checkpoint: {checkpoint_path} to device: {device}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
        state_dict = None
        
        # Try to extract state_dict from common checkpoint structures
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint and isinstance(checkpoint['model'], dict): state_dict = checkpoint['model'] # E.g., RAFT checkpoints
            else:
                # Heuristic: if keys look like state_dict keys, assume it's a raw state_dict
                is_likely_state_dict = all(
                    '.' in k or k.endswith(('_weight', '_bias', '_running_mean', '_running_var', '_num_batches_tracked'))
                    for k in checkpoint.keys()
                )
                if is_likely_state_dict:
                    state_dict = checkpoint
                else:
                    raise KeyError(f"Could not find state_dict in checkpoint. Keys: {list(checkpoint.keys())}")
        elif isinstance(checkpoint, nn.Module): # If entire model was saved
            state_dict = checkpoint.state_dict()
        else:
            raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("State_dict is empty or not a dictionary.")

        # Clean keys (e.g., remove 'module.', '_orig_mod.', 'model.')
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith('module.'): new_k = new_k[len('module.'):]
            if '_orig_mod.' in new_k: new_k = new_k.replace('_orig_mod.', '') # For FSDP or similar wrappers
            if new_k.startswith('model.'): new_k = new_k[len('model.'):] # Common in some saved RAFT models
            cleaned_state_dict[new_k] = v
        state_dict = cleaned_state_dict
        
        # Special handling for VideoCodec: ensure EB buffers are updated before and after load
        if isinstance(model, VideoCodec):
            try:
                # This is a defensive measure. update() should ideally be idempotent or handle uninitialized state.
                model.motion_entropy_bottleneck.update(force=True)
                model.residual_entropy_bottleneck.update(force=True)
            except Exception as e_pre_update:
                print(f"    WARNING: PRE-LOAD EB update failed (might be ok if first load): {e_pre_update}")

        # Load the state dict
        load_result = model.load_state_dict(state_dict, strict=strict_load)
        
        # For VideoCodec, call init_entropy_bottleneck_buffers again to ensure CDFs are correctly populated from loaded params
        if isinstance(model, VideoCodec):
            model.init_entropy_bottleneck_buffers() # This will print its own debug/warnings
            # Sanity check after load and update
            if model.motion_entropy_bottleneck._quantized_cdf is None or \
               model.residual_entropy_bottleneck._quantized_cdf is None:
                 print("    CRITICAL WARNING: CDFs are still None after POST-LOAD update! Compression/decompression will likely fail.")


        if not load_result.missing_keys and not load_result.unexpected_keys:
            print(f"  {model_name} loaded successfully (all keys matched).")
        else:
            print(f"  {model_name} loaded with mismatches (strict={strict_load}).")
            if load_result.missing_keys:
                # Filter out known benign missing keys (like EB buffers if model was just init'd)
                critical_missing = [k for k in load_result.missing_keys if not any(eb_buf_name in k for eb_buf_name in ["_offset", "_quantized_cdf", "_cdf_length"])]
                if critical_missing:
                    print(f"    WARNING: Missing Keys (potentially critical): {critical_missing}")
                # else:
                #     print(f"    INFO: Missing keys are related to entropy bottleneck buffers (expected if model was freshly initialized): {load_result.missing_keys}")

            if load_result.unexpected_keys:
                print(f"    WARNING: Unexpected Keys in checkpoint (model doesn't have them): {load_result.unexpected_keys}")
        
        model.to(device) # Move model to target device
        model.eval()     # Set to evaluation mode
        
        # Clean up to free memory
        del checkpoint, state_dict, cleaned_state_dict
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return True

    except FileNotFoundError:
        print(f"ERROR: File not found: {checkpoint_path}")
    except Exception as e:
        print(f"ERROR loading {model_name} checkpoint '{checkpoint_path}':")
        traceback.print_exc()
    return False


def load_image_as_tensor(image_path_or_bytes, device, is_bytes=False):
    """Loads an image from path or bytes into a BCHW tensor [0,1]."""
    try:
        if is_bytes:
            if not isinstance(image_path_or_bytes, bytes):
                raise TypeError("Expected bytes for image_path_or_bytes when is_bytes=True")
            img_pil = Image.open(io.BytesIO(image_path_or_bytes)).convert('RGB')
        else:
            image_path = Path(image_path_or_bytes)
            if not image_path.is_file():
                print(f"Error: Image file not found: {image_path}")
                return None
            img_pil = Image.open(image_path).convert('RGB')
        
        # To tensor (scales to [0,1] if PIL image is uint8) and add batch dim
        return TF_tv.to_tensor(img_pil).unsqueeze(0).to(device)
    except FileNotFoundError: # Should be caught by Path.is_file() for path case
        print(f"Error: Image file not found: {image_path_or_bytes}")
    except Exception as e:
        source_desc = "image bytes" if is_bytes else f"image {image_path_or_bytes}"
        print(f"Error loading {source_desc}: {e}")
    return None


def tensor_to_cv2_bgr(tensor_bchw_01):
    """Converts a BxCxHxW tensor (range [0,1]) to a list of OpenCV BGR images (uint8)."""
    if tensor_bchw_01 is None: return []
    if not torch.is_tensor(tensor_bchw_01):
        raise TypeError(f"Input not PyTorch Tensor: {type(tensor_bchw_01)}")
    
    # Handle 3D tensor (C,H,W) by adding batch dim
    if tensor_bchw_01.dim() == 3:
        tensor_bchw_01 = tensor_bchw_01.unsqueeze(0)
    
    # Validate shape (B, C, H, W) where C is 1 or 3
    if tensor_bchw_01.dim() != 4 or tensor_bchw_01.shape[1] not in [1, 3]:
        raise ValueError(f"Input tensor must be Bx1xHxW or Bx3xHxW, got {tensor_bchw_01.shape}")
    
    # If grayscale (1 channel), repeat to 3 channels for BGR conversion
    if tensor_bchw_01.shape[1] == 1:
        tensor_bchw_01 = tensor_bchw_01.repeat(1, 3, 1, 1) # B, 3, H, W
    
    # Clamp, scale to [0,255], convert to numpy, permute to HWC, and change color format
    tensor_bchw_01 = torch.clamp(tensor_bchw_01, 0.0, 1.0)
    images_np = []
    for i in range(tensor_bchw_01.shape[0]):
        img_hwc_rgb = tensor_bchw_01[i].detach().cpu().permute(1, 2, 0).numpy() # H, W, C (RGB)
        img_hwc_bgr_uint8 = (img_hwc_rgb * 255).astype(np.uint8)
        
        # OpenCV expects BGR if 3 channels
        if img_hwc_bgr_uint8.shape[2] == 3: # Should always be true after repeat
            img_hwc_bgr_uint8 = cv2.cvtColor(img_hwc_bgr_uint8, cv2.COLOR_RGB2BGR)
        images_np.append(img_hwc_bgr_uint8)
        
    return images_np # List of H, W, C (BGR) uint8 NumPy arrays

def save_tensor_as_image_vis(tensor_bchw, filepath: Path, drange=(0,1)):
    """Saves a BxCxHxW tensor as an image, normalizes from drange to [0,1] first."""
    if tensor_bchw is None:
        print(f"Warning: Attempted to save None tensor to {filepath}"); return
    try:
        min_val, max_val = drange
        # Normalize to [0,1] based on drange
        tensor_normalized = (tensor_bchw - min_val) / (max_val - min_val + 1e-6) # Add epsilon for stability

        img_np_list = tensor_to_cv2_bgr(tensor_normalized) # This handles B=1, C=1 or 3
        if not img_np_list:
            print(f"Warning: Conversion to CV2 format failed for {filepath}"); return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(filepath), img_np_list[0]): # Save first image in batch
             print(f"Warning: Failed to write image to {filepath}")
    except Exception as e:
        print(f"ERROR saving tensor to image {filepath}: {e}"); traceback.print_exc()

def visualize_flow_hsv(flow_tensor_b2hw, filepath: Path, clip_norm=None):
    """Visualizes a Bx2xHxW flow tensor using HSV color space and saves as image."""
    if flow_tensor_b2hw is None: return
    if flow_tensor_b2hw.shape[1] != 2: # Expects (B, 2, H, W)
        print(f"Warning: visualize_flow_hsv expects Bx2xHxW, got {flow_tensor_b2hw.shape}. Skipping."); return
    try:
        # Take the first flow field in the batch
        flow_np_hw2 = flow_tensor_b2hw[0].detach().cpu().numpy().transpose(1, 2, 0) # H, W, 2
        
        # Convert cartesian flow (dx, dy) to polar (magnitude, angle)
        mag, ang_rad = cv2.cartToPolar(flow_np_hw2[..., 0], flow_np_hw2[..., 1])
        
        # Create HSV image
        hsv = np.zeros((flow_np_hw2.shape[0], flow_np_hw2.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang_rad * 180 / np.pi / 2 # Hue from angle (0-179 for OpenCV)
        hsv[..., 1] = 255 # Saturation to max
        
        # Value from magnitude
        if clip_norm is not None: mag = np.clip(mag, 0, clip_norm) # Clip magnitude if specified
        if np.any(mag): # Avoid normalize error if mag is all zeros
            cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX) # Normalize mag to 0-255
        else:
            mag = np.zeros_like(mag) # Ensure mag is an array of zeros if all inputs were zero
        
        hsv[..., 2] = mag.astype(np.uint8)
        
        # Convert HSV to BGR for saving
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), bgr)
    except Exception as e:
        print(f"ERROR visualizing flow {filepath}: {e}"); traceback.print_exc()

# ==============================================================================
# YUV Frame Reader
# ==============================================================================
def read_yuv_frame_generator(filepath: str, width: int, height: int, pixel_format: str = "yuv420p"):
    """Generator to read frames from a YUV file and yield them as RGB NumPy arrays."""
    filepath_obj = Path(filepath)
    if not filepath_obj.is_file():
        print(f"Error: YUV input file not found: {filepath}")
        return # Stop generation

    if pixel_format == "yuv420p":
        if width % 2 != 0 or height % 2 != 0:
            print(f"Error: YUV420p requires width and height to be even. Got {width}x{height}.")
            return
        # Y plane, then U plane, then V plane. U and V are quarter size.
        y_size = width * height
        uv_plane_h, uv_plane_w = height // 2, width // 2
        uv_size = uv_plane_h * uv_plane_w
        frame_size_bytes = y_size + uv_size + uv_size # Total bytes per frame
        
        # OpenCV format for YUV I420 (YYYYYYYY UU VV) -> BGR
        cv_yuv_to_bgr_format_code = cv2.COLOR_YUV2BGR_I420
        # Reshape for cvtColor: height for Y + height for U + height for V (packed)
        # For I420, it's H_y + H_u + H_v = H + H/2 = 3/2 H
        cv_reshape_h = height + height // 2 # Total height of the packed YUV data
        cv_reshape_w = width # Width remains the same
    # elif pixel_format == "yuv422p": ... add other formats here
    else:
        print(f"Error: Pixel format '{pixel_format}' not currently supported for YUV reading.")
        return

    try:
        with open(filepath_obj, 'rb') as f:
            while True:
                frame_data_packed_bytes = f.read(frame_size_bytes)
                if not frame_data_packed_bytes: # End of file
                    break
                if len(frame_data_packed_bytes) < frame_size_bytes:
                    print(f"Warning: Incomplete final frame in YUV file ({len(frame_data_packed_bytes)} < {frame_size_bytes} bytes). Ignoring.")
                    break
                
                # Convert flat byte array to the shape cv2.cvtColor expects for YUV_I420
                yuv_mat_for_cv2 = np.frombuffer(frame_data_packed_bytes, dtype=np.uint8).reshape((cv_reshape_h, cv_reshape_w))
                
                # Convert YUV to BGR
                bgr_frame_np = cv2.cvtColor(yuv_mat_for_cv2, cv_yuv_to_bgr_format_code)
                # Convert BGR to RGB (common format for PyTorch/PIL)
                rgb_frame_np = cv2.cvtColor(bgr_frame_np, cv2.COLOR_BGR2RGB)
                
                yield rgb_frame_np
    except Exception as e:
        print(f"Error reading or processing YUV frame from {filepath}: {e}")
        traceback.print_exc()
        return # Stop generation on error

def _get_frame_source_details(config: CodecConfig):
    """Determines frame source (YUV or video file) and returns iterator and metadata."""
    input_p = Path(config.input_file_path)
    is_yuv_input = input_p.suffix.lower() == ".yuv"

    if is_yuv_input:
        print("Input type: YUV")
        # Validate YUV specific configurations from CodecConfig
        if not all([config.input_yuv_width, config.input_yuv_height, 
                    config.input_yuv_fps, config.input_yuv_pixel_format]):
            print("FATAL: YUV parameters (width, height, fps, pixel_format) are not fully configured in CodecConfig for .yuv input.")
            return None, 0, 0, 0.0, 0 # iterator, width, height, fps, total_frames

        frame_width = config.input_yuv_width
        frame_height = config.input_yuv_height
        original_fps = config.input_yuv_fps
        pixel_format = config.input_yuv_pixel_format

        # Calculate total frames for YUV
        try:
            bytes_per_frame = 0
            if pixel_format == "yuv420p": # Matches read_yuv_frame_generator logic
                if frame_width % 2 != 0 or frame_height % 2 != 0:
                    print(f"ERROR: YUV420p requires even width & height. Got {frame_width}x{frame_height}.")
                    return None, 0, 0, 0.0, 0
                bytes_per_frame = (frame_width * frame_height * 3) // 2
            # Add other pixel formats here if supported by read_yuv_frame_generator
            else:
                print(f"FATAL: Unsupported YUV pixel format for frame size calculation: {pixel_format}")
                return None, 0, 0, 0.0, 0
            
            if bytes_per_frame <= 0: # Should not happen if logic above is correct
                print(f"FATAL: Invalid bytes_per_frame ({bytes_per_frame}) for YUV.")
                return None, 0, 0, 0.0, 0

            if not input_p.is_file(): # Should have been checked before, but double-check
                print(f"FATAL: YUV input file not found: {input_p}")
                return None, 0, 0, 0.0, 0

            file_size = input_p.stat().st_size
            total_frames = file_size // bytes_per_frame
            if file_size % bytes_per_frame != 0:
                print(f"Warning: YUV file size ({file_size}) is not an exact multiple of frame size ({bytes_per_frame}). May truncate last partial frame.")
        except Exception as e:
            print(f"FATAL: Could not determine total frames for YUV input: {e}")
            return None, 0, 0, 0.0, 0
        
        frame_iterator = read_yuv_frame_generator(
            str(input_p), frame_width, frame_height, pixel_format
        )
        return frame_iterator, frame_width, frame_height, original_fps, total_frames
    
    else: # Standard video file (mp4, avi, etc.)
        print(f"Input type: Standard Video File ({input_p.suffix})")
        cap = cv2.VideoCapture(str(input_p))
        if not cap.isOpened():
            print(f"FATAL: Cannot open video file {input_p}")
            return None, 0, 0, 0.0, 0
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Can be unreliable for some formats/streams
        
        if total_frames <= 0: # If header doesn't provide it or it's a stream
             print("Warning: Total frame count from video header is unavailable or zero. Will process until stream ends.")
             total_frames = 0 # Indicate unknown count for progress bar

        def video_capture_generator(cv_capture_object):
            try:
                while True:
                    ret, frame_bgr = cv_capture_object.read()
                    if not ret: # End of video or error
                        break
                    yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Yield RGB
            finally:
                cv_capture_object.release() # Ensure release
        
        return video_capture_generator(cap), frame_width, frame_height, original_fps, total_frames


# ==============================================================================
# Main Encoding Function
# ==============================================================================
def encode_video_main(config: CodecConfig):
    print("\n" + "=" * 60 + "\n--- Starting Video Encoding ---\n" + "=" * 60)
    input_file_p = Path(config.input_file_path)
    output_rdvc_file_p = Path(config.output_rdvc_file)

    print(f"  Input File: {input_file_p}")
    print(f"  Output RDVC File: {output_rdvc_file_p}")
    print(f"  Codec Checkpoint: {config.codec_checkpoint_path}")
    print(f"  I-Frame Format: JPEG (Quality: {config.iframe_jpeg_quality})")
    print(f"  Requested RAFT Backend: {config.raft_backend}")


    if not input_file_p.is_file():
        print(f"FATAL: Input file not found: {input_file_p}"); sys.exit(1)
    try:
        output_rdvc_file_p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"FATAL: Could not create parent directory for RDVC file {output_rdvc_file_p}: {e}")
        sys.exit(1)

    # --- Device Setup ---
    if config.gpu is not None and config.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu}")
        try:
            torch.cuda.set_device(device) # Set default CUDA device for this process
            print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        except Exception as e:
            print(f"CUDA device error: {e}, attempting default CUDA device if available, else CPU.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda": print(f"Switched to default CUDA device: {device}")
    else:
        device = torch.device("cpu")
        config.raft_mixed_precision = False # AMP is only for CUDA
        print(f"Using device: CPU")
    
    raft_amp_enabled = config.raft_mixed_precision and device.type == 'cuda'
    print(f"RAFT Mixed Precision: {'ENABLED' if raft_amp_enabled else 'DISABLED'}")


    print("\n--- Loading Models ---")
    
    # --- Determine RAFT Implementation ---
    selected_raft_impl = None
    selected_raft_needs_args = None # True for local core RAFT, False for torchvision
    selected_raft_source_name = "None"
    
    if config.raft_backend == "torchvision":
        if _TORCHVISION_RAFT_NEW_AVAILABLE:
            selected_raft_impl = _TV_RAFT_NEW_IMPL; selected_raft_needs_args = False
            selected_raft_source_name = "torchvision_new (forced)"
        elif _TORCHVISION_RAFT_OLD_AVAILABLE:
            selected_raft_impl = _TV_RAFT_OLD_IMPL; selected_raft_needs_args = False
            selected_raft_source_name = "torchvision_old (forced)"
        else: print("FATAL: RAFT backend 'torchvision' selected, but no torchvision RAFT implementation is available."); sys.exit(1)
    elif config.raft_backend == "local":
        if _LOCAL_RAFT_CORE_AVAILABLE:
            selected_raft_impl = _LOCAL_RAFT_CORE_IMPL; selected_raft_needs_args = True
            selected_raft_source_name = "local_core (forced)"
        else: print("FATAL: RAFT backend 'local' selected, but local 'core' RAFT implementation is not available."); sys.exit(1)
    elif config.raft_backend == "auto":
        if _TORCHVISION_RAFT_NEW_AVAILABLE:
            selected_raft_impl = _TV_RAFT_NEW_IMPL; selected_raft_needs_args = False
            selected_raft_source_name = "torchvision_new (auto)"
        elif _TORCHVISION_RAFT_OLD_AVAILABLE:
            selected_raft_impl = _TV_RAFT_OLD_IMPL; selected_raft_needs_args = False
            selected_raft_source_name = "torchvision_old (auto)"
        elif _LOCAL_RAFT_CORE_AVAILABLE:
            selected_raft_impl = _LOCAL_RAFT_CORE_IMPL; selected_raft_needs_args = True
            selected_raft_source_name = "local_core (auto)"
        else: print("FATAL: RAFT backend 'auto', but no RAFT implementation available (startup check failed)."); sys.exit(1)
    else: print(f"FATAL: Invalid raft_backend specified in config: {config.raft_backend}"); sys.exit(1)

    print(f"Attempting to use RAFT source: {selected_raft_source_name}")
    actual_raft_checkpoint_description = "N/A" # For metadata
    raft_model = None

    # --- Instantiate and Load RAFT Model ---
    try:
        if selected_raft_needs_args: # Local RAFT from 'core'
            print("Instantiating Local RAFT model...")
            class RaftArgsDummy: pass # Dummy class for RAFT args if needed
            raft_args = RaftArgsDummy()
            raft_args.small = False # Assuming 'large' RAFT model variant
            raft_args.dropout = config.raft_dropout
            # Some local RAFT impls might expect these attributes
            if hasattr(raft_args, 'mixed_precision'): raft_args.mixed_precision = raft_amp_enabled
            if hasattr(raft_args, 'alternate_corr'): raft_args.alternate_corr = False # Common default
            
            raft_model = selected_raft_impl(raft_args)
            
            local_raft_checkpoint_path = config.raft_checkpoint_path # Use if specified
            if not local_raft_checkpoint_path: # Else, try to find in dir
                local_raft_checkpoint_path = find_latest_checkpoint_file(config.raft_checkpoint_dir)
            
            if not local_raft_checkpoint_path or not os.path.isfile(local_raft_checkpoint_path):
                raise RuntimeError(f"Local RAFT ('{selected_raft_source_name}') selected, but no valid checkpoint was specified "
                                   f"via 'raft_checkpoint_path' ('{config.raft_checkpoint_path}') or found in "
                                   f"'raft_checkpoint_dir' ('{config.raft_checkpoint_dir}').")
            
            if not load_model_checkpoint(local_raft_checkpoint_path, raft_model, f"RAFT ({selected_raft_source_name})", device, strict_load=True): # Strict for main models
                raise RuntimeError(f"Failed to load checkpoint for Local RAFT model from {local_raft_checkpoint_path}.")
            actual_raft_checkpoint_description = str(Path(local_raft_checkpoint_path).resolve())
            print(f"Local RAFT model loaded from {actual_raft_checkpoint_description}.")

        else: # Torchvision RAFT
            print(f"Instantiating Torchvision RAFT model ({selected_raft_source_name})...")
            # If a specific RAFT checkpoint is given for torchvision, it's for custom fine-tuning.
            if config.raft_checkpoint_path and os.path.isfile(config.raft_checkpoint_path):
                print(f"  Attempting to load specified checkpoint for Torchvision RAFT: {config.raft_checkpoint_path}")
                if selected_raft_source_name.startswith("torchvision_new"):
                    raft_model = selected_raft_impl(weights=None) # Instantiate without default weights first
                else: # Old torchvision API
                    raft_model = selected_raft_impl(pretrained=False) # Instantiate without default weights
                
                if not load_model_checkpoint(config.raft_checkpoint_path, raft_model, f"RAFT ({selected_raft_source_name} with custom ckpt)", device, strict_load=True):
                    raise RuntimeError(f"Failed to load custom checkpoint {config.raft_checkpoint_path} for Torchvision RAFT model.")
                actual_raft_checkpoint_description = str(Path(config.raft_checkpoint_path).resolve())
                print(f"Torchvision RAFT model loaded with custom checkpoint from {actual_raft_checkpoint_description}.")
            
            # If no specific checkpoint, use torchvision's default pre-trained weights
            else:
                if selected_raft_source_name.startswith("torchvision_new") and _TV_RAFT_NEW_WEIGHTS_ENUM:
                    print(f"  Using Torchvision RAFT (new API) with default weights: {_TV_RAFT_NEW_WEIGHTS_ENUM.DEFAULT}")
                    raft_model = selected_raft_impl(weights=_TV_RAFT_NEW_WEIGHTS_ENUM.DEFAULT)
                    actual_raft_checkpoint_description = f"Torchvision Default Weights ({_TV_RAFT_NEW_WEIGHTS_ENUM.DEFAULT}, {selected_raft_source_name})"
                elif selected_raft_source_name.startswith("torchvision_old") and _TV_RAFT_OLD_IMPL:
                    print("  Using Torchvision RAFT (old API), attempting with pretrained=True.")
                    try: raft_model = selected_raft_impl(pretrained=True)
                    except TypeError: # pretrained not an arg (should not happen for official old raft_large)
                         raft_model = selected_raft_impl(); print("  Instantiated without explicit pretrained=True.")
                    actual_raft_checkpoint_description = f"Torchvision Default Weights (pretrained=True or implicit, {selected_raft_source_name})"
                else: # Fallback if logic is flawed, should not be reached
                    raise RuntimeError(f"Cannot determine how to instantiate Torchvision RAFT with default weights for: {selected_raft_source_name}")
                print("  Torchvision RAFT model instantiated, using its default/pretrained weights.")

            if raft_model is None: # Should be caught by specific errors above
                 raise RuntimeError("Failed to instantiate Torchvision RAFT model for unknown reasons.")
            
            raft_model.to(device) # Move to device
            raft_model.eval()     # Set to eval mode
            print("Torchvision RAFT model prepared and moved to device.")
        
        # --- Load Video Codec Model ---
        print("Instantiating VideoCodec model (parameters should align with training)...")
        model_codec = VideoCodec( # Use parameters from config
            motion_latent_channels=config.motion_latent_channels,
            residual_latent_channels=config.residual_latent_channels,
            mcn_base_channels=config.mcn_base_channels,
            encoder_base_channels=config.encoder_base_channels,
            encoder_res_blocks=config.encoder_res_blocks,
            encoder_downsample_layers=config.encoder_downsample_layers,
            decoder_res_blocks=config.decoder_res_blocks,
            decoder_upsample_layers=config.decoder_upsample_layers,
        )
        # strict_load=False for codec allows flexibility if some minor parts (like EB buffers initially) don't match.
        # EB buffers are re-initialized by init_entropy_bottleneck_buffers post-load anyway.
        if not load_model_checkpoint(config.codec_checkpoint_path, model_codec, "VideoCodec", device, strict_load=False): 
            raise RuntimeError(f"Failed to load VideoCodec checkpoint: {config.codec_checkpoint_path}")
        print("Video Codec model loaded.")

    except Exception as e:
        print(f"FATAL ERROR loading models: {e}"); traceback.print_exc(); sys.exit(1)


    print("\n--- Processing Video for Encoding ---")
    # Get frame source (iterator) and video properties
    frame_source_iter, frame_width, frame_height, original_fps, total_frames_from_source = \
        _get_frame_source_details(config)

    if frame_source_iter is None: # Error already printed by _get_frame_source_details
        sys.exit(1)
    
    print(f"Input Video Details: {frame_width}x{frame_height} @ {original_fps:.2f} FPS, "
          f"Frames: {total_frames_from_source if total_frames_from_source > 0 else 'Unknown/Streaming'}")
    if total_frames_from_source == 0 and not input_file_p.suffix.lower() == ".yuv": 
        print("Warning: Video source reported 0 frames. Will process until stream ends or error.")
    elif total_frames_from_source == 0 and input_file_p.suffix.lower() == ".yuv": # YUV should have frame count
         sys.exit("Input YUV file appears to have 0 frames based on size and dimensions. Cannot proceed.")

    # --- Determine Compression Resolutions ---
    aspect_ratio = frame_width / frame_height if frame_height > 0 else 1.0 # Avoid div by zero
    
    # Flow compression resolution (maintain aspect ratio)
    target_flow_h_comp = config.flow_compress_height
    target_flow_w_comp = int(round(target_flow_h_comp * aspect_ratio))
    # Ensure even dimensions if needed by downstream ops (e.g., some conv layers)
    if target_flow_w_comp % 2 != 0: target_flow_w_comp +=1 
    if target_flow_h_comp % 2 != 0: target_flow_h_comp +=1 
    flow_compress_resolution_wh_print = [target_flow_w_comp, target_flow_h_comp] # For metadata (W,H)
    flow_compress_resolution_hw_tuple = (target_flow_h_comp, target_flow_w_comp) # For resize ops (H,W)
    print(f"Flow will be compressed at resolution: {flow_compress_resolution_wh_print[0]}x{flow_compress_resolution_wh_print[1]}")

    # Residual compression resolution
    target_residual_h_comp = config.residual_compress_height
    target_residual_w_comp = int(round(target_residual_h_comp * aspect_ratio))
    if target_residual_w_comp % 2 != 0: target_residual_w_comp += 1
    if target_residual_h_comp % 2 != 0: target_residual_h_comp += 1
    residual_compress_resolution_wh_print = [target_residual_w_comp, target_residual_h_comp]
    residual_compress_resolution_hw_tuple = (target_residual_h_comp, target_residual_w_comp)
    print(f"Residual will be compressed at resolution: {residual_compress_resolution_wh_print[0]}x{residual_compress_resolution_wh_print[1]}")
    
    # RAFT input resolution (fixed, from config)
    raft_input_size_hw = (config.raft_resize_height, config.raft_resize_width)

    # --- Encoding Loop ---
    frames_processed_count = 0
    previous_frame_np_rgb = None
    previous_frame_codec_tensor = None # Store previous frame as tensor for codec input
    total_pframe_payload_bytes = 0 # For metadata stats
    
    try:
        # Open RDVC file for writing in binary mode
        with open(output_rdvc_file_p, 'wb') as rdvc_file:
            # Buffer for frame data (metadata written at the end)
            frame_data_buffer = io.BytesIO() # In-memory buffer for frame data
            temp_total_pframe_payload_bytes = 0 # Accumulator for P-frame bytes

            # Progress bar setup
            pbar_total = total_frames_from_source if total_frames_from_source > 0 else None
            pbar = tqdm(total=pbar_total, unit="frame", desc="Encoding to RDVC")
            
            for current_frame_rgb_np in frame_source_iter: # Iterates RGB NumPy frames
                if current_frame_rgb_np is None: # Should ideally be caught by generator
                    print(f"Warning: Frame source returned None at frame index {frames_processed_count}. Stopping.")
                    break

                is_iframe = (frames_processed_count % config.iframe_interval == 0) or \
                              (previous_frame_np_rgb is None) # First frame is always I-Frame

                # Write common frame header to buffer
                frame_data_buffer.write(RDVC_FRAME_MARKER)
                frame_data_buffer.write(struct.pack(UINT32_FORMAT, frames_processed_count)) # Frame index

                if is_iframe:
                    pbar.set_postfix_str("I-Frame (JPEG)")
                    frame_data_buffer.write(b'I') # Frame type: I-Frame
                    
                    # Convert NumPy RGB to PIL Image for JPEG compression
                    img_pil = Image.fromarray(current_frame_rgb_np)
                    iframe_bytes_io = io.BytesIO() # In-memory buffer for JPEG bytes
                    img_pil.save(iframe_bytes_io, format="JPEG", quality=config.iframe_jpeg_quality)
                    iframe_data_bytes = iframe_bytes_io.getvalue()
                    
                    # I-Frame payload: extension string + image data
                    iframe_ext_bytes = ".jpg".encode('utf-8') # Store extension for potential type change
                    iframe_content_payload = struct.pack(UINT8_FORMAT, len(iframe_ext_bytes)) + \
                                             iframe_ext_bytes + \
                                             iframe_data_bytes
                    
                    frame_data_buffer.write(struct.pack(UINT64_FORMAT, len(iframe_content_payload))) # Payload length
                    frame_data_buffer.write(iframe_content_payload) # Actual payload
                    
                    # Update reference for next P-Frame
                    previous_frame_np_rgb = current_frame_rgb_np.copy() # Keep NumPy for RAFT
                    previous_frame_codec_tensor = preprocess_frame_codec(previous_frame_np_rgb, device)
                    if previous_frame_codec_tensor is None: 
                        print(f"ERROR: Preprocessing I-Frame {frames_processed_count} for codec failed. Stopping."); break
                
                else: # P-Frame processing
                    pbar.set_postfix_str("P-Frame")
                    frame_data_buffer.write(b'P') # Frame type: P-Frame
                    try:
                        # 1. Preprocess frames for RAFT (at RAFT's required input size)
                        img1_raft = preprocess_frame_raft(previous_frame_np_rgb, raft_input_size_hw, device)
                        img2_raft = preprocess_frame_raft(current_frame_rgb_np, raft_input_size_hw, device)
                        if img1_raft is None or img2_raft is None: raise RuntimeError("RAFT preprocessing failed.")
                        
                        # 2. Estimate Optical Flow with RAFT
                        with torch.no_grad(), autocast(device_type=device.type, enabled=raft_amp_enabled):
                            if selected_raft_needs_args: # Local RAFT from 'core'
                                flow_preds = raft_model(img1_raft, img2_raft, iters=config.raft_iters, test_mode=True)
                            else: # Torchvision RAFT (new or old API)
                                # Torchvision RAFT API for num_flow_updates might vary slightly if old version is very old
                                # The current torchvision.models.optical_flow.raft_large takes it as a direct arg.
                                flow_preds = raft_model(img1_raft, img2_raft, num_flow_updates=config.raft_iters)
                            # RAFT typically returns a list of flows (intermediate updates); use the last one.
                            flow_at_raft_res = flow_preds[-1] if isinstance(flow_preds, list) else flow_preds
                        
                        # 3. Resize flow to original frame resolution
                        flow12_orig_res = resize_flow(flow_at_raft_res, (frame_height, frame_width))
                        if flow12_orig_res is None: raise RuntimeError("Flow upscaling to original resolution failed.")
                        
                        # 4. Prepare current frame for codec (original resolution)
                        current_frame_codec_tensor = preprocess_frame_codec(current_frame_rgb_np, device)
                        if current_frame_codec_tensor is None:
                            raise RuntimeError("Codec preprocessing failed for current P-frame.")
                        
                        # 5. Calculate Motion-Compensated Prediction and Residual (at original resolution)
                        with torch.no_grad(): # Ensure no gradients for these operations
                            warped_prev = model_codec.warping_layer(previous_frame_codec_tensor, flow12_orig_res)
                            frame2_mc_pred = model_codec.motion_compensation_net(warped_prev, flow12_orig_res, previous_frame_codec_tensor)
                            # Residual is Current - Prediction
                            residual_computed_orig_res = current_frame_codec_tensor - frame2_mc_pred

                        # 6. Resize residual and flow to their respective compression resolutions
                        residual_for_comp = TF_tv.resize(
                            residual_computed_orig_res,
                            list(residual_compress_resolution_hw_tuple), # [H,W]
                            interpolation=transforms.InterpolationMode.BILINEAR, # Or area for downsampling
                            antialias=True
                        )
                        
                        flow12_for_comp = resize_flow(flow12_orig_res, flow_compress_resolution_hw_tuple)
                        if flow12_for_comp is None: raise RuntimeError("Flow downscaling for compression failed.")

                        # 7. Compress flow and residual using the VideoCodec model
                        #    compress_simplified expects B=1 inputs
                        compressed_data = model_codec.compress_simplified(flow12_for_comp, residual_for_comp)
                        
                        motion_bs, motion_shape_hw = compressed_data["motion"] # (bitstream, (H,W) of latent)
                        residual_bs, residual_shape_hw = compressed_data["frame_residual"]

                        temp_total_pframe_payload_bytes += len(motion_bs) + len(residual_bs)

                        # P-Frame payload structure:
                        # motion_latent_H (int32), motion_latent_W (int32), motion_bs_len (uint32), motion_bs (bytes)
                        # residual_latent_H (int32), residual_latent_W (int32), residual_bs_len (uint32), residual_bs (bytes)
                        pframe_content_payload = struct.pack(INT32_FORMAT, motion_shape_hw[0]) + \
                                                 struct.pack(INT32_FORMAT, motion_shape_hw[1]) + \
                                                 struct.pack(UINT32_FORMAT, len(motion_bs)) + \
                                                 motion_bs + \
                                                 struct.pack(INT32_FORMAT, residual_shape_hw[0]) + \
                                                 struct.pack(INT32_FORMAT, residual_shape_hw[1]) + \
                                                 struct.pack(UINT32_FORMAT, len(residual_bs)) + \
                                                 residual_bs
                        
                        frame_data_buffer.write(struct.pack(UINT64_FORMAT, len(pframe_content_payload))) # Payload length
                        frame_data_buffer.write(pframe_content_payload) # Actual payload
                        
                        # Update reference for next P-Frame
                        previous_frame_np_rgb = current_frame_rgb_np.copy()
                        previous_frame_codec_tensor = current_frame_codec_tensor.detach().clone() # Use the actual current frame as next prev

                    except Exception as e_pframe:
                        print(f"\nERROR processing P-Frame {frames_processed_count}: {e_pframe}")
                        traceback.print_exc()
                        # Attempt to recover by making next frame an I-Frame?
                        # For now, critical error, stop or mark previous_frame as None to force next as I-Frame.
                        previous_frame_np_rgb, previous_frame_codec_tensor = None, None # Force next to be I-frame if loop continues
                        # Potentially break here or implement more robust error handling for P-frames.
                        # For simplicity in this script, an error in P-frame processing might lead to inconsistent stream.
                
                frames_processed_count += 1
                if pbar_total is not None or frames_processed_count % 10 == 0 : # Avoid too frequent updates if total is unknown
                    pbar.update(1)
            
            pbar.close() # Close progress bar

            total_pframe_payload_bytes = temp_total_pframe_payload_bytes # Finalize count

            # --- Write Metadata and Frame Data to RDVC File ---
            # Metadata structure
            overall_metadata = {
                "rdvc_version": "1.0",
                "input_video_filename": input_file_p.name,
                "output_rdvc_filename": output_rdvc_file_p.name,
                "original_dimensions_wh": [frame_width, frame_height], # W, H
                "flow_compression_resolution_wh": flow_compress_resolution_wh_print, # W, H
                "residual_compression_resolution_wh": residual_compress_resolution_wh_print, # W, H
                "original_fps": original_fps, 
                "total_frames_processed": frames_processed_count,
                "codec_checkpoint_filename": Path(config.codec_checkpoint_path).name,
                "raft_source_used": selected_raft_source_name,
                "raft_checkpoint_info": actual_raft_checkpoint_description,
                "iframe_interval": config.iframe_interval,
                "iframe_format": "JPEG", # Currently hardcoded
                "iframe_jpeg_quality": config.iframe_jpeg_quality,
                "total_pframe_payload_bytes": total_pframe_payload_bytes, # Sum of motion_bs + residual_bs lengths
                # Store relevant parts of encoder config for reproducibility/debugging
                "encoder_config_summary": {
                    "input_file_path": str(input_file_p), # Original full path might be too specific
                    "flow_compress_height": config.flow_compress_height,
                    "residual_compress_height": config.residual_compress_height,
                    "iframe_interval": config.iframe_interval,
                    "iframe_jpeg_quality": config.iframe_jpeg_quality,
                    "raft_backend_requested": config.raft_backend,
                    # Add other key config items if useful
                }
            }
            # Add YUV specific config to metadata if applicable
            if input_file_p.suffix.lower() == ".yuv":
                overall_metadata["encoder_config_summary"].update({
                    "input_yuv_width": config.input_yuv_width,
                    "input_yuv_height": config.input_yuv_height,
                    "input_yuv_pixel_format": config.input_yuv_pixel_format,
                    "input_yuv_fps": config.input_yuv_fps,
                })

            metadata_json_bytes = json.dumps(overall_metadata, indent=4).encode('utf-8')
            
            # Write metadata block: Marker, Length, JSON_Bytes
            rdvc_file.write(RDVC_METADATA_MARKER)
            rdvc_file.write(struct.pack(UINT32_FORMAT, len(metadata_json_bytes)))
            rdvc_file.write(metadata_json_bytes)

            # Write all buffered frame data
            frame_data_buffer.seek(0) # Rewind buffer to the beginning
            rdvc_file.write(frame_data_buffer.read()) # Write its content to file

            # Write EOF marker
            rdvc_file.write(RDVC_EOF_MARKER)

            print(f"RDVC file successfully saved to {output_rdvc_file_p}")

    except Exception as e:
        print(f"An UNEXPECTED ERROR occurred during the encoding to RDVC at/after frame {frames_processed_count}: {e}")
        traceback.print_exc()
    finally:
        if 'pbar' in locals() and pbar: pbar.close() # Ensure pbar is closed
        if 'frame_data_buffer' in locals() and frame_data_buffer: frame_data_buffer.close()
        print("\n--- Video Encoding to RDVC Finished ---")

    print("\n" + "=" * 60 + "\n--- Encoding Complete ---\n" + "=" * 60)

# ==============================================================================
# Helper Function for Histogram Matching (Decoder Post-processing)
# ==============================================================================
# Helper Function for Histogram Matching (Decoder Post-processing)
# ==============================================================================
def _match_histograms_cv(source_tensor_bchw, reference_tensor_bchw, device):
    """
    Matches histogram of source_tensor to reference_tensor using skimage.exposure.match_histograms
    by matching Y, Cr, Cb channels in YCrCb space (color space conversion still uses OpenCV).
    Assumes B=1 for both tensors.
    Args:
        source_tensor_bchw (torch.Tensor): BxCxHxW, range [0,1], on device.
        reference_tensor_bchw (torch.Tensor): BxCxHxW, range [0,1], on device.
        device (torch.device): Device for the output tensor.
    Returns:
        torch.Tensor: Matched BxCxHxW tensor, range [0,1], on device.
    """
    try:
        import skimage
        print(f"!!! Using skimage.exposure.match_histograms (skimage version: {skimage.__version__}) for matching. OpenCV for color conversion. !!!")
    except ImportError:
        print(f"!!! skimage not found, but _match_histograms_cv was called. This will fail. !!!")
        # Optionally, you could fall back to the original frame or raise an error
        return source_tensor_bchw # Fallback if skimage is missing after all

    if source_tensor_bchw.shape[0] != 1 or reference_tensor_bchw.shape[0] != 1:
        raise ValueError("Histogram matching currently supports B=1 only for source and reference.")
    if source_tensor_bchw.shape[1] != 3 or reference_tensor_bchw.shape[1] != 3:
        raise ValueError("Histogram matching requires 3-channel (RGB) tensors.")

    source_cv_bgr_list = tensor_to_cv2_bgr(source_tensor_bchw)
    ref_cv_bgr_list = tensor_to_cv2_bgr(reference_tensor_bchw)

    if not source_cv_bgr_list or not ref_cv_bgr_list:
        raise RuntimeError("Conversion to CV2 BGR failed for histogram matching input.")

    source_cv_bgr = source_cv_bgr_list[0]
    ref_cv_bgr = ref_cv_bgr_list[0]
    # print(f"DEBUG: source_cv_bgr.dtype: {source_cv_bgr.dtype}")


    source_ycrcb = cv2.cvtColor(source_cv_bgr, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb = cv2.cvtColor(ref_cv_bgr, cv2.COLOR_BGR2YCrCb)
    # print(f"DEBUG: source_ycrcb.dtype: {source_ycrcb.dtype}")


    source_y, source_cr, source_cb = cv2.split(source_ycrcb)
    ref_y, ref_cr, ref_cb = cv2.split(ref_ycrcb)
    # print(f"DEBUG: source_y.dtype: {source_y.dtype}")


    # Match histograms for each channel using skimage.exposure.match_histograms
    # These might return float64 arrays in some skimage versions/cases
    matched_y_processed = skimage_match_histograms(source_y, ref_y)
    matched_cr_processed = skimage_match_histograms(source_cr, ref_cr)
    matched_cb_processed = skimage_match_histograms(source_cb, ref_cb)
    # print(f"DEBUG: matched_y_processed.dtype after skimage: {matched_y_processed.dtype}")


    # Explicitly clip to [0, 255] and cast to uint8 to ensure compatibility with OpenCV
    # This handles cases where skimage_match_histograms might return float or values outside uint8 range.
    matched_y = np.clip(matched_y_processed, 0, 255).astype(np.uint8)
    matched_cr = np.clip(matched_cr_processed, 0, 255).astype(np.uint8)
    matched_cb = np.clip(matched_cb_processed, 0, 255).astype(np.uint8)
    # print(f"DEBUG: matched_y.dtype after clip/cast: {matched_y.dtype}")

    matched_ycrcb = cv2.merge([matched_y, matched_cr, matched_cb])
    # print(f"DEBUG: matched_ycrcb.dtype after merge: {matched_ycrcb.dtype}")


    # Convert back to BGR (using OpenCV for this)
    matched_bgr = cv2.cvtColor(matched_ycrcb, cv2.COLOR_YCrCb2BGR) # This line was erroring
    
    matched_rgb_np = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)
    
    matched_tensor_chw = TF_tv.to_tensor(matched_rgb_np)
    
    return matched_tensor_chw.unsqueeze(0).to(device)
# ==============================================================================
# Main Decoding Function
# ==============================================================================
def decode_video_main(config: CodecConfig):
    print("\n" + "=" * 60 + "\n--- Starting Video Decoding from RDVC ---\n" + "=" * 60)
    input_rdvc_file_p = Path(config.input_rdvc_file)
    output_video_file_path = Path(config.output_video_path_decode)
    debug_output_path_decode = Path(config.debug_frames_dir_decode) if config.debug_frames_dir_decode else None

    print(f"  Input RDVC File: {input_rdvc_file_p}")
    print(f"  Output Video Path: {output_video_file_path}")
    if config.low_motion_replacement_threshold > 0:
        print(f"  Low Motion Replacement: ENABLED (threshold: {config.low_motion_replacement_threshold} px)")
    else:
        print(f"  Low Motion Replacement: DISABLED")
    print(f"  P-Frame Histogram Matching: ENABLED (to previous post-processed frame)")
    print(f"  Temporal Low-Pass Filter: ENABLED (alpha: {config.temporal_filter_alpha})")


    if not input_rdvc_file_p.is_file(): 
        print(f"FATAL: Input RDVC file not found: {input_rdvc_file_p}"); sys.exit(1)
    
    if debug_output_path_decode:
        try: debug_output_path_decode.mkdir(parents=True, exist_ok=True)
        except Exception as e: print(f"Warning: Could not create debug dir {debug_output_path_decode}: {e}"); debug_output_path_decode = None
    
    video_metadata = None
    video_writer = None # Initialize to ensure it's in scope for finally
    pbar = None # Initialize for finally block
    
    # State variables for post-processing
    ref_frame_for_codec_tensor = None      # Raw reconstruction from previous step, for P-frame MC
    postprocessed_previous_frame_tensor = None # Final output from previous step, for hist match & temporal filter
    latest_decoded_iframe_tensor = None    # For low motion P-frame replacement

    try:
        with open(input_rdvc_file_p, 'rb') as rdvc_file:
            # --- 1. Read Metadata ---
            marker = rdvc_file.read(len(RDVC_METADATA_MARKER))
            if marker != RDVC_METADATA_MARKER:
                raise ValueError("Invalid RDVC file: Missing or incorrect METADATA marker at the beginning.")
            
            metadata_json_len = struct.unpack(UINT32_FORMAT, rdvc_file.read(4))[0]
            metadata_json_bytes = rdvc_file.read(metadata_json_len)
            video_metadata = json.loads(metadata_json_bytes.decode('utf-8'))

            original_width, original_height = video_metadata['original_dimensions_wh']
            original_fps = video_metadata.get('original_fps', 30.0) 
            total_frames_processed = video_metadata['total_frames_processed']
            # ... (other metadata extraction as before) ...
            
            print(f"  Video Details from RDVC Metadata: {original_width}x{original_height} @ {original_fps:.2f} FPS, {total_frames_processed} frames")
            # ... (print other metadata details) ...

            if total_frames_processed == 0: 
                print("RDVC Metadata indicates 0 frames. Nothing to decode."); sys.exit(0)

            # --- 2. Setup Device and Model ---
            if config.gpu is not None and config.gpu >= 0 and torch.cuda.is_available():
                device = torch.device(f"cuda:{config.gpu}")
                try: torch.cuda.set_device(device); print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
                except Exception as e_cuda: print(f"CUDA device error: {e_cuda}, using default CUDA or CPU."); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else: device = torch.device("cpu"); print(f"Using device: CPU")

            print("\n--- Loading Video Codec Model ---")
            model_codec = VideoCodec( # Use parameters from config
                motion_latent_channels=config.motion_latent_channels,
                # ... (rest of model parameters as before) ...
                decoder_upsample_layers=config.decoder_upsample_layers,
            )
            if not load_model_checkpoint(config.codec_checkpoint_path, model_codec, "VideoCodec", device, strict_load=False):
                raise RuntimeError(f"Failed to load VideoCodec checkpoint: {config.codec_checkpoint_path}")
            print("Video Codec model loaded.")

            # --- 3. Setup Video Writer ---
            output_video_file_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(str(output_video_file_path), fourcc, original_fps, (original_width, original_height))
            if not video_writer.isOpened(): raise IOError(f"Failed to open VideoWriter for {output_video_file_path}.")
            print(f"Video writer initialized for: {output_video_file_path}")

            # --- 4. Frame Decoding Loop ---
            print("\n--- Processing Frames for Decoding from RDVC ---")
            decoding_error_occurred = False
            pbar = tqdm(range(total_frames_processed), unit="frame", desc="Decoding from RDVC")
            
            for expected_frame_idx in pbar:
                # Read Frame Header (marker, index, type, content_len)
                # ... (same as before) ...
                marker = rdvc_file.read(len(RDVC_FRAME_MARKER))
                if not marker: 
                    print(f"ERROR: Unexpected EOF while expecting frame marker for frame {expected_frame_idx}.")
                    decoding_error_occurred = True; break
                if marker == RDVC_EOF_MARKER: 
                    print(f"INFO: Reached EOF marker after processing {expected_frame_idx} frames.")
                    if expected_frame_idx < total_frames_processed:
                         print(f"Warning: EOF marker found but metadata expected {total_frames_processed} frames.")
                    break 
                if marker != RDVC_FRAME_MARKER:
                    raise ValueError(f"Invalid RDVC file: Missing or incorrect FRAME marker for frame {expected_frame_idx}. Found: {marker}")

                stored_frame_idx = struct.unpack(UINT32_FORMAT, rdvc_file.read(4))[0]
                frame_type_char = rdvc_file.read(1) 
                frame_content_len = struct.unpack(UINT64_FORMAT, rdvc_file.read(8))[0]
                
                if stored_frame_idx != expected_frame_idx: 
                    print(f"Warning: Frame index mismatch. Expected {expected_frame_idx}, found {stored_frame_idx} in RDVC.")
                
                pbar.set_postfix_str(f"Frame {stored_frame_idx} ({frame_type_char.decode()})")
                
                frame_content_bytes = rdvc_file.read(frame_content_len)
                if len(frame_content_bytes) != frame_content_len:
                    raise EOFError(f"Could not read full frame content for frame {stored_frame_idx}.")

                content_io = io.BytesIO(frame_content_bytes)
                save_debug = (debug_output_path_decode and 
                              stored_frame_idx % config.debug_frame_interval_decode == 0)
                
                # --- Stage 1: Raw Reconstruction ---
                raw_reconstructed_this_step_tensor = None
                flow_rec_upscaled = None # Needed for LMR and P-frame debug

                if frame_type_char == b'I':
                    pbar.set_postfix_str(f"I-Frame {stored_frame_idx}")
                    iframe_ext_len = struct.unpack(UINT8_FORMAT, content_io.read(1))[0]
                    iframe_ext = content_io.read(iframe_ext_len).decode('utf-8')
                    iframe_data_bytes = content_io.read()

                    raw_reconstructed_this_step_tensor = load_image_as_tensor(iframe_data_bytes, device, is_bytes=True)
                    if raw_reconstructed_this_step_tensor is None: 
                        print(f"ERROR: Failed to load I-Frame {stored_frame_idx} data. Stopping.")
                        decoding_error_occurred=True; break
                    
                    latest_decoded_iframe_tensor = raw_reconstructed_this_step_tensor.detach().clone()
                    if save_debug: 
                        dbg_pfx_i = debug_output_path_decode / f"frame_{stored_frame_idx:06d}"
                        save_tensor_as_image_vis(raw_reconstructed_this_step_tensor, 
                                                 dbg_pfx_i.with_name(f"{dbg_pfx_i.name}_05_iframe_reconstructed_raw.png"))
                
                elif frame_type_char == b'P':
                    pbar.set_postfix_str(f"P-Frame {stored_frame_idx}")
                    if ref_frame_for_codec_tensor is None: 
                        print(f"ERROR: No reference frame for P-Frame {stored_frame_idx}. Stream error. Stopping.")
                        decoding_error_occurred=True; break
                    
                    try:
                        motion_h = struct.unpack(INT32_FORMAT, content_io.read(4))[0]
                        motion_w = struct.unpack(INT32_FORMAT, content_io.read(4))[0]
                        motion_len = struct.unpack(UINT32_FORMAT, content_io.read(4))[0]
                        motion_bs = content_io.read(motion_len)
                        residual_h = struct.unpack(INT32_FORMAT, content_io.read(4))[0]
                        residual_w = struct.unpack(INT32_FORMAT, content_io.read(4))[0]
                        residual_len = struct.unpack(UINT32_FORMAT, content_io.read(4))[0]
                        residual_bs = content_io.read(residual_len)

                        if len(motion_bs) != motion_len or len(residual_bs) != residual_len:
                            raise ValueError("P-Frame bitstream length mismatch.")

                        motion_data = (motion_bs, (motion_h, motion_w))
                        res_data = (residual_bs, (residual_h, residual_w))
                        
                        (raw_reconstructed_this_step_tensor, flow_rec_upscaled, 
                         warped_prev, mcn_out, res_rec_upscaled) = \
                            model_codec.decompress_frame_simplified(
                                ref_frame_for_codec_tensor, # Use the raw reconstruction from previous step
                                motion_data, res_data, (original_height, original_width)
                            )
                        
                        if save_debug:
                            dbg_pfx_p = debug_output_path_decode / f"frame_{stored_frame_idx:06d}"
                            save_tensor_as_image_vis(ref_frame_for_codec_tensor, dbg_pfx_p.with_name(f"{dbg_pfx_p.name}_00_prev_ref_for_codec.png"))
                            if flow_rec_upscaled is not None: visualize_flow_hsv(flow_rec_upscaled, dbg_pfx_p.with_name(f"{dbg_pfx_p.name}_01_flow_reconstructed.png"))
                            if warped_prev is not None: save_tensor_as_image_vis(warped_prev, dbg_pfx_p.with_name(f"{dbg_pfx_p.name}_02_warped_prev.png"))
                            if mcn_out is not None: save_tensor_as_image_vis(mcn_out, dbg_pfx_p.with_name(f"{dbg_pfx_p.name}_03_mc_prediction.png"))
                            if res_rec_upscaled is not None: save_tensor_as_image_vis(res_rec_upscaled, dbg_pfx_p.with_name(f"{dbg_pfx_p.name}_04_residual_reconstructed.png"), drange=(-0.5,0.5))
                            if raw_reconstructed_this_step_tensor is not None:
                                save_tensor_as_image_vis(raw_reconstructed_this_step_tensor, dbg_pfx_p.with_name(f"{dbg_pfx_p.name}_05_pframe_reconstructed_raw.png"))
                    except Exception as e_pframe_proc: 
                        print(f"\nERROR processing P-Frame {stored_frame_idx} content: {e_pframe_proc}")
                        traceback.print_exc(); decoding_error_occurred=True; break
                else:
                    print(f"ERROR: Unknown frame type '{frame_type_char.decode()}' for frame {stored_frame_idx}. Stopping.")
                    decoding_error_occurred=True; break

                if raw_reconstructed_this_step_tensor is None:
                    print(f"ERROR: Raw reconstruction failed for frame {stored_frame_idx}. Stopping.")
                    decoding_error_occurred=True; break
                
                current_processing_tensor = raw_reconstructed_this_step_tensor.detach().clone()

                # --- Stage 2: Low Motion Region Replacement (P-frames only) ---
                if frame_type_char == b'P':
                    if config.low_motion_replacement_threshold > 0.0 and \
                       latest_decoded_iframe_tensor is not None and \
                       flow_rec_upscaled is not None and \
                       current_processing_tensor is not None:
                        
                        if latest_decoded_iframe_tensor.shape == current_processing_tensor.shape:
                            flow_magnitude = torch.sqrt(
                                flow_rec_upscaled[:, 0:1]**2 + flow_rec_upscaled[:, 1:2]**2
                            )
                            low_motion_mask = flow_magnitude < config.low_motion_replacement_threshold
                            low_motion_mask_expanded = low_motion_mask.expand_as(current_processing_tensor)
                            num_replaced_pixels = torch.sum(low_motion_mask).item()
                            
                            if num_replaced_pixels > 0:
                                pbar.set_postfix_str(f"P-Frame {stored_frame_idx} (LMR {num_replaced_pixels}px)")
                                current_processing_tensor = torch.where(
                                    low_motion_mask_expanded,
                                    latest_decoded_iframe_tensor,
                                    current_processing_tensor
                                )
                                if save_debug:
                                    dbg_pfx_lmr = debug_output_path_decode / f"frame_{stored_frame_idx:06d}"
                                    save_tensor_as_image_vis(low_motion_mask.float(), 
                                                             dbg_pfx_lmr.with_name(f"{dbg_pfx_lmr.name}_06_low_motion_mask.png"))
                        else:
                            print(f"  Warning: Skipping LMR for P-frame {stored_frame_idx} due to shape mismatch: "
                                  f"current {current_processing_tensor.shape}, I-frame {latest_decoded_iframe_tensor.shape}.")
                    if save_debug: # Save current state of tensor after LMR block
                        dbg_pfx_lmr_res = debug_output_path_decode / f"frame_{stored_frame_idx:06d}"
                        save_tensor_as_image_vis(current_processing_tensor, 
                                                 dbg_pfx_lmr_res.with_name(f"{dbg_pfx_lmr_res.name}_07_after_low_motion.png"))

                # --- Stage 3: Histogram Matching (P-frames only) ---
                if frame_type_char == b'P':
                    if postprocessed_previous_frame_tensor is not None:
                        if current_processing_tensor.shape == postprocessed_previous_frame_tensor.shape:
                            try:
                                current_processing_tensor = _match_histograms_cv(
                                    current_processing_tensor, 
                                    postprocessed_previous_frame_tensor,
                                    device
                                )
                                pbar.set_postfix_str(f"P-Frame {stored_frame_idx} (LMR+HM)") # Update postfix
                            except Exception as e_hm:
                                print(f"  Warning: Histogram matching failed for frame {stored_frame_idx}: {e_hm}. Using frame before matching.")
                        else:
                             print(f"  Warning: Skipping histogram matching for P-frame {stored_frame_idx} due to shape mismatch: "
                                   f"current {current_processing_tensor.shape}, prev_postproc {postprocessed_previous_frame_tensor.shape}.")
                    if save_debug: # Save current state of tensor after HM block
                        dbg_pfx_hm_res = debug_output_path_decode / f"frame_{stored_frame_idx:06d}"
                        save_tensor_as_image_vis(current_processing_tensor, 
                                                 dbg_pfx_hm_res.with_name(f"{dbg_pfx_hm_res.name}_08_hist_matched.png"))
                
                # --- Stage 4: Temporal Low-Pass Filter (All frames) ---
                alpha = config.temporal_filter_alpha
                final_output_frame_tensor = None

                if postprocessed_previous_frame_tensor is not None and \
                   current_processing_tensor.shape == postprocessed_previous_frame_tensor.shape:
                    final_output_frame_tensor = alpha * postprocessed_previous_frame_tensor + \
                                                (1.0 - alpha) * current_processing_tensor
                    final_output_frame_tensor = torch.clamp(final_output_frame_tensor, 0.0, 1.0)
                    if frame_type_char == b'I': pbar.set_postfix_str(f"I-Frame {stored_frame_idx} (TF)")
                    else: pbar.set_postfix_str(f"P-Frame {stored_frame_idx} (LMR+HM+TF)")
                else:
                    if postprocessed_previous_frame_tensor is not None and \
                       current_processing_tensor.shape != postprocessed_previous_frame_tensor.shape:
                        print(f"  Warning: Skipping temporal filter for frame {stored_frame_idx} due to shape mismatch: "
                              f"current {current_processing_tensor.shape}, prev_postproc {postprocessed_previous_frame_tensor.shape}.")
                    final_output_frame_tensor = current_processing_tensor # Use current if no previous or shape mismatch for filter

                if save_debug:
                    dbg_pfx_tf_res = debug_output_path_decode / f"frame_{stored_frame_idx:06d}"
                    save_tensor_as_image_vis(final_output_frame_tensor, 
                                             dbg_pfx_tf_res.with_name(f"{dbg_pfx_tf_res.name}_09_final_to_video.png"))
                
                # --- Write Reconstructed and Post-Processed Frame to Video ---
                if final_output_frame_tensor is not None:
                    frame_to_write_list = tensor_to_cv2_bgr(final_output_frame_tensor)
                    if frame_to_write_list:
                        video_writer.write(frame_to_write_list[0]) 
                    else: 
                        print(f"Error: tensor_to_cv2_bgr failed for frame {stored_frame_idx}. Cannot write to video.");
                        decoding_error_occurred=True; break
                else: 
                    print(f"Error: Final output tensor is None for frame {stored_frame_idx}. Cannot write. Stopping.");
                    decoding_error_occurred=True; break
                
                # --- Update References for Next Iteration ---
                ref_frame_for_codec_tensor = raw_reconstructed_this_step_tensor.detach().clone()
                if final_output_frame_tensor is not None:
                    postprocessed_previous_frame_tensor = final_output_frame_tensor.detach().clone()
                
                if device.type == 'cuda' and stored_frame_idx > 0 and stored_frame_idx % 50 == 0:
                    torch.cuda.empty_cache() 
            
            # --- After Loop: Check for EOF Marker ---
            # ... (same as before) ...
            if not decoding_error_occurred and marker != RDVC_EOF_MARKER: 
                final_marker_check = rdvc_file.read(len(RDVC_EOF_MARKER))
                if final_marker_check and final_marker_check != RDVC_EOF_MARKER:
                    print(f"Warning: Expected EOF marker at the end of RDVC file after all frames, but found: {final_marker_check}")
                elif not final_marker_check and expected_frame_idx + 1 == total_frames_processed : 
                    pass 
                elif not final_marker_check: 
                     print(f"Warning: RDVC file ended without EOF marker. Processed {pbar.n}/{total_frames_processed} frames based on counter.")


    except ValueError as ve: 
        print(f"\nVALUE ERROR (likely RDVC format issue): {ve}"); traceback.print_exc(); decoding_error_occurred = True
    except EOFError as eofe: 
        print(f"\nEOF ERROR (likely truncated RDVC file or content length mismatch): {eofe}"); traceback.print_exc(); decoding_error_occurred = True
    except Exception as loop_err: 
        print(f"\nUNEXPECTED ERROR during RDVC decoding: {loop_err}"); traceback.print_exc(); decoding_error_occurred = True
    finally:
        if pbar: pbar.close()
        if video_writer: video_writer.release()
        
        print("\n--- Video Decoding from RDVC Finished ---")
        status_msg = "Reconstructed video" + \
                     (" (potentially incomplete or with errors)" if decoding_error_occurred else "") + \
                     f" saved at: {output_video_file_path}"
        print(status_msg)
        if debug_output_path_decode and any(debug_output_path_decode.iterdir()): 
            print(f"Debug frames (if any) are in: {debug_output_path_decode}")
        elif debug_output_path_decode:
             print(f"Debug frames directory specified ({debug_output_path_decode}), but no files were saved (check interval or errors).")

    print("\n" + "=" * 60 + "\n--- Decoding Complete ---\n" + "=" * 60)

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Codec - Encode to RDVC or Decode from RDVC")
    parser.add_argument('--mode', type=str, choices=['encode', 'decode'], required=True, help="Operation mode")
    parser.add_argument('--gpu', type=int, default=None, 
                        help="GPU ID (0, 1, ...), or -1 for CPU. "
                             "Default (if not specified): uses CodecConfig default (e.g., GPU 0 if available, else CPU).")
    parser.add_argument('--raft_backend', type=str, choices=['auto', 'torchvision', 'local'], default=None, 
                        help="Which RAFT implementation to use for encoding. Default: uses CodecConfig default (e.g., 'auto').")
    parser.add_argument('--temporal_filter_alpha', type=float, default=None,
                        help="Alpha value for temporal low-pass filter in decoder (0.0 to 1.0). Default: uses CodecConfig default.")
    # ... (other CLI arguments if needed) ...

    args = parser.parse_args()
    
    config = CodecConfig() 
    config.update_from_args(args) 

    # --- General Validations ---
    if not os.path.isfile(config.codec_checkpoint_path):
        print(f"FATAL: Codec checkpoint '{Path(config.codec_checkpoint_path).resolve()}' not found.")
        sys.exit(1)

    # --- Mode Specific Logic ---
    if args.mode == 'encode':
        print("Mode: ENCODE (Input Video --> .rdvc file)")
        # ... (encode validations as before) ...
        input_file_for_check = Path(config.input_file_path)
        if not input_file_for_check.is_file():
            print(f"FATAL: Input file '{input_file_for_check.resolve()}' (from CodecConfig) not found or invalid.")
            sys.exit(1)
        if input_file_for_check.suffix.lower() == ".yuv":
            if not all([config.input_yuv_width, config.input_yuv_height, 
                        config.input_yuv_fps, config.input_yuv_pixel_format]):
                print("FATAL ERROR: YUV parameters in CodecConfig are not fully set for .yuv input.")
                sys.exit(1)
            if config.input_yuv_pixel_format != "yuv420p":
                 print(f"FATAL ERROR: Unsupported input_yuv_pixel_format: '{config.input_yuv_pixel_format}'. Only 'yuv420p' supported.")
                 sys.exit(1)
        Path(config.output_rdvc_file).parent.mkdir(parents=True, exist_ok=True)
        encode_video_main(config)

    elif args.mode == 'decode':
        print("Mode: DECODE (.rdvc file --> Output Video)")
        # ... (decode validations as before) ...
        input_rdvc_file_for_check = Path(config.input_rdvc_file)
        if not input_rdvc_file_for_check.is_file():
            print(f"FATAL: Input RDVC file '{input_rdvc_file_for_check.resolve()}' (from CodecConfig) not found.")
            sys.exit(1)
        Path(config.output_video_path_decode).parent.mkdir(parents=True, exist_ok=True)
        if config.debug_frames_dir_decode:
            Path(config.debug_frames_dir_decode).mkdir(parents=True, exist_ok=True)
        decode_video_main(config)
    else:
        parser.print_help()
        sys.exit(1)
        
    print("\nProcess finished.")