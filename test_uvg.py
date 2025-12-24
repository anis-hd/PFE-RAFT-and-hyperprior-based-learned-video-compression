# test_codec_metrics.py
import argparse
import os
import sys
import numpy as np
import torch
import cv2
import math
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Attempt to import from codec_processing.py
try:
    from codec_processing import (
        CodecConfig,
        encode_video_main,
        decode_video_main,
        read_yuv_frame_generator,
        preprocess_frame_codec
    )
except ImportError as e:
    print(f"Error importing from codec_processing.py: {e}")
    print("Please ensure 'codec_processing.py' (your original codec script) is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)
except Exception as e_codec_load:
    print(f"An unexpected error occurred while trying to load modules from codec_processing.py: {e_codec_load}")
    print("This might be due to an issue within codec_processing.py itself during its import time.")
    sys.exit(1)


try:
    from pytorch_msssim import ms_ssim
except ImportError:
    print("WARNING: pytorch_msssim not found. MS-SSIM will not be calculated.")
    print("You can install it by running: pip install pytorch-msssim")
    ms_ssim = None

# --- Configuration ---
NUM_FRAMES_TO_TEST = 50
INPUT_YUV_FILE = Path("./input.yuv") # Path to your source YUV file

# --- THESE MUST MATCH YOUR input.yuv PROPERTIES ---
YUV_WIDTH = 1920
YUV_HEIGHT = 1080
YUV_FPS = 30.0
YUV_PIXEL_FORMAT = "yuv420p" # codec_processing.py's reader only supports yuv420p


# --- Helper Functions ---
def calculate_psnr(img1_tensor, img2_tensor, max_val=1.0):
    """Calculates PSNR between two PyTorch tensors (BCHW, range [0,1])."""
    if not torch.is_tensor(img1_tensor) or not torch.is_tensor(img2_tensor):
        raise TypeError("Inputs must be PyTorch tensors.")
    if img1_tensor.shape != img2_tensor.shape:
        # Attempt to resize img2_tensor to img1_tensor's spatial dimensions
        # This can happen if decoder output has slightly different padding/cropping
        print(f"Warning: PSNR shape mismatch. Img1: {img1_tensor.shape}, Img2: {img2_tensor.shape}. Attempting resize.")
        try:
            img2_tensor = torch.nn.functional.interpolate(
                img2_tensor,
                size=(img1_tensor.shape[2], img1_tensor.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            print(f"Resized Img2 to: {img2_tensor.shape}")
        except Exception as e_resize:
            print(f"Could not resize for PSNR: {e_resize}")
            raise ValueError(f"Input shapes must match. Got {img1_tensor.shape} and {img2_tensor.shape}, resize failed.")

    mse = torch.mean((img1_tensor - img2_tensor) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())

def read_frames_from_video_file(video_path, num_frames_expected, target_width, target_height):
    """Reads specified number of frames from a video file and converts to RGB NumPy."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    frames = []
    for i in range(num_frames_expected):
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"Warning: Video ended prematurely. Read {len(frames)} frames, expected {num_frames_expected}.")
            break
        # Ensure consistent dimensions with original frames if decoder altered them
        if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
            frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def write_frames_to_yuv(frames_rgb_np, output_yuv_path, width, height, pixel_format="yuv420p"):
    """Writes a list of RGB NumPy frames to a YUV file (only yuv420p supported)."""
    if pixel_format != "yuv420p":
        raise NotImplementedError("Only yuv420p is supported for writing.")
    if not frames_rgb_np:
        print("Warning: No frames provided to write_frames_to_yuv.")
        return

    # YUV420p requires even dimensions
    if width % 2 != 0:
        print(f"Warning: Width {width} is odd, adjusting to {width -1 } for YUV420p.")
        width -= 1
    if height % 2 != 0:
        print(f"Warning: Height {height} is odd, adjusting to {height -1 } for YUV420p.")
        height -= 1

    try:
        with open(output_yuv_path, 'wb') as f:
            for frame_rgb in tqdm(frames_rgb_np, desc="Writing temp YUV"):
                # Resize frame to target width/height if necessary (e.g., after cropping for even dims)
                if frame_rgb.shape[1] != width or frame_rgb.shape[0] != height:
                    frame_rgb_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
                else:
                    frame_rgb_resized = frame_rgb

                # OpenCV's YUV_I420 is a packed format: Y plane, then U, then V.
                # It results in a single matrix of shape (height * 3/2, width).
                frame_yuv_i420 = cv2.cvtColor(frame_rgb_resized, cv2.COLOR_RGB2YUV_I420)
                f.write(frame_yuv_i420.tobytes())
    except Exception as e:
        print(f"Error writing YUV file {output_yuv_path}: {e}")
        raise

def main_test_codec(config_override=None):
    print("--- Starting Codec Metrics Test ---")

    cfg = CodecConfig()
    if config_override:
        for key, value in config_override.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                print(f"Overriding config: {key} = {value}")
            else:
                print(f"Warning: Unknown config override for CodecConfig: {key}")

    # Check codec checkpoint existence early
    if not Path(cfg.codec_checkpoint_path).exists():
        print(f"CRITICAL ERROR: Codec checkpoint '{cfg.codec_checkpoint_path}' not found!")
        print("Please ensure `codec_checkpoint_path` in `CodecConfig` (inside codec_processing.py) is correct,")
        print("or provide a valid path if overriding.")
        return

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="codec_test_")
    temp_dir = Path(temp_dir_obj.name)
    print(f"Using temporary directory: {temp_dir}")

    temp_input_yuv = temp_dir / "temp_input_for_codec.yuv"
    temp_rdvc_file = temp_dir / "compressed_video.rdvc"
    temp_decoded_mp4 = temp_dir / "reconstructed_video.mp4"

    # --- 1. Read Original Frames ---
    print(f"Reading {NUM_FRAMES_TO_TEST} frames from {INPUT_YUV_FILE}...")
    if not INPUT_YUV_FILE.exists():
        print(f"ERROR: Input YUV file '{INPUT_YUV_FILE}' not found!")
        return

    original_frames_np_rgb = []
    original_frames_tensors = []

    frame_gen = read_yuv_frame_generator(
        str(INPUT_YUV_FILE), YUV_WIDTH, YUV_HEIGHT, YUV_PIXEL_FORMAT
    )

    # Determine device for metrics (and for preprocess_frame_codec)
    # Based on CodecConfig's GPU setting
    if cfg.gpu is not None and cfg.gpu >=0 and torch.cuda.is_available():
        device_for_metrics = torch.device(f"cuda:{cfg.gpu}")
    else:
        device_for_metrics = torch.device("cpu")
    print(f"Using device for metrics and preprocessing: {device_for_metrics}")

    for i, frame_np_rgb in enumerate(frame_gen):
        if i >= NUM_FRAMES_TO_TEST:
            break
        original_frames_np_rgb.append(frame_np_rgb)
        frame_tensor = preprocess_frame_codec(frame_np_rgb, device_for_metrics)
        if frame_tensor is None:
            print(f"Error preprocessing original frame {i}. Aborting test.")
            return
        original_frames_tensors.append(frame_tensor)

    if not original_frames_np_rgb:
        print("No frames read from input. Aborting.")
        return
    
    actual_num_frames_tested = len(original_frames_np_rgb)
    if actual_num_frames_tested < NUM_FRAMES_TO_TEST:
        print(f"Warning: Only read {actual_num_frames_tested} frames, less than requested {NUM_FRAMES_TO_TEST}.")
    print(f"Successfully read {actual_num_frames_tested} original frames.")


    # --- 2. Write selected frames to a temporary YUV for encoding ---
    print(f"Writing {actual_num_frames_tested} frames to temporary YUV file: {temp_input_yuv}")
    try:
        # Use the YUV_WIDTH/HEIGHT constants, assuming they represent the true frame dimensions
        write_frames_to_yuv(original_frames_np_rgb, temp_input_yuv, YUV_WIDTH, YUV_HEIGHT, YUV_PIXEL_FORMAT)
    except Exception as e:
        print(f"Failed to write temporary YUV for encoding: {e}")
        return

    # --- 3. Encode the temporary YUV file ---
    print("Encoding frames...")
    cfg.input_file_path = str(temp_input_yuv)
    cfg.output_rdvc_file = str(temp_rdvc_file)
    # These must match the temporary YUV file's properties
    cfg.input_yuv_width = YUV_WIDTH
    cfg.input_yuv_height = YUV_HEIGHT
    cfg.input_yuv_fps = YUV_FPS # FPS for metadata in RDVC
    cfg.input_yuv_pixel_format = YUV_PIXEL_FORMAT

    try:
        encode_video_main(cfg)
    except SystemExit as e:
        print(f"Encoding process exited (possibly expected on error): {e}")
        # Check if RDVC file was created before failing
    except Exception as e:
        print(f"An unexpected error occurred during encoding: {e}")
        import traceback
        traceback.print_exc()
        return

    if not temp_rdvc_file.exists() or temp_rdvc_file.stat().st_size == 0:
        print("ERROR: Encoding did not produce an RDVC file or the file is empty.")
        return
    
    compressed_size_bytes = temp_rdvc_file.stat().st_size
    print(f"Encoding successful. Compressed RDVC size: {compressed_size_bytes} bytes.")

    # --- 4. Decode the RDVC file ---
    print("Decoding frames...")
    cfg.input_rdvc_file = str(temp_rdvc_file) # Input for decoder
    cfg.output_video_path_decode = str(temp_decoded_mp4) # Output MP4 from decoder
    # Ensure debug options are off or point to temp_dir if desired for codec_processing
    cfg.debug_frames_dir_decode = str(temp_dir / "decode_debug_frames")

    try:
        decode_video_main(cfg)
    except SystemExit as e:
        print(f"Decoding process exited (possibly expected on error): {e}")
    except Exception as e:
        print(f"An unexpected error occurred during decoding: {e}")
        import traceback
        traceback.print_exc()
        return

    if not temp_decoded_mp4.exists() or temp_decoded_mp4.stat().st_size == 0:
        print("ERROR: Decoding did not produce an MP4 file or the file is empty.")
        return
    print("Decoding successful.")

    # --- 5. Read Decoded Frames from the MP4 ---
    print(f"Reading {actual_num_frames_tested} decoded frames from {temp_decoded_mp4}...")
    # Pass YUV_WIDTH/HEIGHT as target dimensions for consistency
    decoded_frames_np_rgb = read_frames_from_video_file(temp_decoded_mp4, actual_num_frames_tested, YUV_WIDTH, YUV_HEIGHT)

    if len(decoded_frames_np_rgb) != actual_num_frames_tested:
        print(f"CRITICAL ERROR: Read {len(decoded_frames_np_rgb)} decoded frames, but expected {actual_num_frames_tested}.")
        print("Metrics calculation will be unreliable or fail. Aborting.")
        return

    decoded_frames_tensors = []
    for i, frame_np_rgb in enumerate(decoded_frames_np_rgb):
        frame_tensor = preprocess_frame_codec(frame_np_rgb, device_for_metrics)
        if frame_tensor is None:
            print(f"Error preprocessing decoded frame {i}. Aborting test.")
            return
        decoded_frames_tensors.append(frame_tensor)
    
    if len(original_frames_tensors) != len(decoded_frames_tensors):
        print(f"CRITICAL ERROR: Mismatch in number of original ({len(original_frames_tensors)}) and decoded ({len(decoded_frames_tensors)}) tensors.")
        return
    
    # --- 6. Calculate Metrics ---
    print("Calculating metrics...")
    all_psnr = []
    all_ms_ssim = []

    for i in tqdm(range(actual_num_frames_tested), desc="Calculating PSNR/MS-SSIM"):
        orig_tensor = original_frames_tensors[i]
        deco_tensor = decoded_frames_tensors[i]

        try:
            psnr_val = calculate_psnr(orig_tensor, deco_tensor)
            all_psnr.append(psnr_val)
        except ValueError as e_psnr: # Catch shape mismatch from calculate_psnr
            print(f"Skipping PSNR for frame {i} due to error: {e_psnr}")
            all_psnr.append(float('nan'))


        if ms_ssim is not None:
            try:
                # Ensure shapes match for MS-SSIM too
                if orig_tensor.shape != deco_tensor.shape:
                     deco_tensor_resized_msssim = torch.nn.functional.interpolate(
                        deco_tensor,
                        size=(orig_tensor.shape[2], orig_tensor.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    deco_tensor_resized_msssim = deco_tensor

                ms_ssim_val = ms_ssim(orig_tensor, deco_tensor_resized_msssim, data_range=1.0, size_average=True)
                all_ms_ssim.append(ms_ssim_val.item())
            except Exception as e_msssim:
                print(f"Error calculating MS-SSIM for frame {i}: {e_msssim}. Skipping MS-SSIM for this frame.")
                all_ms_ssim.append(float('nan'))
        else:
            all_ms_ssim.append(float('nan'))


    avg_psnr = np.nanmean([p for p in all_psnr if not math.isinf(p)]) if all_psnr else float('nan')
    avg_ms_ssim = np.nanmean(all_ms_ssim) if all_ms_ssim else float('nan')


    # BPP (Bits Per Pixel)
    total_pixels_in_sequence = actual_num_frames_tested * YUV_WIDTH * YUV_HEIGHT
    bpp = (compressed_size_bytes * 8) / total_pixels_in_sequence if total_pixels_in_sequence > 0 else float('nan')

    # Bitrate (kbps)
    duration_seconds = actual_num_frames_tested / YUV_FPS if YUV_FPS > 0 else float('nan')
    bitrate_kbps = (compressed_size_bytes * 8) / (duration_seconds * 1000) if duration_seconds > 0 and not math.isnan(duration_seconds) else float('nan')

    # --- 7. Print Results ---
    print("\n--- Test Results ---")
    print(f"Frames tested: {actual_num_frames_tested}")
    print(f"Average PSNR (dB): {avg_psnr:.2f}")
    if ms_ssim is not None:
        print(f"Average MS-SSIM: {avg_ms_ssim:.4f}")
    else:
        print("Average MS-SSIM: Not calculated (pytorch_msssim not installed or error during calculation)")
    print(f"Average BPP (bits per pixel): {bpp:.4f}")
    print(f"Bitrate (kbps): {bitrate_kbps:.2f}")
    print(f"Total compressed file size (bytes): {compressed_size_bytes}")

    # temp_dir_obj.cleanup() will be called when obj goes out of scope
    print(f"Temporary directory {temp_dir} will be cleaned up.")
    print("--- Codec Metrics Test Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metrics test on the video codec.")
    parser.add_argument('--gpu', type=int, default=None,
                        help="GPU ID to use (e.g., 0, 1). Default uses CodecConfig setting. Use -1 for CPU.")
    # Example: Add override for codec checkpoint path from CLI
    parser.add_argument('--codec_checkpoint', type=str, default=None,
                        help="Path to the codec checkpoint file to use.")

    cli_args = parser.parse_args()
    config_override = {}

    if cli_args.gpu is not None:
        config_override['gpu'] = cli_args.gpu if cli_args.gpu != -1 else None # None for CPU

    if cli_args.codec_checkpoint:
        config_override['codec_checkpoint_path'] = cli_args.codec_checkpoint
    
    # Check if CUDA is available if a GPU is requested
    if 'gpu' in config_override and config_override['gpu'] is not None:
        if not torch.cuda.is_available():
            print(f"Warning: GPU {config_override['gpu']} requested, but CUDA is not available. Falling back to CPU.")
            config_override['gpu'] = None
        else:
            try:
                # Test if the specific GPU ID is valid
                torch.cuda.get_device_name(config_override['gpu'])
            except (AssertionError, RuntimeError) as e: # AssertionError for invalid device id
                print(f"Warning: GPU {config_override['gpu']} is not a valid CUDA device ({e}). Falling back to default GPU or CPU.")
                # Fallback to default GPU if any, else CPU
                config_override['gpu'] = 0 if torch.cuda.device_count() > 0 else None


    main_test_codec(config_override)