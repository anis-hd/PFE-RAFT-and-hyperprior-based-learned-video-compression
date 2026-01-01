
import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def classical_fft_compress_patch(image_patch_array, keep_n_coeffs=None):
    if image_patch_array.ndim != 2:
        raise ValueError("Input image_patch_array must be 2D.")
    H, W = image_patch_array.shape
    num_total_coeffs_classical = H * W
    fft_coeffs = fft2(image_patch_array.astype(np.float64))

    if keep_n_coeffs is None or keep_n_coeffs <= 0 or keep_n_coeffs >= num_total_coeffs_classical:
        thresholded_coeffs = fft_coeffs
    else:
        flat_coeffs = fft_coeffs.flatten()
        magnitudes = np.abs(flat_coeffs)
        indices_to_keep = np.argsort(magnitudes)[-keep_n_coeffs:]
        thresholded_flat_coeffs = np.zeros_like(flat_coeffs, dtype=np.complex128)
        thresholded_flat_coeffs[indices_to_keep] = flat_coeffs[indices_to_keep]
        thresholded_coeffs = thresholded_flat_coeffs.reshape(fft_coeffs.shape)

    reconstructed_patch = ifft2(thresholded_coeffs)
    reconstructed_patch_real = np.real(reconstructed_patch)
    reconstructed_patch_clipped = np.clip(reconstructed_patch_real, 0, 255)
    return reconstructed_patch_clipped.astype(np.uint8)

def visualize_classical_fft_spectrum(image_patch_array, ax=None, title="Classical FFT Spectrum"):
    if image_patch_array.ndim != 2: raise ValueError("Input image_patch_array must be 2D.")
    fft_coeffs = fft2(image_patch_array.astype(np.float64))
    shifted_fft = fftshift(fft_coeffs)
    magnitude_spectrum = np.log1p(np.abs(shifted_fft))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(magnitude_spectrum, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label="Log Magnitude")
    if ax is None: 
        plt.show()
    return magnitude_spectrum

def classical_fft_compress_full_image_patched(original_pil_image_gray,
                                              patch_width, patch_height,
                                              coeffs_percentage_to_keep,
                                              progress_desc="Classical FFT Processing Patches"):
    img_width, img_height = original_pil_image_gray.size
    reconstructed_large_img_array = np.zeros((img_height, img_width), dtype=np.uint8)
    num_total_classical_coeffs_per_patch = patch_width * patch_height
    coeffs_to_keep_val = max(1, int(coeffs_percentage_to_keep * num_total_classical_coeffs_per_patch))
    num_patches_x = math.ceil(img_width / patch_width)
    num_patches_y = math.ceil(img_height / patch_height)
    patch_coords = [(i, j) for j in range(num_patches_y) for i in range(num_patches_x)]

    for i, j in tqdm(patch_coords, desc=progress_desc, unit="patch"):
        x0, y0 = i * patch_width, j * patch_height
        x1, y1 = min(x0 + patch_width, img_width), min(y0 + patch_height, img_height)
        current_patch_pil = original_pil_image_gray.crop((x0, y0, x1, y1))
        actual_w, actual_h = current_patch_pil.size
        padded_patch_for_processing_pil = Image.new("L", (patch_width, patch_height), color=0)
        padded_patch_for_processing_pil.paste(current_patch_pil, (0,0))
        current_patch_arr_padded = np.array(padded_patch_for_processing_pil)
        reconstructed_patch_array_padded = classical_fft_compress_patch(
            current_patch_arr_padded, keep_n_coeffs=coeffs_to_keep_val
        )
        reconstructed_patch_to_paste = reconstructed_patch_array_padded[:actual_h, :actual_w]
        reconstructed_large_img_array[y0:y1, x0:x1] = reconstructed_patch_to_paste
    return reconstructed_large_img_array
