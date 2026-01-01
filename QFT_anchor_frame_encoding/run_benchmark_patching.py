
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from qft import qft_compress_full_image_patched,visualize_qft_fourier_space_from_statevector
from fft import classical_fft_compress_full_image_patched,visualize_classical_fft_spectrum






def save_pil_image(image_array, output_filename="image.png"):
    if image_array is None or image_array.ndim != 2 or image_array.shape[0] == 0 or image_array.shape[1] == 0:
        pil_image = Image.new("L", (10, 10), color=0)
    else:
        pil_image = Image.fromarray(image_array, "L")
    pil_image.save(output_filename)
    return pil_image

def calculate_psnr(original_image_array, compressed_image_array):
    original_flat = original_image_array.astype(np.float64).flatten()
    compressed_flat = compressed_image_array.astype(np.float64).flatten()

    if original_flat.shape[0] == 0 or compressed_flat.shape[0] == 0: return 0.0
    min_len = min(len(original_flat), len(compressed_flat))
    original_flat = original_flat[:min_len]
    compressed_flat = compressed_flat[:min_len]

    mse = np.mean((original_flat - compressed_flat) ** 2)
    if mse < 1e-10: return 100.0
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

if __name__ == "__main__":
    input_image_filename = "input.png" 
    if not os.path.exists(input_image_filename):
        print(f"Creating dummy 1024x1024 '{input_image_filename}' for testing. This may take a moment...")
        dummy_img = Image.new('L', (1024, 1024), color=128)
        for x_ in range(0, 1024, 32): 
            for y_ in range(1024):
                dummy_img.putpixel((x_,y_), (x_ + y_) % 256)
                dummy_img.putpixel((y_,x_), (x_ * y_) % 256)
        dummy_img.save(input_image_filename)
        print("Dummy image created.")


    output_prefix = "large_image_1024_benchmark"

    TARGET_IMAGE_SIZE = 1024 
    PROCESSING_PATCH_WIDTH = 16
    PROCESSING_PATCH_HEIGHT = 16



    shots_for_reconstruction = 1024 

    # Percentages of coefficients to keep
    coeffs_percentages = [0.10, 0.50] 

    try:
        img_original_pil_raw = Image.open(input_image_filename)

        w_raw, h_raw = img_original_pil_raw.size
        min_dim_raw = min(w_raw, h_raw)

        left_sq = (w_raw - min_dim_raw) // 2
        top_sq = (h_raw - min_dim_raw) // 2
        right_sq = left_sq + min_dim_raw
        bottom_sq = top_sq + min_dim_raw
        img_square_pil = img_original_pil_raw.crop((left_sq, top_sq, right_sq, bottom_sq))

        if min_dim_raw < TARGET_IMAGE_SIZE:
            final_image_for_processing_pil = Image.new("L", (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), color=0)
            paste_x = (TARGET_IMAGE_SIZE - min_dim_raw) // 2
            paste_y = (TARGET_IMAGE_SIZE - min_dim_raw) // 2
            final_image_for_processing_pil.paste(img_square_pil.convert("L"), (paste_x, paste_y))
        elif min_dim_raw > TARGET_IMAGE_SIZE:
            left_crop = (min_dim_raw - TARGET_IMAGE_SIZE) // 2
            top_crop = (min_dim_raw - TARGET_IMAGE_SIZE) // 2
            right_crop = left_crop + TARGET_IMAGE_SIZE
            bottom_crop = top_crop + TARGET_IMAGE_SIZE
            final_image_for_processing_pil = img_square_pil.crop((left_crop, top_crop, right_crop, bottom_crop)).convert("L")
        else: 
            final_image_for_processing_pil = img_square_pil.convert("L")

        original_image_pil_gray = final_image_for_processing_pil
        original_image_np_array = np.array(original_image_pil_gray)

        save_pil_image(original_image_np_array, f"{output_prefix}_original_processed_{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}.png")
    except Exception as e:
        exit()

    results_data = []
    reconstructed_images_qft = {} 
    reconstructed_images_fft = {} 

    first_patch_qft_sv_for_viz = None
    first_patch_original_for_viz = None

    for percent_coeffs in coeffs_percentages:
        print(f"\n--- Processing with {percent_coeffs*100:.1f}% coefficients ---")

        # QFT Full Image Compression
        desc_qft = f"QFT {percent_coeffs*100:.0f}%"
        start_qft = time.time()
        # Get patch data only on the run for the highest percentage, to ensure it's a "good" representation
        should_get_patch_data = (percent_coeffs == max(coeffs_percentages)) and (first_patch_qft_sv_for_viz is None)

        recon_qft_np, patch_data_qft = qft_compress_full_image_patched(
            original_image_pil_gray, PROCESSING_PATCH_WIDTH, PROCESSING_PATCH_HEIGHT,
            percent_coeffs, shots_for_reconstruction, return_first_patch_data=should_get_patch_data,
            progress_desc=desc_qft
        )
        if should_get_patch_data and patch_data_qft:
            first_patch_qft_sv_for_viz, first_patch_original_for_viz = patch_data_qft

        psnr_qft = calculate_psnr(original_image_np_array, recon_qft_np)
        end_qft = time.time()
        time_qft = end_qft - start_qft
        reconstructed_images_qft[percent_coeffs] = recon_qft_np
        print(f"  QFT: PSNR={psnr_qft:.2f} dB, Time={time_qft:.2f}s")

        # Classical FFT Full Image Compression
        desc_fft = f"FFT {percent_coeffs*100:.0f}%"
        start_fft = time.time()
        recon_fft_np = classical_fft_compress_full_image_patched(
            original_image_pil_gray, PROCESSING_PATCH_WIDTH, PROCESSING_PATCH_HEIGHT, percent_coeffs,
            progress_desc=desc_fft
        )
        psnr_classical = calculate_psnr(original_image_np_array, recon_fft_np)
        end_fft = time.time()
        time_fft = end_fft - start_fft
        reconstructed_images_fft[percent_coeffs] = recon_fft_np
        print(f"  Classical FFT: PSNR={psnr_classical:.2f} dB, Time={time_fft:.2f}s")

        results_data.append({
            "Coeff %": percent_coeffs * 100,
            "PSNR QFT (dB)": psnr_qft,
            "Time QFT (s)": time_qft,
            "PSNR FFT (dB)": psnr_classical,
            "Time FFT (s)": time_fft,
        })

    results_df = pd.DataFrame(results_data)
    print("\n--- Benchmark Results ---")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{output_prefix}_benchmark_results.csv", index=False)

    fig_psnr_time, (ax_psnr, ax_time) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax_psnr.plot(results_df["Coeff %"], results_df["PSNR QFT (dB)"], marker='o', linestyle='-', label=f'QFT Patch-Based')
    ax_psnr.plot(results_df["Coeff %"], results_df["PSNR FFT (dB)"], marker='s', linestyle='--', label=f'Classical FFT Patch-Based')
    ax_psnr.set_ylabel('PSNR (dB)')
    ax_psnr.set_title(f'Overall PSNR vs. Percentage of Coefficients Kept\n(Image: {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}, Patch: {PROCESSING_PATCH_WIDTH}x{PROCESSING_PATCH_HEIGHT})')
    ax_psnr.legend()
    ax_psnr.grid(True)

    ax_time.plot(results_df["Coeff %"], results_df["Time QFT (s)"], marker='o', linestyle='-', label=f'QFT Time')
    ax_time.plot(results_df["Coeff %"], results_df["Time FFT (s)"], marker='s', linestyle='--', label=f'Classical FFT Time')
    ax_time.set_xlabel('Percentage of Coefficients Kept per Patch (%)')
    ax_time.set_ylabel('Processing Time (s)')
    ax_time.set_title('Processing Time vs. Percentage of Coefficients Kept')
    ax_time.legend()
    ax_time.grid(True)
    ax_time.set_yscale('log') 
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_psnr_time_comparison.png")
    plt.show()

    vis_percentages = sorted(list(set([min(coeffs_percentages),
                                  coeffs_percentages[len(coeffs_percentages)//2],
                                  max(coeffs_percentages)])))

    num_vis_levels = len(vis_percentages)
    fig_recon_grid = plt.figure(figsize=(5 * (num_vis_levels +1), 10)) 
    gs = gridspec.GridSpec(2, num_vis_levels + 1, figure=fig_recon_grid) 

    ax_orig = fig_recon_grid.add_subplot(gs[:, 0]) 
    ax_orig.imshow(original_image_np_array, cmap='gray')
    ax_orig.set_title(f"Original\n{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}")
    ax_orig.axis('off')

    for i, p_coeff in enumerate(vis_percentages):
        # QFT images in top row
        ax_qft = fig_recon_grid.add_subplot(gs[0, i + 1])
        if p_coeff in reconstructed_images_qft:
            img_qft = reconstructed_images_qft[p_coeff]
            psnr_qft_val = results_df[results_df["Coeff %"] == p_coeff*100]["PSNR QFT (dB)"].iloc[0]
            ax_qft.imshow(img_qft, cmap='gray')
            ax_qft.set_title(f"QFT {p_coeff*100:.0f}% Coeffs\nPSNR: {psnr_qft_val:.2f} dB")
        else:
            ax_qft.text(0.5, 0.5, "N/A", ha='center', va='center')
            ax_qft.set_title(f"QFT {p_coeff*100:.0f}% Coeffs")
        ax_qft.axis('off')

        # FFT images in bottom row
        ax_fft = fig_recon_grid.add_subplot(gs[1, i + 1])
        if p_coeff in reconstructed_images_fft:
            img_fft = reconstructed_images_fft[p_coeff]
            psnr_fft_val = results_df[results_df["Coeff %"] == p_coeff*100]["PSNR FFT (dB)"].iloc[0]
            ax_fft.imshow(img_fft, cmap='gray')
            ax_fft.set_title(f"FFT {p_coeff*100:.0f}% Coeffs\nPSNR: {psnr_fft_val:.2f} dB")
        else:
            ax_fft.text(0.5, 0.5, "N/A", ha='center', va='center')
            ax_fft.set_title(f"FFT {p_coeff*100:.0f}% Coeffs")
        ax_fft.axis('off')

    fig_recon_grid.suptitle(f"Full Image Reconstruction Comparison (Patch: {PROCESSING_PATCH_WIDTH}x{PROCESSING_PATCH_HEIGHT})", fontsize=16)
    fig_recon_grid.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_prefix}_full_image_reconstruction_grid.png")
    plt.show()




    if first_patch_qft_sv_for_viz is not None and first_patch_original_for_viz is not None:
        n_qubits_viz = int(np.ceil(np.log2(first_patch_original_for_viz.size)))

        fig_fourier, axes_fourier = plt.subplots(1, 3, figsize=(18, 6))

        axes_fourier[0].imshow(first_patch_original_for_viz, cmap='gray')
        axes_fourier[0].set_title(f"Original First Patch\n({PROCESSING_PATCH_WIDTH}x{PROCESSING_PATCH_HEIGHT})")
        axes_fourier[0].axis('off')

        visualize_qft_fourier_space_from_statevector(
            first_patch_qft_sv_for_viz, n_qubits_viz, ax=axes_fourier[1],
            title=f"QFT Fourier Space (Statevector)\n{n_qubits_viz}-qubit"
        )

        visualize_classical_fft_spectrum(
            first_patch_original_for_viz, ax=axes_fourier[2],
            title=f"Classical FFT Spectrum"
        )
        fig_fourier.suptitle("Fourier Space Comparison of First Processed Patch", fontsize=16)
        fig_fourier.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"{output_prefix}_fourier_space_comparison.png")
        plt.show()

    else:
        print("Could not visualize Fourier space for the first patch (data not collected or error).")

    print("\nProcess Complete.")
