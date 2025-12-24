# backend/quantum_compressor_wrapper.py
import pennylane as qml
from pennylane import numpy as np_pl # Use alias to avoid confusion with standard numpy
from PIL import Image
from scipy.fftpack import dct, idct
import numpy # Standard numpy for some operations if needed
import matplotlib
matplotlib.use('Agg') # For non-interactive plotting
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm 
import math
import os
import argparse
import json
import sys
import traceback

# --- Parameters (will be set by argparse) ---
# IMAGE_PATH, OUTPUT_DIR, BLOCK_SIZE, COEFFICIENT_QUBITS, Q_FACTOR
# POS_QUBITS_PER_DIM, X_POS_QUBITS, Y_POS_QUBITS, AUX_QUBITS, TOTAL_QUBITS

# --- Helper Functions --- (Copied from your script, with minor adjustments)
def image_to_grayscale_matrix(image_path):
    try:
        img = Image.open(image_path).convert('L')
        return np_pl.array(img, dtype=float), img 
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Creating a dummy image.", file=sys.stderr)
        dummy_img_arr = np_pl.zeros((128,128), dtype=float)
        for i in range(128):
            for j in range(128):
                dummy_img_arr[i,j] = (i+j) % 256
        dummy_pil_img = Image.fromarray(dummy_img_arr.astype(numpy.uint8), 'L')
        return dummy_img_arr, dummy_pil_img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}", file=sys.stderr)
        raise

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, q_factor):
    return np_pl.round(block / q_factor)

def dequantize(quantized_block, q_factor):
    return quantized_block * q_factor

def get_non_zero_coeffs(quantized_block):
    coeffs = []
    for r in range(quantized_block.shape[0]):
        for c in range(quantized_block.shape[1]):
            val = int(quantized_block[r, c])
            if val != 0:
                coeffs.append({'value_quantized': val, 'row': r, 'col': c})
    return coeffs

def binary_representation(value, num_bits):
    if not isinstance(value, int):
        value = int(value) # Ensure it's an int for format
    if value < 0:
        # For EFRQI, we typically encode abs(value) and handle sign separately if needed.
        # Here, the script seems to use abs(quant_val) for encoding.
        # The control_values for MultiControlledX expect '0' or '1'.
        # If negative values were to be directly encoded in binary string for control,
        # two's complement or a sign bit would be needed,
        # but current EFRQI examples often focus on magnitude.
        # Let's assume positive for control_values based on typical EFRQI usage.
        raise ValueError("Binary representation for control values expects positive integers or 0/1 strings.")
    return format(value, f'0{num_bits}b')


def calculate_psnr(original_img, compressed_img):
    mse = np_pl.mean((original_img - compressed_img) ** 2)
    if mse == 0: return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def save_image_from_array(image_array, path, original_pil_img=None):
    """Saves a NumPy array as an image file."""
    # Ensure array is in uint8 format for saving standard image types
    if image_array.dtype != numpy.uint8:
        clipped_array = np_pl.clip(image_array, 0, 255)
        image_to_save = Image.fromarray(clipped_array.astype(numpy.uint8), 'L')
    else:
        image_to_save = Image.fromarray(image_array, 'L')
    
    file_format = 'PNG' # Default to PNG for consistency
    path_with_ext = os.path.splitext(path)[0] + '.png'

    try:
        image_to_save.save(path_with_ext, format=file_format)
        print(f"Saved image to: {path_with_ext}")
        return path_with_ext
    except Exception as e:
        print(f"Could not save image to {path_with_ext}. Error: {e}", file=sys.stderr)
        return None

# --- Main Processing ---
def main():
    parser = argparse.ArgumentParser(description="Experimental Quantum I-Frame Compressor (EFRQI based)")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output images and plots.")
    parser.add_argument('--q_factor', type=int, default=30, help="Quantization factor.")
    parser.add_argument('--block_size', type=int, default=8, choices=[4, 8, 16], help="Block size for DCT (e.g., 8 for 8x8).")
    parser.add_argument('--coeff_qubits', type=int, default=8, help="Number of qubits for coefficient magnitude.")
    parser.add_argument('--visualize_first_block', action='store_true', help="Enable detailed visualization for the first non-empty block (can be slow).")

    args = parser.parse_args()

    # --- Parameters from args ---
    IMAGE_PATH = args.input_image
    OUTPUT_DIR = args.output_dir
    BLOCK_SIZE = args.block_size
    COEFFICIENT_QUBITS = args.coeff_qubits
    Q_FACTOR = args.q_factor

    POS_QUBITS_PER_DIM = math.ceil(math.log2(BLOCK_SIZE))
    X_POS_QUBITS = POS_QUBITS_PER_DIM
    Y_POS_QUBITS = POS_QUBITS_PER_DIM
    AUX_QUBITS = 1  # For EFRQI construction
    TOTAL_QUBITS = COEFFICIENT_QUBITS + X_POS_QUBITS + Y_POS_QUBITS + AUX_QUBITS

    results = {
        "status": "processing",
        "parameters": vars(args),
        "output_files": {}
    }
    
    print(f"--- Quantum Compressor Configuration ---")
    print(f"Input image: {IMAGE_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Block size: {BLOCK_SIZE}x{BLOCK_SIZE}")
    print(f"Coefficient qubits: {COEFFICIENT_QUBITS}")
    print(f"X position qubits: {X_POS_QUBITS}")
    print(f"Y position qubits: {Y_POS_QUBITS}")
    print(f"Auxiliary qubits: {AUX_QUBITS}")
    print(f"Total qubits for EFRQI circuit: {TOTAL_QUBITS}")
    print(f"Quantization Factor (Q): {Q_FACTOR}")
    print(f"Visualize first block: {args.visualize_first_block}")
    print(f"--------------------------------------\n")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    try:
        original_image_matrix, original_pil_img = image_to_grayscale_matrix(IMAGE_PATH)
        H, W = original_image_matrix.shape
        print(f"Original image dimensions: {H}x{W}")

        base_image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

        raw_image_filename = f"{base_image_name}_original_grayscale"
        raw_image_path = os.path.join(OUTPUT_DIR, raw_image_filename)
        saved_raw_path = save_image_from_array(original_image_matrix.astype(numpy.uint8), raw_image_path, original_pil_img)
        if saved_raw_path: results["output_files"]["original_grayscale"] = saved_raw_path
        
        padded_H = math.ceil(H / BLOCK_SIZE) * BLOCK_SIZE
        padded_W = math.ceil(W / BLOCK_SIZE) * BLOCK_SIZE
        padded_image_matrix = np_pl.zeros((padded_H, padded_W), dtype=float) # Ensure float for dct
        padded_image_matrix[0:H, 0:W] = original_image_matrix
        reconstructed_image_matrix = np_pl.zeros((padded_H, padded_W), dtype=float)

        # --- EFRQI Quantum Circuit Definition ---
        coeff_wires = list(range(COEFFICIENT_QUBITS))
        aux_wire = COEFFICIENT_QUBITS 
        x_pos_wires = list(range(aux_wire + 1, aux_wire + 1 + X_POS_QUBITS))
        y_pos_wires = list(range(x_pos_wires[-1] + 1, x_pos_wires[-1] + 1 + Y_POS_QUBITS))
        
        dev = qml.device("default.qubit", wires=TOTAL_QUBITS)

        @qml.qnode(dev, interface="autograd") # Autograd for pennylane.numpy
        def efrqi_block_encoder_visual(non_zero_coeffs_list):
            # Initialize auxiliary and position qubits
            qml.PauliX(wires=aux_wire) # Ancilla for control
            for wire in x_pos_wires + y_pos_wires:
                qml.Hadamard(wires=wire) # Superposition for all positions

            # Encode coefficients
            for coeff_info in non_zero_coeffs_list:
                quant_val = coeff_info['value_quantized']
                # Encode absolute value, sign handled classically or by convention
                value_to_encode_coeff_qubits = int(np_pl.clip(abs(quant_val), 0, 2**(COEFFICIENT_QUBITS) - 1))
                row = coeff_info['row']
                col = coeff_info['col']
                
                val_bin_str = binary_representation(value_to_encode_coeff_qubits, COEFFICIENT_QUBITS)
                row_bin_str = binary_representation(row, Y_POS_QUBITS) # Y for rows
                col_bin_str = binary_representation(col, X_POS_QUBITS) # X for columns

                for k in range(COEFFICIENT_QUBITS): # For each bit of the coefficient value
                    if val_bin_str[k] == '1':
                        # Control on aux, y_pos, x_pos to target specific coeff_qubit
                        control_wires_for_mcx = [aux_wire] + y_pos_wires + x_pos_wires
                        control_values_str = "1" + row_bin_str + col_bin_str # MSB for aux
                        
                        target_coeff_qubit = coeff_wires[k]
                        qml.MultiControlledX(wires=control_wires_for_mcx + [target_coeff_qubit],
                                             control_values=control_values_str)
            return qml.probs(wires=coeff_wires) # Not directly used for reconstruction here, but part of EFRQI concept


        num_blocks_processed = 0
        total_non_zero_classical_coeffs = 0
        first_block_processed_for_viz = False

        print(f"\nProcessing image in {BLOCK_SIZE}x{BLOCK_SIZE} blocks...")
        for r_base in range(0, padded_H, BLOCK_SIZE):
            for c_base in range(0, padded_W, BLOCK_SIZE):
                block_original = padded_image_matrix[r_base:r_base+BLOCK_SIZE, c_base:c_base+BLOCK_SIZE]
                dct_block = dct2(block_original)
                quantized_dct_block = quantize(dct_block, Q_FACTOR)
                non_zero_classical = get_non_zero_coeffs(quantized_dct_block)
                total_non_zero_classical_coeffs += len(non_zero_classical)

                if args.visualize_first_block and not first_block_processed_for_viz and len(non_zero_classical) > 0:
                    print(f"\n--- Visualizing First Non-Empty Block Processing (Block at {r_base},{c_base}) ---")
                    fig_dct, axes_dct = plt.subplots(1, 4, figsize=(20, 5))
                    axes_dct[0].imshow(block_original, cmap='gray', vmin=0, vmax=255)
                    axes_dct[0].set_title("Original Block")
                    axes_dct[0].axis('off')
                    norm_dct_coeffs = SymLogNorm(linthresh=0.1, linscale=0.3, base=10, vmin=dct_block.min(), vmax=dct_block.max())
                    im_dct = axes_dct[1].imshow(dct_block, cmap='coolwarm', norm=norm_dct_coeffs)
                    axes_dct[1].set_title("DCT Coefficients")
                    axes_dct[1].axis('off')
                    fig_dct.colorbar(im_dct, ax=axes_dct[1], orientation='vertical', fraction=0.046, pad=0.04)
                    norm_quant_coeffs = SymLogNorm(linthresh=0.1, linscale=0.3, base=10, vmin=quantized_dct_block.min(), vmax=quantized_dct_block.max())
                    im_quant = axes_dct[2].imshow(quantized_dct_block, cmap='coolwarm', norm=norm_quant_coeffs)
                    axes_dct[2].set_title(f"Quantized DCT (Q={Q_FACTOR})")
                    axes_dct[2].axis('off')
                    fig_dct.colorbar(im_quant, ax=axes_dct[2], orientation='vertical', fraction=0.046, pad=0.04)
                    
                    temp_reconstructed_quant_dct_block = np_pl.zeros((BLOCK_SIZE, BLOCK_SIZE))
                    for coeff_info in non_zero_classical:
                         temp_reconstructed_quant_dct_block[coeff_info['row'], coeff_info['col']] = coeff_info['value_quantized']
                    temp_dequantized_block = dequantize(temp_reconstructed_quant_dct_block, Q_FACTOR)
                    temp_idct_block = idct2(temp_dequantized_block)
                    axes_dct[3].imshow(np_pl.clip(temp_idct_block,0,255), cmap='gray', vmin=0, vmax=255)
                    axes_dct[3].set_title("Reconstructed Block")
                    axes_dct[3].axis('off')
                    plt.suptitle(f"DCT Processing for Block at ({r_base},{c_base})", fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    dct_viz_filename = f"{base_image_name}_dct_visualization_block_{r_base}_{c_base}_Q{Q_FACTOR}.png"
                    dct_viz_path = os.path.join(OUTPUT_DIR, dct_viz_filename)
                    plt.savefig(dct_viz_path)
                    results["output_files"]["dct_visualization"] = dct_viz_path
                    print(f"Saved DCT visualization plot to {dct_viz_path}")
                    plt.close(fig_dct)

                    if len(non_zero_classical) > 0:
                        representative_coeff_for_circuit = [non_zero_classical[0]]
                        print(f"\nDrawing EFRQI quantum circuit for encoding ONE representative coefficient:")
                        try:
                            fig_circuit, ax_circuit = qml.draw_mpl(efrqi_block_encoder_visual, decimals=0, style="solarized_light", expansion_strategy="device")(representative_coeff_for_circuit)
                            fig_circuit.suptitle("EFRQI Circuit for One Coefficient", fontsize=14)
                            circuit_viz_filename = f"{base_image_name}_efrqi_circuit_one_coeff_Q{Q_FACTOR}.png"
                            circuit_viz_path = os.path.join(OUTPUT_DIR, circuit_viz_filename)
                            plt.savefig(circuit_viz_path)
                            results["output_files"]["efrqi_circuit_visualization"] = circuit_viz_path
                            print(f"Saved circuit visualization plot to {circuit_viz_path}")
                            plt.close(fig_circuit)
                        except Exception as e_mpl:
                            print(f"  Matplotlib circuit drawing failed: {e_mpl}.", file=sys.stderr)
                    print(f"--- End Visualization for First Non-Empty Block ---")
                    first_block_processed_for_viz = True
                
                # Reconstruct this block for the final image
                reconstructed_quant_dct_block_current = np_pl.zeros((BLOCK_SIZE, BLOCK_SIZE))
                for coeff_info in non_zero_classical:
                    reconstructed_quant_dct_block_current[coeff_info['row'], coeff_info['col']] = coeff_info['value_quantized']
                
                dequantized_block = dequantize(reconstructed_quant_dct_block_current, Q_FACTOR)
                idct_block = idct2(dequantized_block)
                reconstructed_image_matrix[r_base:r_base+BLOCK_SIZE, c_base:c_base+BLOCK_SIZE] = idct_block
                
                num_blocks_processed += 1
                if num_blocks_processed % 200 == 0 and num_blocks_processed > 0 :
                    print(f"Processed {num_blocks_processed} blocks...")
        
        print(f"\nFinished processing {num_blocks_processed} blocks.")

        reconstructed_image_matrix_clipped = np_pl.clip(reconstructed_image_matrix, 0, 255)
        reconstructed_image_final = reconstructed_image_matrix_clipped[0:H, 0:W]
        
        psnr_value = calculate_psnr(original_image_matrix, reconstructed_image_final)
        results["psnr"] = float(psnr_value) # Ensure it's standard float for JSON
        print(f"\nPSNR: {psnr_value:.2f} dB")

        compressed_image_filename = f"{base_image_name}_reconstructed_Q{Q_FACTOR}"
        compressed_image_path = os.path.join(OUTPUT_DIR, compressed_image_filename)
        saved_reconstructed_path = save_image_from_array(reconstructed_image_final.astype(numpy.uint8), compressed_image_path, original_pil_img)
        if saved_reconstructed_path: results["output_files"]["reconstructed_image"] = saved_reconstructed_path

        # Final comparison plot
        fig_final, axes_final = plt.subplots(1, 2, figsize=(12, 6))
        axes_final[0].imshow(original_image_matrix, cmap='gray', vmin=0, vmax=255)
        axes_final[0].set_title("Original Grayscale Image")
        axes_final[0].axis('off')
        axes_final[1].imshow(reconstructed_image_final, cmap='gray', vmin=0, vmax=255)
        axes_final[1].set_title(f"Reconstructed (Simulated EFRQI)\nPSNR: {psnr_value:.2f} dB, Q={Q_FACTOR}")
        axes_final[1].axis('off')
        plt.tight_layout()
        final_comp_filename = f"{base_image_name}_comparison_Q{Q_FACTOR}.png"
        final_comp_path = os.path.join(OUTPUT_DIR, final_comp_filename)
        plt.savefig(final_comp_path)
        results["output_files"]["comparison_plot"] = final_comp_path
        print(f"Saved final comparison plot to {final_comp_path}")
        plt.close(fig_final)

        bits_per_nz_coeff_classical = COEFFICIENT_QUBITS + X_POS_QUBITS + Y_POS_QUBITS
        original_total_bits = H * W * 8 
        compressed_total_bits_classical = total_non_zero_classical_coeffs * bits_per_nz_coeff_classical
        
        results["original_total_bits"] = original_total_bits
        results["compressed_total_bits_classical_data"] = compressed_total_bits_classical
        if compressed_total_bits_classical > 0:
            compression_ratio = original_total_bits / compressed_total_bits_classical
            results["estimated_classical_compression_ratio"] = float(compression_ratio)
            print(f"Original size (est.): {original_total_bits / 8 / 1024:.2f} KB")
            print(f"Compressed size (classical data for quantum encoding, est.): {compressed_total_bits_classical / 8 / 1024:.2f} KB")
            print(f"Estimated Classical Compression Ratio for EFRQI data: {compression_ratio:.2f}:1")
        else:
            results["estimated_classical_compression_ratio"] = "Infinity (no non-zero coeffs)"
            print("No non-zero coefficients after quantization.")
        
        results["status"] = "success"

    except Exception as e:
        print(f"Error during quantum compression processing: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        results["status"] = "error"
        results["error_message"] = str(e)
    finally:
        # Output results as JSON as the last line to stdout
        sys.stdout.flush() # Ensure all prior prints are done
        sys.stderr.flush()
        print(json.dumps(results))


if __name__ == "__main__":
    main()