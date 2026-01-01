
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import qiskit.quantum_info
import os
import math
from tqdm import tqdm
from scipy.fft import fftshift

# Function to encode the image onto a quantum state
def encode_image(image_path, n_qubits_for_state, original_image_width, original_image_height):
    im = Image.open(image_path, 'r')
    im = im.convert("L") # grayscale
    if im.size[0] != original_image_width or im.size[1] != original_image_height:
        pass

    pixel_value_list = list(im.getdata())
    pixel_value_array_unpadded = np.array(pixel_value_list, dtype=np.float64)

    num_actual_pixels = len(pixel_value_array_unpadded)
    required_state_vector_len = 2**n_qubits_for_state

    if num_actual_pixels > required_state_vector_len:
        raise ValueError(f"Image patch has {num_actual_pixels} pixels, which is more than the {required_state_vector_len} states available for {n_qubits_for_state} qubits.")

    padded_pixel_array = np.zeros(required_state_vector_len, dtype=np.float64)
    padded_pixel_array[:num_actual_pixels] = pixel_value_array_unpadded

    norm = np.linalg.norm(padded_pixel_array)
    if np.isclose(norm, 0):
        normalized_array = np.zeros(required_state_vector_len, dtype=np.complex128)
        if len(normalized_array) > 0:
            normalized_array[0] = 1.0
    else:
        normalized_array = padded_pixel_array / norm

    qc = QuantumCircuit(n_qubits_for_state)
    qc.initialize(normalized_array, qc.qubits)
    return qc, norm, pixel_value_array_unpadded

# Function to apply quantum fourier transform
def apply_qft(qc, n_qubits):
    qc.append(QFT(n_qubits, do_swaps=False, name="qft"), qc.qubits)
    return qc

# Function to get QFT statevector
def get_qft_statevector(qc_input_circuit):
    if qc_input_circuit.num_qubits == 0:
        raise ValueError("Input circuit for statevector simulation has 0 qubits.")
    if any(instruction.operation.name == 'measure' for instruction in qc_input_circuit.data):
        raise ValueError("Cannot run statevector simulation on a circuit with measurements.")
    qc_for_sim_main = qc_input_circuit.copy()
    qc_for_sim_main.save_statevector(label="main_final_statevec")
    sv_sim_main = AerSimulator(method='statevector')
    t_qc_main = transpile(qc_for_sim_main, sv_sim_main)
    job_main = sv_sim_main.run(t_qc_main)
    result_main = job_main.result()
    if not result_main.success:
        job_error_msg_final = job_main.error_message() if hasattr(job_main, 'error_message') and job_main.error_message() else "N/A"
        raise qiskit.QiskitError(f"Statevector simulation job reported failure. Error: {job_error_msg_final}")
    try:
        statevector_data_main = result_main.data(0)['main_final_statevec']
        if isinstance(statevector_data_main, np.ndarray):
            statevector_obj = qiskit.quantum_info.Statevector(statevector_data_main)
        elif isinstance(statevector_data_main, qiskit.quantum_info.Statevector):
            statevector_obj = statevector_data_main
        else:
            raise TypeError(f"Unexpected type from save_statevector: {type(statevector_data_main)}")
    except KeyError:
        raise qiskit.QiskitError("Failed to retrieve statevector using 'main_final_statevec' label.")
    return statevector_obj

# Function to threshold QFT coefficients
def threshold_qft_coefficients(qft_coeffs_vector_obj, keep_n_coeffs=None):
    qft_coeffs_numpy = qft_coeffs_vector_obj.data
    magnitudes = np.abs(qft_coeffs_numpy)
    thresholded_coeffs_numpy = qft_coeffs_numpy.copy()
    num_total_coeffs = len(magnitudes)

    if keep_n_coeffs is None or keep_n_coeffs <= 0 or keep_n_coeffs >= num_total_coeffs:
        norm_current = np.linalg.norm(thresholded_coeffs_numpy)
        if np.isclose(norm_current, 0.0):
            zero_state = np.zeros_like(thresholded_coeffs_numpy, dtype=np.complex128)
            if len(zero_state) > 0: zero_state[0] = 1.0
            else: return qiskit.quantum_info.Statevector(np.array([1.0], dtype=np.complex128)), 0
            return qiskit.quantum_info.Statevector(zero_state), 0
        return qiskit.quantum_info.Statevector(thresholded_coeffs_numpy / norm_current if norm_current != 0 else thresholded_coeffs_numpy), num_total_coeffs

    indices_to_keep = np.argsort(magnitudes)[-keep_n_coeffs:]
    thresholded_coeffs_numpy_temp = np.zeros_like(thresholded_coeffs_numpy, dtype=np.complex128)
    thresholded_coeffs_numpy_temp[indices_to_keep] = thresholded_coeffs_numpy[indices_to_keep]
    thresholded_coeffs_numpy = thresholded_coeffs_numpy_temp
    num_kept = len(indices_to_keep)
    norm_thresholded = np.linalg.norm(thresholded_coeffs_numpy)

    if norm_thresholded > 1e-9:
        thresholded_coeffs_normalized_numpy = thresholded_coeffs_numpy / norm_thresholded
    else:
        thresholded_coeffs_normalized_numpy = np.zeros_like(thresholded_coeffs_numpy, dtype=np.complex128)
        if len(thresholded_coeffs_normalized_numpy) > 0:
            thresholded_coeffs_normalized_numpy[0] = 1.0
            num_kept = 1 if len(thresholded_coeffs_normalized_numpy) > 0 else 0
        else:
             thresholded_coeffs_normalized_numpy = np.array([1.0], dtype=np.complex128)
             num_kept = 0
    return qiskit.quantum_info.Statevector(thresholded_coeffs_normalized_numpy), num_kept





# Function to apply IQFT and reconstruct image
def apply_iqft_and_reconstruct(thresholded_qft_coeffs_statevector, n_qubits,
                               target_image_width, target_image_height,
                               original_padded_norm, shots):
    qc_iqft = QuantumCircuit(n_qubits)
    qc_iqft.initialize(thresholded_qft_coeffs_statevector, qc_iqft.qubits)
    qc_iqft.append(QFT(n_qubits, inverse=True, do_swaps=False, name="iqft"), qc_iqft.qubits)
    qc_iqft.measure_all()
    aer_sim = AerSimulator()
    t_qc_iqft = transpile(qc_iqft, aer_sim)
    result = aer_sim.run(t_qc_iqft, shots=shots).result()
    counts = result.get_counts(qc_iqft)
    num_states_total = 2**n_qubits
    reconstructed_amplitudes_from_probs = np.zeros(num_states_total, dtype=float)

    for i in range(num_states_total):
        bin_repr = format(i, f'0{n_qubits}b')
        if bin_repr in counts:
            reconstructed_amplitudes_from_probs[i] = np.sqrt(counts[bin_repr] / shots)

    denormalized_full_vector = reconstructed_amplitudes_from_probs * original_padded_norm
    num_actual_pixels = target_image_width * target_image_height
    denormalized_pixels_image_part = denormalized_full_vector[:num_actual_pixels]

    reconstructed_pixel_values = np.clip(denormalized_pixels_image_part, 0, 255).astype(np.uint8)
    picture_array = np.zeros((target_image_height, target_image_width), dtype=np.uint8)
    if num_actual_pixels > 0:
        picture_array = reconstructed_pixel_values.reshape((target_image_height, target_image_width))
    return picture_array





def visualize_qft_fourier_space_from_statevector(qft_statevector_obj, n_qubits, ax=None, title="QFT Fourier Space (Statevector)"):
    qft_coeffs_numpy = qft_statevector_obj.data
    magnitudes = np.abs(qft_coeffs_numpy)
    log_magnitudes = np.log1p(magnitudes * 100)
    num_states_total = 2**n_qubits
    qft_mag_image_dim_h = int(2**(n_qubits // 2))
    qft_mag_image_dim_w = int(2**(n_qubits - (n_qubits // 2)))
    if num_states_total == 0: return np.array([[]])
    reshaped_magnitudes = log_magnitudes.reshape((qft_mag_image_dim_h, qft_mag_image_dim_w))
    shifted_magnitudes = fftshift(reshaped_magnitudes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(shifted_magnitudes, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label="Log Magnitude (Statevector)")
    if ax is None:
        plt.show()
    return shifted_magnitudes


def qft_compress_full_image_patched(original_pil_image_gray,
                                    patch_width, patch_height,
                                    coeffs_percentage_to_keep,
                                    shots,
                                    temp_patch_filename_prefix="temp_qft_patch",
                                    return_first_patch_data=False,
                                    progress_desc="QFT Processing Patches"):
    n_qubits_per_patch = int(np.ceil(np.log2(patch_width * patch_height)))
    num_total_qft_coeffs_per_patch = 2**n_qubits_per_patch
    coeffs_to_keep_val = max(1, int(coeffs_percentage_to_keep * num_total_qft_coeffs_per_patch))

    img_width, img_height = original_pil_image_gray.size
    reconstructed_large_img_array = np.zeros((img_height, img_width), dtype=np.uint8)

    num_patches_x = math.ceil(img_width / patch_width)
    num_patches_y = math.ceil(img_height / patch_height)

    first_patch_qft_sv = None
    first_patch_original_arr = None
    temp_patch_file = f"{temp_patch_filename_prefix}_{os.getpid()}.png"

    patch_coords = [(i, j) for j in range(num_patches_y) for i in range(num_patches_x)]

    for patch_idx, (i, j) in enumerate(tqdm(patch_coords, desc=progress_desc, unit="patch")):
        x0, y0 = i * patch_width, j * patch_height
        x1, y1 = min(x0 + patch_width, img_width), min(y0 + patch_height, img_height)
        current_patch_pil = original_pil_image_gray.crop((x0, y0, x1, y1))
        actual_w, actual_h = current_patch_pil.size

        padded_patch_for_processing_pil = Image.new("L", (patch_width, patch_height), color=0)
        padded_patch_for_processing_pil.paste(current_patch_pil, (0,0))
        padded_patch_for_processing_pil.save(temp_patch_file)

        qc_encoded, patch_norm, _ = encode_image(
            temp_patch_file, n_qubits_per_patch, patch_width, patch_height
        )
        qc_with_qft = apply_qft(qc_encoded.copy(), n_qubits_per_patch)
        qft_statevector = get_qft_statevector(qc_with_qft)

        if return_first_patch_data and patch_idx == 0:
            first_patch_qft_sv = qft_statevector.copy()
            first_patch_original_arr = np.array(padded_patch_for_processing_pil)

        thresholded_coeffs_sv, _ = threshold_qft_coefficients(
            qft_statevector, keep_n_coeffs=coeffs_to_keep_val
        )
        reconstructed_patch_array_padded = apply_iqft_and_reconstruct(
            thresholded_coeffs_sv, n_qubits_per_patch,
            patch_width, patch_height, patch_norm, shots=shots
        )
        reconstructed_patch_to_paste = reconstructed_patch_array_padded[:actual_h, :actual_w]
        reconstructed_large_img_array[y0:y1, x0:x1] = reconstructed_patch_to_paste

    if os.path.exists(temp_patch_file):
        os.remove(temp_patch_file)
    if return_first_patch_data:
        return reconstructed_large_img_array, (first_patch_qft_sv, first_patch_original_arr)
    return reconstructed_large_img_array
