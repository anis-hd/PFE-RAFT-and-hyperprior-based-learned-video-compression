# RDVC: Raft Deep Video Compression

**A Hybrid Video Compression Framework combining RAFT Optical Flow, Hyperprior Entropy Coding, and Quantum-Inspired I-Frame Encoding (Not included in this repo).**


## Context

This repo implements **RDVC (Raft Deep Video Compression)**, a Final Year Project developed at **Talan Tunisie**  in collaboration with the **National School of Computer Science (ENSI)** .

The system addresses the limitations of traditional codecs by proposing a deep learning-based inter-frame compression pipeline. It integrates **RAFT** (Recurrent All-Pairs Field Transforms) for optical flow estimation and **Hyperprior Autoencoders** for entropy coding. Additionally, the project explores **Quantum Computing** simulations for I-frame compression. (NOT INCLUDED IN THIS REPOSITORY)

## Key Features

- **Learned Optical Flow (RAFT)**: Uses the RAFT architecture trained on MPI Sintel to estimate dense motion between frames, enabling effective motion compensation.
- **Motion Compensation Network (MCN)**: Refines warped frames to correct artifacts derived from occlusion or flow errors.
- **End-to-End Compression**:
  - **Motion Autoencoder**: Compresses optical flow maps using a hyperprior architecture.
  - **Residual Autoencoder**: Compresses residual errors (difference between predicted and actual frames).
- **Multi-Phase Training**: A strategy used to stabilize convergence and compensate for the low performance of the training hardware (My personal laptop):
  1.  **Phase 1**: Train Residual AE using Ground Truth flow (MSE loss).
  2.  **Phase 2**: End-to-end training with reconstructed flow.
  3.  **Phase 3**: Fine-tuning with perceptual loss (MS-SSIM) and Bitrate constraints.
- **Post-Processing**: Includes Low Motion Region replacement, Histogram Matching, and Temporal IIR Filtering to compensate for compression artifacts.
- **Custom File Format**: Encodes videos into `.rdvc` (RAFT Deep Video Compression) files.

## System Architecture & Workflows

The RDVC codec operates on a P-frame architecture involving Motion and Residual branches.

### High-Level Schematics
**Detailed Encoding Block Diagram:**
![RDVC Encoding](benchmark/RDVCencoding.png)

**Detailed Decoding Block Diagram:**
![RDVC Decoding](benchmark/rdvcdecoding.png)

### Encoding & Decoding Processes
The system uses a synchronized loop to ensure the encoder and decoder states remain identical.

**Video Encoding Workflow:**
![Encoding Process](benchmark/sequenceuml%20encoding%20process.png)

**Video Decoding Workflow:**
![Decoding Process](benchmark/sequenceuml%20decoding%20process.png)

### Core Components
**RAFT Optical Flow Architecture:**
![RAFT Architecture](benchmark/raftarchitecture.png)

**Hyperprior Entropy Model:**
![Hyperprior Graph](benchmark/hyperpriorcomponent%20graph.png)

## Training & Optimization

### Training Dynamics
**RAFT Training:**
![RAFT Training](benchmark/rafttraining.png)

**Phase 1 Metrics (Initialization):**
![Phase 1](training_plots/phase_1_metrics.png)

**Phase 2 Metrics (End-to-End Loop):**
![Phase 2](training_plots/phase_2_metrics.png)

**Phase 3 Optimization (Perceptual Tuning):**
| MS-SSIM Optimization | Loss Convergence |
|:---:|:---:|
| ![MS-SSIM](visualization_phase3/Avg_MS_SSIM__opt_.png) | ![Loss](visualization_phase3/Avg_Loss.png) |

### Training Progression Visualization
Visualizing the model's output at the latest training epoch (Phase 3). This includes the motion compensation output and final reconstruction.

![Epoch Visualization](codec_visualizations_3phase_resAE_vis/epoch_0115_phase3_vis.png)

## Results & Evaluation

The system was evaluated on the **UVG Dataset** (Beauty, Jockey, ReadySetGo).

### Qualitative Reconstruction
Comparison between the original frame and the reconstructed frame after compression/decompression.

| Original Frame | Reconstructed Frame |
|:---:|:---:|
| ![Original](two_frame_output_histmatch/im2_original_for_comparison.png) | ![Reconstructed](two_frame_output_histmatch/im2_reconstructed_hist_matched.png) |


### Internal Representations
The model explicitly handles motion and residuals. Below are the reconstructed optical flow and residual maps.

| Reconstructed Optical Flow | Reconstructed Residual |
|:---:|:---:|
| ![Flow](two_frame_output_histmatch/flow_reconstructed_final.png) | ![Residual](two_frame_output_histmatch/residual_reconstructed_final.png) |

**RAFT Flow Prediction Sample:**
![RAFT Prediction](benchmark/raftprediction.png)

### Benchmarks
**Rate-Distortion Curves:**
| PSNR vs Bitrate | MS-SSIM vs Bitrate |
|:---:|:---:|
| ![RD Curve PSNR](benchmark/rd_curve_psnr.png) | ![RD Curve MS-SSIM](benchmark/rd_curve_msssim.png) |

**Resolution Analysis:**
| PSNR by Resolution | MS-SSIM by Resolution |
|:---:|:---:|
| ![PSNR Resolution](benchmark/psnr_by_resolution.png) | ![MS-SSIM Resolution](benchmark/msssim_by_resolution.png) |

### Post-Processing
**Temporal Filtering (First Order IIR):**
![First Order IIR Filtering](benchmark/first%20order%20IIR%20temporal%20filtering.png)

## Usage

### Prerequisites
- Python 3.11
- PyTorch (preferrably with CUDA enabled for GPU utilization)
- `compressai`
- `torchvision`, `numpy`, `opencv-python`, `pillow`, `tqdm`

### Training
To start the training process (ensure data paths are configured in `config`):
```bash
python new_train.py
```

### Encoding & Decoding
The `codec_processing.py` script provides a command-line interface for encoding and decoding videos.

**Encode a video:**
```bash
python codec_processing.py --encode --input_file input.yuv --output_file compressed.rdvc
```

**Decode a video:**
```bash
python newcodec.py --decode --input_file compressed.rdvc --output_file output.mp4
```

## Project Structure

- `codec_processing.py`: Core implementation of the `VideoCodec` class, including Encoders, Decoders, Warping layers, and Entropy Bottlenecks.
- `new_train.py`: The main training script implementing the 3-phase training loop utilizing `CompressAI`.
- `codec_checkpoints_*`: Directories storing model checkpoints.
- `training_plots/`: Stores metric plots generated during training.
- `visualization_*/`: Stores qualitative results (reconstructed frames, flow maps, residuals).
- `benchmark/`: Contains performance graphs, RD curves, and architecture diagrams.

If you want to train the models from scratch, you can use a pretrained RAFT model provided by the TorchVision library or refer to my older repository [RAFT](https://github.com/anis-hd/RAFT_motion_estimation) to train a RAFT model from scratch on the MPI Sintel dataset.

Always happy to collaborate.
