# Deep Learned Video Compression with RAFT and Hyperprior

This repository implements a learned video compression framework leveraging **RAFT** (Recurrent All-Pairs Field Transforms) for high-quality optical flow estimation and **Hyperprior Autoencoders** for efficient entropy coding of motion and residuals. The system includes a custom codec for encoding videos into a compact `.rdvc` format and decoding them back with high fidelity.


full report: [PFE Report](https://www.researchgate.net/publication/392626523_Recurrent_All-Pairs_Field_Transform_RAFT_Hyperprior_Entropy_Coding_and_Quantum_I-Frame_Encoding_for_Video_Compression?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InByb2ZpbGUiLCJwYWdlIjoicHJvZmlsZSJ9fQ)
##  Features

- **Learned Optical Flow**: Uses RAFT to estimate motion between frames, enabling effective motion compensation.
- **End-to-End Compression**:
  - **Motion Autoencoder**: Compresses the optical flow maps using a hyperprior architecture.
  - **Residual Autoencoder**: Compresses the residual errors (difference between predicted and actual frames).
- **Multi-Phase Training**: A robust 3-phase training strategy to stabilize convergence:
  1. **Phase 1**: Train Residual Autoencoder using Ground Truth flow.
  2. **Phase 2**: End-to-end training with reconstructed flow.
  3. **Phase 3**: Fine-tuning with perceptual loss (MS-SSIM) and BPP (Bits Per Pixel) constraints.
- **Custom File Format**: Encodes videos into `.rdvc` (Raw Deep Video Compression) files.
- **Visualizations**: Comprehensive logging and plotting of metrics (PSNR, MS-SSIM, Loss) and intermediate feature maps.

##  Project Structure

- `newcodec.py`: Core implementation of the `VideoCodec` class, including Encoders, Decoders, Warping layers, and Entropy Bottlenecks.
- `new_train.py`: The main training script implementing the 3-phase training loop utilizing `CompressAI`.
- `app.py`: Application entry point (inference/demo).
- `codec_checkpoints_*`: Directories storing model checkpoints.
- `training_plots/`: Stores metric plots generated during training.
- `visualization_*/`: Stores qualitative results (reconstructed frames, flow maps, residuals).
- `benchmark/`: Contains performance graphs, RD curves, and architecture diagrams.

##  Results & Visualizations

### Qualitative Reconstruction
Comparison between the original frame and the reconstructed frame after compression/decompression.

| Original Frame | Reconstructed Frame |
|:---:|:---:|
| ![Original](two_frame_output_histmatch/im2_original_for_comparison.png) | ![Reconstructed](two_frame_output_histmatch/im2_reconstructed_hist_matched.png) |

**Detailed Reconstruction View:**
![Real vs Reconstructed](benchmark/realvsreconstructed.png)

### Internal Representations
The model explicitly handles motion and residuals. Below are the reconstructed optical flow and residual maps.

| Reconstructed Optical Flow | Reconstructed Residual |
|:---:|:---:|
| ![Flow](two_frame_output_histmatch/flow_reconstructed_final.png) | ![Residual](two_frame_output_histmatch/residual_reconstructed_final.png) |

**RAFT Flow Prediction Sample:**
![RAFT Prediction](benchmark/raftprediction.png)

### Benchmarks & Performance
We evaluate the codec's performance using Rate-Distortion (RD) curves and resolution-specific metrics.

**Rate-Distortion Curves:**
| PSNR vs Bitrate | MS-SSIM vs Bitrate |
|:---:|:---:|
| ![RD Curve PSNR](benchmark/rd_curve_psnr.png) | ![RD Curve MS-SSIM](benchmark/rd_curve_msssim.png) |

**Resolution Analysis:**
| PSNR by Resolution | MS-SSIM by Resolution |
|:---:|:---:|
| ![PSNR Resolution](benchmark/psnr_by_resolution.png) | ![MS-SSIM Resolution](benchmark/msssim_by_resolution.png) |

### Training Progression
Visualizing the model's output at the latest training epoch (Phase 3). This includes the motion compensation output and final reconstruction.

![Epoch Visualization](codec_visualizations_3phase_resAE_vis/epoch_0115_phase3_vis.png)

### Training Metrics
Performance metrics tracked over training phases.

| Phase 1 Metrics | Phase 2 Metrics |
|:---:|:---:|
| ![Phase 1](training_plots/phase_1_metrics.png) | ![Phase 2](training_plots/phase_2_metrics.png) |

**Phase 3 Optimization Metrics:**
| MS-SSIM | Loss |
|:---:|:---:|
| ![MS-SSIM](visualization_phase3/Avg_MS_SSIM__opt_.png) | ![Loss](visualization_phase3/Avg_Loss.png) |

## 🛠 Usage

### Prerequisites
- Python 3.x
- PyTorch
- `compressai`
- `torchvision`
- `numpy`, `opencv-python`, `pillow`, `tqdm`

### Training
To start the training process (ensure data paths are configured in `new_train.py`):
```bash
python new_train.py
```

### Encoding & Decoding
The `newcodec.py` script provides a command-line interface for encoding and decoding videos.

**Encode a video:**
```bash
python newcodec.py --encode --input_file input.mp4 --output_file compressed.rdvc
```

**Decode a video:**
```bash
python newcodec.py --decode --input_file compressed.rdvc --output_file output.mp4
```

##  Model Architecture & Workflows

The `VideoCodec` consists of two main branches:
1. **Motion Branch**:
   - Takes optical flow (from RAFT) as input.
   - Compresses it into a latent representation.
   - Reconstructs flow for warping the previous frame.
2. **Residual Branch**:
   - Computes the difference between the current frame and the motion-compensated previous frame.
   - Compresses this residual using a separate Deep Autoencoder with Entropy Bottleneck.

Final reconstruction is obtained by adding the decoded residual to the motion-compensated frame.

### System Diagrams
**Video Encoding Workflow:**
![Encoding Process](benchmark/sequenceuml%20encoding%20process.png)

**Video Decoding Workflow:**
![Decoding Process](benchmark/sequenceuml%20decoding%20process.png)

### Core Components
**RAFT Optical Flow Architecture:**
![RAFT Architecture](benchmark/raftarchitecture.png)

**Hyperprior Entropy Model:**
![Hyperprior Graph](benchmark/hyperpriorcomponent%20graph.png)

### Training & Filtering
**RAFT Training Dynamics:**
![RAFT Training](benchmark/rafttraining.png)

**Temporal Filtering:**
![First Order IIR Filtering](benchmark/first%20order%20IIR%20temporal%20filtering.png)

### Detailed System Diagrams
**Detailed Encoding Block Diagram:**
![RDVC Encoding](benchmark/RDVCencoding.png)

**Detailed Decoding Block Diagram:**
![RDVC Decoding](benchmark/rdvcdecoding.png)

---
