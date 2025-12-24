import matplotlib.pyplot as plt
import numpy as np

# Hardcoded data (same as before)
data = {
    "Beauty": {
        "PSNR":    [30.31, 29.65, 28.80, 28.79, 27.57, 26.60],
        "MS-SSIM": [0.8775, 0.8655, 0.8464, 0.8459, 0.8182, 0.8075],
        "BPP":     [0.5151, 0.2809, 0.1959, 0.1678, 0.1485, 0.1386],
        "Bitrate": [32043.22, 17476.97, 12186.59, 10436.54, 9238.79, 8618.99]
    },
    "ReadySetGo": { # Corrected from "ReasySetGo"
        "PSNR":    [27.87, 27.09, 24.36, 25.10, 22.10, 20.98],
        "MS-SSIM": [0.9365, 0.9219, 0.8541, 0.8721, 0.7458, 0.6720],
        "BPP":     [0.6180, 0.3855, 0.2862, 0.2487, 0.2191, 0.2014],
        "Bitrate": [38442.38, 23983.44, 17801.85, 15470.81, 13626.86, 12531.56]
    },
    "Jockey": {
        "PSNR":    [29.13, 28.61, 26.82, 27.42, 23.70, 21.97],
        "MS-SSIM": [0.9046, 0.8847, 0.8388, 0.8423, 0.7357, 0.6673],
        "BPP":     [1.0363, 0.4180, 0.2497, 0.1977, 0.1603, 0.1394],
        "Bitrate": [64464.15, 26002.48, 15533.90, 12298.39, 9972.06, 8669.84]
    }
}

resolutions_labels = ["1080p", "720p", "480p", "360p", "240p", "140p"] # For annotating points if needed
video_names = ["Beauty", "ReadySetGo", "Jockey"]
colors = ['blue', 'green', 'red']

# --- Plotting Individual Metrics vs. Resolution (as before) ---
metrics_to_plot_vs_resolution = {
    "PSNR": "Average PSNR (dB)",
    "MS-SSIM": "Average MS-SSIM",
    "BPP": "Average BPP (bits per pixel)",
    "Bitrate": "Bitrate (kbps)"
}

print("--- Plotting Individual Metrics vs. Resolution ---")
for metric_key, metric_label in metrics_to_plot_vs_resolution.items():
    plt.figure(figsize=(10, 6))
    for i, video_name in enumerate(video_names):
        values = data[video_name][metric_key]
        plt.plot(resolutions_labels, values, marker='o', linestyle='-', color=colors[i], label=video_name)
    plt.title(f'{metric_label} vs. Resolution')
    plt.xlabel('Resolution (Residual and Motion Flow)')
    plt.ylabel(metric_label)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Plotting Rate-Distortion Curves ---
print("\n--- Plotting Rate-Distortion Curves ---")

# RD Plot 1: PSNR vs. Bitrate
plt.figure(figsize=(10, 6))
for i, video_name in enumerate(video_names):
    bitrates = data[video_name]["Bitrate"]
    psnrs = data[video_name]["PSNR"]
    # Sort by bitrate for a proper RD curve (optional if data is already ordered by rate for each resolution)
    # sorted_pairs = sorted(zip(bitrates, psnrs))
    # sorted_bitrates = [p[0] for p in sorted_pairs]
    # sorted_psnrs = [p[1] for p in sorted_pairs]
    # plt.plot(sorted_bitrates, sorted_psnrs, marker='o', linestyle='-', color=colors[i], label=video_name)
    
    # Given the data is already ordered by resolution (which implies an order for bitrate),
    # direct plotting is fine and shows the progression through resolutions.
    plt.plot(bitrates, psnrs, marker='o', linestyle='-', color=colors[i], label=video_name)
    # Optionally, annotate points with resolution
    for j, txt in enumerate(resolutions_labels):
        plt.annotate(txt.replace("p",""), (bitrates[j], psnrs[j]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, alpha=0.6)


plt.title('Rate-Distortion: PSNR vs. Bitrate')
plt.xlabel('Bitrate (kbps)')
plt.ylabel('Average PSNR (dB)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# RD Plot 2: MS-SSIM vs. BPP
plt.figure(figsize=(10, 6))
for i, video_name in enumerate(video_names):
    bpps = data[video_name]["BPP"]
    ms_ssims = data[video_name]["MS-SSIM"]
    # Similar sorting logic could be applied if needed, but direct plot shows resolution steps
    plt.plot(bpps, ms_ssims, marker='o', linestyle='-', color=colors[i], label=video_name)
    # Optionally, annotate points with resolution
    for j, txt in enumerate(resolutions_labels):
        plt.annotate(txt.replace("p",""), (bpps[j], ms_ssims[j]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, alpha=0.6)

plt.title('Rate-Distortion: MS-SSIM vs. BPP')
plt.xlabel('Average BPP (bits per pixel)')
plt.ylabel('Average MS-SSIM')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nVisualization complete. Check the displayed plots.")