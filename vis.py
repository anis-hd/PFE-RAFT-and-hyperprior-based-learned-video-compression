import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def parse_log_file(filepath):
    """
    Parses the log file to extract epoch-wise data for each phase.
    """
    data_by_phase = defaultdict(list)
    current_epoch_data = {}
    current_phase = None
    current_epoch_num = None

    # Regex patterns
    start_epoch_pattern = re.compile(r"^-+ Starting Epoch (\d+)/\d+ \(Phase (\d+) .*\) -+$")
    summary_time_pattern = re.compile(r"Epoch \d+/\d+ Summary .* \| Time: ([\d.]+)s")
    loss_mse_pattern = re.compile(r"Avg Loss: ([\d.]+) \| MSE_Flow: ([\d.]+) \| MSE_Res: ([\d.]+)")
    bpp_pattern = re.compile(r"Avg BPP_M\(est\): ([\d.]+) \| BPP_R\(est\): ([\d.]+)")

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                match_start_epoch = start_epoch_pattern.search(line)
                if match_start_epoch:
                    # If we were processing a previous epoch, try to save it
                    # This happens if a new "Starting Epoch" line is found before the old one was "closed"
                    if current_epoch_num is not None and current_phase is not None and \
                       all(k in current_epoch_data for k in ['time', 'avg_loss', 'mse_flow', 'mse_res', 'bpp_m_est', 'bpp_r_est']):
                        data_by_phase[current_phase].append(current_epoch_data)

                    # Start new epoch processing
                    current_epoch_num = int(match_start_epoch.group(1))
                    current_phase = int(match_start_epoch.group(2))
                    current_epoch_data = {'epoch': current_epoch_num}
                    # print(f"DEBUG: Starting Epoch {current_epoch_num}, Phase {current_phase}")
                    continue

                if current_epoch_num is None: # Skip lines if not inside an epoch block
                    continue

                match_summary_time = summary_time_pattern.search(line)
                if match_summary_time:
                    current_epoch_data['time'] = float(match_summary_time.group(1))
                    # print(f"DEBUG: Found time {current_epoch_data['time']} for epoch {current_epoch_num}")
                    continue

                match_loss_mse = loss_mse_pattern.search(line)
                if match_loss_mse:
                    current_epoch_data['avg_loss'] = float(match_loss_mse.group(1))
                    current_epoch_data['mse_flow'] = float(match_loss_mse.group(2))
                    current_epoch_data['mse_res'] = float(match_loss_mse.group(3))
                    # print(f"DEBUG: Found Loss/MSE for epoch {current_epoch_num}")
                    continue

                match_bpp = bpp_pattern.search(line)
                if match_bpp:
                    current_epoch_data['bpp_m_est'] = float(match_bpp.group(1))
                    current_epoch_data['bpp_r_est'] = float(match_bpp.group(2))
                    # print(f"DEBUG: Found BPPs for epoch {current_epoch_num}")

                    # This is often the last piece of summary data.
                    # Check if all expected keys are present and then store.
                    if all(k in current_epoch_data for k in ['time', 'avg_loss', 'mse_flow', 'mse_res', 'bpp_m_est', 'bpp_r_est']):
                        data_by_phase[current_phase].append(current_epoch_data)
                        # print(f"DEBUG: Stored data for Epoch {current_epoch_num}, Phase {current_phase}: {current_epoch_data}")
                        # Reset for next potential epoch data (though new 'Starting Epoch' will also reset)
                        current_epoch_data = {} # Prepare for a new set or end of file
                        # current_epoch_num = None # Don't reset current_epoch_num here, wait for next "Starting Epoch"
                    continue

            # After loop, check if there's any pending data for the last epoch
            if current_epoch_num is not None and current_phase is not None and \
               'epoch' in current_epoch_data and \
               all(k in current_epoch_data for k in ['time', 'avg_loss', 'mse_flow', 'mse_res', 'bpp_m_est', 'bpp_r_est']):
                data_by_phase[current_phase].append(current_epoch_data)
                # print(f"DEBUG: Stored final data for Epoch {current_epoch_num}, Phase {current_phase}: {current_epoch_data}")


    except FileNotFoundError:
        print(f"Error: Log file '{filepath}' not found.")
        return None
    
    # Sort data within each phase by epoch number
    for phase in data_by_phase:
        data_by_phase[phase] = sorted(data_by_phase[phase], key=lambda x: x['epoch'])
        
    return data_by_phase

def plot_phase_data(phase_num, epoch_data_list, output_dir="plots"):
    """
    Generates and saves plots for a single phase.
    """
    if not epoch_data_list:
        print(f"No data to plot for Phase {phase_num}.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = [d['epoch'] for d in epoch_data_list]
    avg_loss = [d['avg_loss'] for d in epoch_data_list]
    mse_flow = [d['mse_flow'] for d in epoch_data_list]
    mse_res = [d['mse_res'] for d in epoch_data_list]
    bpp_m_est = [d['bpp_m_est'] for d in epoch_data_list]
    bpp_r_est = [d['bpp_r_est'] for d in epoch_data_list]
    time_s = [d['time'] for d in epoch_data_list]

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18)) # 3 rows, 2 columns
    fig.suptitle(f'Training Metrics for Phase {phase_num}', fontsize=16)

    # Avg Loss
    axs[0, 0].plot(epochs, avg_loss, marker='o', linestyle='-', label='Avg Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Avg Loss')
    axs[0, 0].set_title('Average Loss vs. Epoch')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # MSE Flow
    axs[0, 1].plot(epochs, mse_flow, marker='o', linestyle='-', color='green', label='MSE Flow')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('MSE Flow')
    axs[0, 1].set_title('MSE Flow vs. Epoch')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # MSE Residual
    axs[1, 0].plot(epochs, mse_res, marker='o', linestyle='-', color='red', label='MSE Residual')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('MSE Residual')
    axs[1, 0].set_title('MSE Residual vs. Epoch')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Avg BPP_M(est)
    axs[1, 1].plot(epochs, bpp_m_est, marker='o', linestyle='-', color='purple', label='Avg BPP_M(est)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Avg BPP_M(est)')
    axs[1, 1].set_title('Avg BPP_M(est) vs. Epoch')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # BPP_R(est)
    axs[2, 0].plot(epochs, bpp_r_est, marker='o', linestyle='-', color='orange', label='BPP_R(est)')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('BPP_R(est)')
    axs[2, 0].set_title('BPP_R(est) vs. Epoch')
    axs[2, 0].grid(True)
    axs[2, 0].legend()
    
    # Time per Epoch
    axs[2, 1].plot(epochs, time_s, marker='o', linestyle='-', color='brown', label='Time (s)')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Time (s)')
    axs[2, 1].set_title('Time per Epoch')
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    
    plot_filename = os.path.join(output_dir, f'phase_{phase_num}_metrics.png')
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")
    plt.close(fig) # Close the figure to free memory

# --- Main execution ---
if __name__ == "__main__":
    log_file = 'training_log_3phase_clean.txt'
    parsed_data = parse_log_file(log_file)

    if parsed_data:
        print(f"\nFound data for {len(parsed_data)} phases: {list(parsed_data.keys())}")
        for phase_num, data_list in parsed_data.items():
            print(f"\n--- Phase {phase_num} ---")
            if data_list:
                # Print first entry for brevity
                # print(f"  Epochs found: {len(data_list)}")
                # print(f"  Example data (first epoch): {data_list[0]}")
                for entry in data_list:
                    print(f"  Epoch {entry['epoch']}: Loss={entry['avg_loss']:.4f}, MSE_Flow={entry['mse_flow']:.4f}, MSE_Res={entry['mse_res']:.6f}, BPP_M={entry['bpp_m_est']:.4f}, BPP_R={entry['bpp_r_est']:.4f}, Time={entry['time']:.2f}s")
                plot_phase_data(phase_num, data_list, output_dir="training_plots")
            else:
                print(f"  No complete epoch data found for Phase {phase_num}.")
        
        print("\nPlotting complete. Check the 'training_plots' directory.")
    else:
        print("No data parsed from the log file.")