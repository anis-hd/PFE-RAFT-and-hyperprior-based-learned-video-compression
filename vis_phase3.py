import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
log_file_path = r"C:\Users\Anis\Desktop\XX\codec_checkpoints_2phase_visual\training_log_3phase_resAE_vis.txt"
output_dir = "./visualization_phase3"

# --- 1. Create output directory ---
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' ensured.")

# --- 2. Define regular expressions for parsing ---
# Regex to find the epoch number
epoch_regex = re.compile(r"Epoch (\d+)/\d+ Summary")

# Regexes for individual metrics
# Note: Using re.escape for parts of the string that are literal and could contain regex special characters
# (though not strictly necessary for these specific strings, it's good practice).
metrics_regexes = {
    "Avg Loss": re.compile(r"Avg Loss: ([\d\.]+)"),
    "Avg MS-SSIM (opt)": re.compile(r"Avg MS-SSIM \(opt\): ([\d\.]+)"),
    # For lines with two metrics, we'll capture them separately
    "Avg MSE_Flow (mon)": re.compile(r"Avg MSE_Flow \(mon\): ([\d\.]+)"),
    "MSE_Res (opt)": re.compile(r"MSE_Res \(opt\): ([\d\.]+)"), # Note: This is part of the same line as MSE_Flow
    "Avg BPP_M(est,mon)": re.compile(r"Avg BPP_M\(est,mon\): ([\d\.]+)"),
    "BPP_R(est,opt)": re.compile(r"BPP_R\(est,opt\): ([\d\.]+)"), # Note: This is part of the same line as BPP_M
}

# --- 3. Parse the log file ---
# Using defaultdict(list) to easily append to lists of metrics
collected_data = defaultdict(list)
current_epoch_metrics_found = set() # To ensure we only add one value per metric per epoch

print(f"Parsing log file: {log_file_path}")
try:
    with open(log_file_path, 'r') as f:
        current_epoch_value = None
        for line in f:
            # Check for epoch summary line first
            epoch_match = epoch_regex.search(line)
            if epoch_match:
                # New epoch found, store the previous epoch's metrics if any
                # (This logic assumes metrics for an epoch are printed *after* its summary line)
                current_epoch_value = int(epoch_match.group(1))
                collected_data["epoch"].append(current_epoch_value)
                # Reset which metrics we've found for this new epoch
                current_epoch_metrics_found = set()
                # Pad other metric lists if they are shorter than the epoch list
                # This handles cases where some metrics might be missing for an epoch
                # but we still want to plot what we have.
                # However, for this specific log format, metrics seem to always follow.
                # Let's assume for now metrics will be found. If not, lists will be uneven.
                continue # Move to the next line after processing epoch summary

            # If we are within an epoch's data (i.e., current_epoch_value is set)
            # and we have not yet added an epoch number for this block of metrics
            if current_epoch_value is None:
                continue # Skip lines until we find an epoch summary

            # Try to match metric patterns
            # Special handling for lines with two metrics
            if "Avg MSE_Flow (mon)" in line and "MSE_Res (opt)" in line:
                mse_flow_match = metrics_regexes["Avg MSE_Flow (mon)"].search(line)
                mse_res_match = metrics_regexes["MSE_Res (opt)"].search(line)
                if mse_flow_match and "Avg MSE_Flow (mon)" not in current_epoch_metrics_found:
                    collected_data["Avg MSE_Flow (mon)"].append(float(mse_flow_match.group(1)))
                    current_epoch_metrics_found.add("Avg MSE_Flow (mon)")
                if mse_res_match and "MSE_Res (opt)" not in current_epoch_metrics_found:
                    collected_data["MSE_Res (opt)"].append(float(mse_res_match.group(1)))
                    current_epoch_metrics_found.add("MSE_Res (opt)")
                continue # Processed this line

            if "Avg BPP_M(est,mon)" in line and "BPP_R(est,opt)" in line:
                bpp_m_match = metrics_regexes["Avg BPP_M(est,mon)"].search(line)
                bpp_r_match = metrics_regexes["BPP_R(est,opt)"].search(line)
                if bpp_m_match and "Avg BPP_M(est,mon)" not in current_epoch_metrics_found:
                    collected_data["Avg BPP_M(est,mon)"].append(float(bpp_m_match.group(1)))
                    current_epoch_metrics_found.add("Avg BPP_M(est,mon)")
                if bpp_r_match and "BPP_R(est,opt)" not in current_epoch_metrics_found:
                    collected_data["BPP_R(est,opt)"].append(float(bpp_r_match.group(1)))
                    current_epoch_metrics_found.add("BPP_R(est,opt)")
                continue # Processed this line

            # Handle single metrics per line
            for metric_name, regex_pattern in metrics_regexes.items():
                # Skip the composite keys we handled above for individual matching
                if metric_name in ["Avg MSE_Flow (mon)", "MSE_Res (opt)", "Avg BPP_M(est,mon)", "BPP_R(est,opt)"]:
                    # These are either parts of combined lines (handled above) or need to be carefully matched
                    # For this loop, we only want single matchers
                    if metric_name == "Avg MSE_Flow (mon)" and "MSE_Res (opt)" in line: continue
                    if metric_name == "MSE_Res (opt)" and "Avg MSE_Flow (mon)" in line: continue
                    if metric_name == "Avg BPP_M(est,mon)" and "BPP_R(est,opt)" in line: continue
                    if metric_name == "BPP_R(est,opt)" and "Avg BPP_M(est,mon)" in line: continue


                match = regex_pattern.search(line)
                if match and metric_name not in current_epoch_metrics_found:
                    try:
                        value = float(match.group(1))
                        collected_data[metric_name].append(value)
                        current_epoch_metrics_found.add(metric_name)
                        break # Found a metric on this line, move to next line
                    except ValueError:
                        print(f"Warning: Could not convert value '{match.group(1)}' to float for {metric_name} in line: {line.strip()}")
                        # Append a placeholder or skip if conversion fails
                        # For simplicity, we'll skip if conversion fails.
                        # collected_data[metric_name].append(float('nan')) # Or append NaN

except FileNotFoundError:
    print(f"Error: Log file not found at {log_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred during parsing: {e}")
    exit()

# --- 4. Data Validation (Optional but Recommended) ---
if not collected_data["epoch"]:
    print("No epoch data found. Cannot generate plots.")
    exit()

epochs = collected_data.pop("epoch") # Get epochs and remove from metrics dict
num_epochs_found = len(epochs)
print(f"Found data for {num_epochs_found} epochs.")

# Check if all metric lists have the same length as the epochs list
valid_metrics_to_plot = {}
for metric_name, values in collected_data.items():
    if len(values) == num_epochs_found:
        valid_metrics_to_plot[metric_name] = values
    else:
        print(f"Warning: Data mismatch for '{metric_name}'. Expected {num_epochs_found} values, got {len(values)}. This metric will not be plotted.")
        # print(f"Values for {metric_name}: {values}") # for debugging

if not valid_metrics_to_plot:
    print("No valid metrics to plot after data validation.")
    exit()

# --- 5. Generate and Save Plots ---
print("Generating and saving plots...")
for metric_name, values in valid_metrics_to_plot.items():
    if not values: # Skip if for some reason the list is empty
        continue

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, values, marker='o', linestyle='-', markersize=4)
    plt.title(f"{metric_name} vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace('(','').replace(')','').replace(',','')) # Clean up label
    plt.grid(True)
    plt.xticks(epochs[::max(1, len(epochs)//20)]) # Show a reasonable number of x-ticks

    # Sanitize metric_name for filename (remove special chars, replace spaces)
    safe_filename = "".join(c if c.isalnum() else "_" for c in metric_name)
    plot_path = os.path.join(output_dir, f"{safe_filename}.png")

    try:
        plt.savefig(plot_path)
        print(f"  Saved plot: {plot_path}")
    except Exception as e:
        print(f"  Error saving plot {plot_path}: {e}")
    plt.close() # Close the figure to free up memory

print("Visualization complete.")