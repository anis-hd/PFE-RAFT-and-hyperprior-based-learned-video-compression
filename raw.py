import subprocess
import os

def convert_mp4_to_raw_yuv(input_video_path, output_yuv_path, pixel_format="yuv420p"):
    """
    Converts an MP4 video to a raw YUV video file using FFmpeg.

    Args:
        input_video_path (str): Path to the input MP4 video file.
        output_yuv_path (str): Path to save the output raw YUV file.
        pixel_format (str): The YUV pixel format (e.g., "yuv420p", "yuv422p", "yuv444p").
                            "yuv420p" is very common for video.
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video file not found at '{input_video_path}'")
        return False

    # FFmpeg command
    # -i <input_file> : Specifies the input file
    # -c:v rawvideo   : Forces the output video codec to be raw video
    # -pix_fmt <format>: Specifies the pixel format of the raw output
    # -y                : Overwrite output file if it exists without asking
    # <output_file>   : Specifies the output file
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-c:v', 'rawvideo',
        '-pix_fmt', pixel_format,
        '-y',  # Overwrite output without asking
        output_yuv_path
    ]

    print(f"Converting '{input_video_path}' to '{output_yuv_path}' with format '{pixel_format}'...")
    print(f"Executing command: {' '.join(command)}")

    try:
        # Run the command
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("FFmpeg stdout:")
        print(process.stdout)
        print("FFmpeg stderr:") # FFmpeg often prints progress and info to stderr
        print(process.stderr)
        print(f"Conversion successful! Raw YUV video saved to '{output_yuv_path}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: FFmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH.")
        return False

if __name__ == "__main__":
    input_file = './input.mp4'
    output_file = './output.yuv'
    # Common pixel formats:
    # yuv420p: Most common, chroma subsampled 4:2:0 planar
    # yuv422p: Chroma subsampled 4:2:2 planar
    # yuv444p: No chroma subsampling, planar
    # Choose the one you need. yuv420p is a good default.
    desired_pixel_format = "yuv420p"

    # --- Create a dummy input.mp4 for testing if it doesn't exist ---
    if not os.path.exists(input_file):
        print(f"'{input_file}' not found. Creating a dummy MP4 for testing...")
        # This requires FFmpeg to be installed to create the dummy file too.
        try:
            dummy_command = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=320x240:rate=25',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', input_file
            ]
            subprocess.run(dummy_command, check=True, capture_output=True, text=True)
            print(f"Dummy '{input_file}' created successfully.")
        except Exception as e:
            print(f"Could not create dummy '{input_file}'. Please provide your own.")
            print(f"Error: {e}")
            exit()
    # --- End of dummy file creation ---

    if convert_mp4_to_raw_yuv(input_file, output_file, pixel_format=desired_pixel_format):
        print("\nTo play the raw YUV file, you'll need its properties.")
        print("You can get width, height, and framerate from the original MP4 using ffprobe:")
        print(f"  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1:nokey=1 {input_file}")
        print("\nExample ffplay command (replace W, H, FPS with actual values):")
        print(f"  ffplay -f rawvideo -pixel_format {desired_pixel_format} -video_size WxH -framerate FPS {output_file}")
        print(f"  E.g., if 320x240 and 25 FPS: ffplay -f rawvideo -pixel_format {desired_pixel_format} -video_size 320x240 -framerate 25 {output_file}")