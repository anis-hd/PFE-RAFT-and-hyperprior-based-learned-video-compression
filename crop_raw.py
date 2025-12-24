import subprocess

def trim_yuv_video(input_path, output_path, width, height, fps, pixel_format='yuv420p', frames_to_keep=50):
    frame_size = {
        'yuv420p': width * height * 3 // 2,
        'yuv422p': width * height * 2,
        'yuv444p': width * height * 3,
    }

    if pixel_format not in frame_size:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")

    # Duration to keep based on FPS and frame count
    duration = frames_to_keep / fps

    cmd = [
        "ffmpeg",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-pix_fmt", pixel_format,
        "-f", "rawvideo",
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        "-f", "rawvideo",
        output_path
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Saved first {frames_to_keep} frames to {output_path}")

# Example usage
trim_yuv_video(
    input_path="./video.yuv",
    output_path="./input.yuv",
    width=1920,              # <-- Set your actual width
    height=1080,             # <-- Set your actual height
    fps=30,                  # or 60 or 120
    pixel_format='yuv420p'   # Adjust based on your input format
)
