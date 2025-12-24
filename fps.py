import numpy as np

def read_yuv_frame(f, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size

    y = f.read(y_size)
    u = f.read(uv_size)
    v = f.read(uv_size)

    if len(y) < y_size or len(u) < uv_size or len(v) < uv_size:
        return None  # End of file

    return y + u + v  # Return raw bytes of one frame

def downsample_yuv_fps(input_path, output_path, width, height, input_fps=120, output_fps=30):
    assert input_fps % output_fps == 0, "Input FPS must be divisible by output FPS"
    skip_ratio = input_fps // output_fps

    with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
        frame_index = 0
        while True:
            frame_data = read_yuv_frame(infile, width, height)
            if frame_data is None:
                break
            if frame_index % skip_ratio == 0:
                outfile.write(frame_data)
            frame_index += 1

    print(f"Done. Written {frame_index // skip_ratio} frames at {output_fps} FPS.")

# Example usage
downsample_yuv_fps(
    input_path='aa.yuv',
    output_path='output.yuv',
    width=1920,
    height=1080,
    input_fps=120,
    output_fps=30
)
