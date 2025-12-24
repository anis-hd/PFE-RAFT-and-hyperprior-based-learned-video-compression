import os
import uuid
import json
import subprocess
import re
import itertools
import base64
import io

from flask import Flask, render_template, request, jsonify, Response, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Assuming codec.py is in the same directory and its functions are importable
# For YUV inspection, we need read_yuv_frame_generator.
# If codec.py is strictly a script, this import might fail or need adjustment.
# For this solution, we assume it's possible to import specific utilities.
try:
    from codec import read_yuv_frame_generator
    # PIL (Pillow) is used by codec.py and needed here too for YUV inspection
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Could not import parts of codec.py or dependencies: {e}")
    print("YUV inspection feature might not work if 'read_yuv_frame_generator' or PIL/Numpy is unavailable.")
    read_yuv_frame_generator = None # Fallback

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'outputs')
ALLOWED_EXTENSIONS_VIDEO = {'yuv', 'mp4', 'mov', 'avi'} # Add more if needed for non-YUV input if codec.py supports
ALLOWED_EXTENSIONS_CHECKPOINT = {'pth', 'pt', 'tar', 'ckpt'}
ALLOWED_EXTENSIONS_RDVC = {'rdvc'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create upload and output directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file_storage, subfolder=''):
    if file_storage and file_storage.filename != '':
        original_filename = secure_filename(file_storage.filename)
        # Generate unique filename to prevent overwrites and ensure security
        unique_id = uuid.uuid4().hex
        extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{extension}"
        
        target_folder = os.path.join(app.config['UPLOAD_FOLDER'], subfolder)
        os.makedirs(target_folder, exist_ok=True)
        
        save_path = os.path.join(target_folder, unique_filename)
        file_storage.save(save_path)
        return save_path
    return None

# --- Routes ---
@app.route('/')
def index():
    return render_template('encode.html')

@app.route('/encode', methods=['GET'])
def encode_page():
    return render_template('encode.html')

@app.route('/decode', methods=['GET'])
def decode_page():
    return render_template('decode.html')

@app.route('/inspect_yuv_frames', methods=['POST'])
def inspect_yuv_frames_route():
    if not read_yuv_frame_generator or not 'Image' in globals() or not 'np' in globals():
        return jsonify({'error': 'YUV inspection dependencies not met (codec.py parts, PIL, Numpy).'}), 500

    yuv_file_path = request.form.get('yuv_file_path')
    try:
        width = int(request.form.get('width'))
        height = int(request.form.get('height'))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid width or height for YUV.'}), 400
    
    pixel_format = request.form.get('pixel_format', 'yuv420p') # Default if not provided

    if not yuv_file_path or not os.path.exists(yuv_file_path):
        return jsonify({'error': 'YUV file not found or path not provided.'}), 400
    if not (width > 0 and height > 0):
        return jsonify({'error': 'Width and Height must be positive integers.'}), 400

    try:
        # Get up to 5 frames
        frames_generator = read_yuv_frame_generator(yuv_file_path, width, height, pixel_format)
        frames_rgb_np = list(itertools.islice(frames_generator, 5))
        
        base64_frames = []
        for frame_np_rgb in frames_rgb_np:
            if frame_np_rgb is None: continue
            img_pil = Image.fromarray(frame_np_rgb.astype(np.uint8), 'RGB')
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_frames.append(f"data:image/png;base64,{img_str}")
        
        if not base64_frames:
             return jsonify({'error': 'Could not read any frames from YUV. Check parameters/file.'}), 400

        return jsonify({'frames': base64_frames})
    except Exception as e:
        app.logger.error(f"Error inspecting YUV: {e}", exc_info=True)
        return jsonify({'error': f'Error processing YUV: {str(e)}'}), 500


def stream_process_output(command_args, result_callback):
    process = None
    try:
        # Ensure codec.py is executable and python interpreter is correct
        # Using sys.executable ensures we use the same python interpreter
        # that runs Flask.
        executable_command = [sys.executable, os.path.join(APP_ROOT, 'codec.py')] + command_args
        app.logger.info(f"Executing command: {' '.join(executable_command)}")

        process = subprocess.Popen(
            executable_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1, # Line buffered
            universal_newlines=True, # Ensure text mode
            cwd=APP_ROOT # Run codec.py from the app's root directory
        )

        # Stream stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if not line: break
                yield f"data: {json.dumps({'type': 'log', 'source': 'stdout', 'message': line.strip()})}\n\n"
        
        # Stream stderr (often contains tqdm progress)
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                if not line: break
                yield f"data: {json.dumps({'type': 'log', 'source': 'stderr', 'message': line.strip()})}\n\n"
        
        retcode = process.wait()
        
        if retcode == 0:
            results = result_callback() # This function will get file sizes, paths etc.
            yield f"data: {json.dumps({'type': 'result', 'status': 'success', **results})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': f'Process failed with code {retcode}. Check logs for details.'})}\n\n"

    except FileNotFoundError:
        app.logger.error(f"Error: codec.py or python interpreter not found.")
        yield f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Error: codec.py or python interpreter not found.'})}\n\n"
    except Exception as e:
        app.logger.error(f"Error streaming process output: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'})}\n\n"
    finally:
        if process:
            # Ensure streams are closed
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


@app.route('/start_encode', methods=['POST'])
def start_encode():
    if 'yuv_file' not in request.files or \
       'codec_checkpoint' not in request.files:
        # RAFT checkpoint is optional if codec.py can find one or doesn't need it.
        # For this example, let's make it optional in the form submission.
        return Response(
            f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Missing YUV file or Codec checkpoint.'})}\n\n",
            mimetype='text/event-stream'
        )

    yuv_file = request.files['yuv_file']
    codec_checkpoint_file = request.files['codec_checkpoint']
    raft_checkpoint_file = request.files.get('raft_checkpoint') # Optional

    # YUV parameters
    try:
        yuv_width = int(request.form['yuv_width'])
        yuv_height = int(request.form['yuv_height'])
        yuv_fps = float(request.form['yuv_fps'])
    except (KeyError, ValueError):
        return Response(
            f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Missing or invalid YUV dimensions/FPS.'})}\n\n",
            mimetype='text/event-stream'
        )
    yuv_pixel_format = request.form.get('yuv_pixel_format', 'yuv420p')
    
    # Validate file types (basic check)
    if not (yuv_file.filename != '' and allowed_file(yuv_file.filename, ALLOWED_EXTENSIONS_VIDEO | {'yuv'})): # Allow general video if codec.py supports
        return Response(f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Invalid YUV file type.'})}\n\n", mimetype='text/event-stream')
    if not (codec_checkpoint_file.filename != '' and allowed_file(codec_checkpoint_file.filename, ALLOWED_EXTENSIONS_CHECKPOINT)):
        return Response(f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Invalid Codec checkpoint file type.'})}\n\n", mimetype='text/event-stream')
    if raft_checkpoint_file and raft_checkpoint_file.filename != '' and not allowed_file(raft_checkpoint_file.filename, ALLOWED_EXTENSIONS_CHECKPOINT):
        return Response(f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Invalid RAFT checkpoint file type.'})}\n\n", mimetype='text/event-stream')

    yuv_path = save_uploaded_file(yuv_file, 'encode_inputs')
    codec_checkpoint_path = save_uploaded_file(codec_checkpoint_file, 'encode_inputs')
    raft_checkpoint_path = save_uploaded_file(raft_checkpoint_file, 'encode_inputs') if raft_checkpoint_file else None

    if not yuv_path or not codec_checkpoint_path:
        return Response(
            f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Failed to save uploaded files.'})}\n\n",
            mimetype='text/event-stream'
        )

    output_rdvc_filename = f"{uuid.uuid4().hex}.rdvc"
    output_rdvc_path = os.path.join(app.config['OUTPUT_FOLDER'], output_rdvc_filename)

    command_args = [
        '--mode', 'encode',
        '--input-file', yuv_path,
        '--output-file', output_rdvc_path,
        '--codec-checkpoint', codec_checkpoint_path,
        '--yuv-width', str(yuv_width),
        '--yuv-height', str(yuv_height),
        '--yuv-fps', str(yuv_fps),
        '--yuv-pixel-format', yuv_pixel_format
    ]
    if raft_checkpoint_path:
        command_args.extend(['--raft-checkpoint', raft_checkpoint_path])
    # Add GPU argument if desired, e.g., command_args.extend(['--gpu', '0'])
    # For simplicity, let codec.py use its default GPU handling.

    def get_encode_results():
        yuv_size_bytes = os.path.getsize(yuv_path) if os.path.exists(yuv_path) else 0
        rdvc_size_bytes = os.path.getsize(output_rdvc_path) if os.path.exists(output_rdvc_path) else 0
        compression_ratio = (yuv_size_bytes / rdvc_size_bytes) if rdvc_size_bytes > 0 else 0
        
        return {
            'yuv_file_path_for_inspection': yuv_path, # For inspect frames button if needed after upload
            'output_rdvc_filename': output_rdvc_filename, # Relative to OUTPUT_FOLDER for download link
            'yuv_size': f"{yuv_size_bytes / (1024*1024):.2f} MB" if yuv_size_bytes else "N/A",
            'rdvc_size': f"{rdvc_size_bytes / (1024*1024):.2f} MB" if rdvc_size_bytes else "N/A",
            'compression_ratio': f"{compression_ratio:.2f}" if compression_ratio else "N/A"
        }

    return Response(stream_process_output(command_args, get_encode_results), mimetype='text/event-stream')


@app.route('/start_decode', methods=['POST'])
def start_decode():
    if 'rdvc_file' not in request.files or \
       'codec_checkpoint_decode' not in request.files:
        return Response(
            f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Missing RDVC file or Codec checkpoint.'})}\n\n",
            mimetype='text/event-stream'
        )

    rdvc_file = request.files['rdvc_file']
    codec_checkpoint_file = request.files['codec_checkpoint_decode']

    if not (rdvc_file.filename != '' and allowed_file(rdvc_file.filename, ALLOWED_EXTENSIONS_RDVC)):
         return Response(f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Invalid RDVC file type.'})}\n\n", mimetype='text/event-stream')
    if not (codec_checkpoint_file.filename != '' and allowed_file(codec_checkpoint_file.filename, ALLOWED_EXTENSIONS_CHECKPOINT)):
        return Response(f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Invalid Codec checkpoint file type.'})}\n\n", mimetype='text/event-stream')


    rdvc_path = save_uploaded_file(rdvc_file, 'decode_inputs')
    codec_checkpoint_path = save_uploaded_file(codec_checkpoint_file, 'decode_inputs')

    if not rdvc_path or not codec_checkpoint_path:
        return Response(
            f"data: {json.dumps({'type': 'result', 'status': 'error', 'message': 'Failed to save uploaded files.'})}\n\n",
            mimetype='text/event-stream'
        )

    output_mp4_filename = f"{uuid.uuid4().hex}.mp4"
    output_mp4_path = os.path.join(app.config['OUTPUT_FOLDER'], output_mp4_filename)

    command_args = [
        '--mode', 'decode',
        '--input-file', rdvc_path,
        '--output-file', output_mp4_path,
        '--codec-checkpoint', codec_checkpoint_path
    ]
    # Add GPU argument if desired

    def get_decode_results():
        return {
            'output_mp4_filename': output_mp4_filename # Relative to OUTPUT_FOLDER for download/display
        }

    return Response(stream_process_output(command_args, get_decode_results), mimetype='text/event-stream')


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# For serving the decoded video to the <video> tag
@app.route('/stream_video/<path:filename>')
def stream_video(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    import sys # for sys.executable
    app.run(debug=True, host='0.0.0.0', port=5000)