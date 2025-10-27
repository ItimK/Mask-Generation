from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import numpy as np
import io
import os
import base64
from datetime import datetime
import zipfile
import traceback

app = Flask(__name__)
CORS(app)

# Configure for production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads and outputs folders
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

def process_image(image_file):
    """Process image and return alpha and mask versions"""
    try:
        # Read image
        input_image = Image.open(image_file)
        
        # Resize if too large (to save memory)
        max_size = 2048
        if max(input_image.size) > max_size:
            ratio = max_size / max(input_image.size)
            new_size = tuple(int(dim * ratio) for dim in input_image.size)
            input_image = input_image.resize(new_size, Image.LANCZOS)
            print(f"Resized image to {new_size} to save memory")
        
        # Remove background
        print("Removing background...")
        output_image = remove(input_image)
        print("Background removed successfully")
        
        # Convert to numpy array to extract alpha channel
        output_array = np.array(output_image)
        
        # Extract alpha channel for B&W mask
        if len(output_array.shape) == 3 and output_array.shape[2] == 4:  # Has alpha channel
            alpha_channel = output_array[:, :, 3]
        else:
            alpha_channel = np.ones((output_array.shape[0], output_array.shape[1]), dtype=np.uint8) * 255
        
        # Create B&W mask (white = object, black = background)
        mask_image = Image.fromarray(alpha_channel, mode='L')
        
        return output_image, mask_image
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        print(traceback.format_exc())
        raise

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffered = io.BytesIO()
        # Compress images for base64 preview
        image.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error in image_to_base64: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

@app.route('/process', methods=['POST'])
def process():
    try:
        print("\n=== NEW REQUEST ===")
        print(f"Headers: {dict(request.headers)}")
        print(f"Files in request: {request.files}")
        
        # Try to get files from different possible field names
        files = request.files.getlist('files[]')
        if not files or not files[0].filename:
            files = request.files.getlist('files')
        if not files or not files[0].filename:
            # Fallback to single file
            single_file = request.files.get('file')
            if single_file and single_file.filename:
                files = [single_file]
        
        if not files or not files[0].filename:
            print("ERROR: No files uploaded")
            return jsonify({'error': 'No files uploaded'}), 400
        
        print(f"Processing {len(files)} file(s)")
        
        results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for idx, file in enumerate(files):
            try:
                print(f"\n--- Processing file {idx + 1}/{len(files)}: {file.filename} ---")
                
                # Check file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                print(f"File size: {file_size / 1024 / 1024:.2f} MB")
                
                if file_size > 10 * 1024 * 1024:  # 10MB limit
                    raise Exception(f"File too large: {file_size / 1024 / 1024:.2f} MB. Maximum is 10MB.")
                
                # Process the image
                alpha_image, mask_image = process_image(file)
                
                print("Creating base64 previews...")
                # Convert to base64 for preview
                alpha_base64 = image_to_base64(alpha_image)
                mask_base64 = image_to_base64(mask_image)
                
                print("Saving files...")
                # Save files with timestamp
                filename = os.path.splitext(file.filename)[0]
                
                alpha_path = f'outputs/{filename}_{timestamp}_{idx}_alpha.png'
                mask_path = f'outputs/{filename}_{timestamp}_{idx}_mask.png'
                
                alpha_image.save(alpha_path, optimize=True)
                mask_image.save(mask_path, optimize=True)
                
                results.append({
                    'filename': file.filename,
                    'alpha_preview': alpha_base64,
                    'mask_preview': mask_base64,
                    'alpha_path': alpha_path,
                    'mask_path': mask_path
                })
                
                print(f"✓ Successfully processed: {file.filename}")
                
                # Clean up memory
                del alpha_image
                del mask_image
                
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Error processing {file.filename}: {error_msg}")
                print(traceback.format_exc())
                results.append({
                    'filename': file.filename,
                    'error': error_msg
                })
        
        print(f"\n=== REQUEST COMPLETE: {len(results)} results ===\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'batch_id': timestamp
        })
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n!!! SERVER ERROR !!!")
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/download/<path:filename>')
def download(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/download-batch/<batch_id>')
def download_batch(batch_id):
    """Download all files from a batch as a ZIP"""
    try:
        # Create ZIP file in memory
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Find all files with this batch_id
            for filename in os.listdir('outputs'):
                if batch_id in filename:
                    file_path = os.path.join('outputs', filename)
                    zf.write(file_path, filename)
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'batch_{batch_id}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render provides this as PORT)
    port = int(os.environ.get('PORT', 10000))
    
    print("\n" + "="*50)
    print("🚀 REMBG Background Removal Server")
    print("="*50)
    print(f"Port: {port}")
    print(f"Max file size: 16MB")
    print(f"Max processing size: 2048px")
    print("="*50 + "\n")
    
    # Run server
    app.run(host='0.0.0.0', port=port, debug=False)
