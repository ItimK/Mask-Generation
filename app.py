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

app = Flask(__name__)
CORS(app)

# Create uploads and outputs folders
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

def process_image(image_file):
    """Process image and return alpha and mask versions"""
    # Read image
    input_image = Image.open(image_file)
    
    # Remove background
    output_image = remove(input_image)
    
    # Convert to numpy array to extract alpha channel
    output_array = np.array(output_image)
    
    # Extract alpha channel for B&W mask
    if output_array.shape[2] == 4:  # Has alpha channel
        alpha_channel = output_array[:, :, 3]
    else:
        alpha_channel = np.ones((output_array.shape[0], output_array.shape[1]), dtype=np.uint8) * 255
    
    # Create B&W mask (white = object, black = background)
    mask_image = Image.fromarray(alpha_channel, mode='L')
    
    return output_image, mask_image

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
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
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for idx, file in enumerate(files):
            try:
                print(f"Processing file {idx + 1}/{len(files)}: {file.filename}")
                
                # Process the image
                alpha_image, mask_image = process_image(file)
                
                # Convert to base64 for preview
                alpha_base64 = image_to_base64(alpha_image)
                mask_base64 = image_to_base64(mask_image)
                
                # Save files with timestamp
                filename = os.path.splitext(file.filename)[0]
                
                alpha_path = f'outputs/{filename}_{timestamp}_{idx}_alpha.png'
                mask_path = f'outputs/{filename}_{timestamp}_{idx}_mask.png'
                
                alpha_image.save(alpha_path)
                mask_image.save(mask_path)
                
                results.append({
                    'filename': file.filename,
                    'alpha_preview': alpha_base64,
                    'mask_preview': mask_base64,
                    'alpha_path': alpha_path,
                    'mask_path': mask_path
                })
                
                print(f"✓ Successfully processed: {file.filename}")
                
            except Exception as e:
                print(f"✗ Error processing {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'batch_id': timestamp
        })
    
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

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
    print("\n" + "="*50)
    print("🚀 REMBG Web UI Server Starting...")
    print("="*50)
    print("\n📱 Server starting...")
    print("\n💡 Features:")
    print("   ✅ Upload multiple images at once")
    print("   ✅ Process them all automatically")
    print("   ✅ Download individually or as ZIP")
    print("="*50 + "\n")
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 10000))
    
    # Run on all network interfaces
    app.run(host='0.0.0.0', port=port, debug=False)
```

**Key changes:**

1. ✅ **Removed duplicate Flask app initialization** (you had it twice at the bottom)
2. ✅ **Fixed port binding** - Now reads from `PORT` environment variable (Render requires this)
3. ✅ **Set debug=False** for production

**Also make sure your `templates` folder contains the `index.html` file!**

Your folder structure should be:
```
your-project/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── uploads/ (auto-created)
└── outputs/ (auto-created)
